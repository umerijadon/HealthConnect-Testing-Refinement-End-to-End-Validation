# HealthConnect-Testing-Refinement-End-to-End-Validation

# HealthConnect Clinic: Predicting Patient No-Shows

**Data Science Track | AnalystLab Africa Experience Lab | Week 7: Model Testing, Error Analysis & Refinement**

> How can HealthConnect Clinic use appointment data to identify patients who may miss their appointments and give the clinic a chance to intervene?

This repository contains my Data Science track work for the HealthConnect project, a fictional clinic created as part of the AnalystLab Africa Experience Lab.

Week 7 was less about building another model and more about testing the one we already had. I looked at whether individual features were actually helping, whether the model was overfitting, how performance varied across patient groups, and whether the default 0.5 classification threshold made sense for the clinic's use case.

---

## Project Snapshot

|                       |                                                                                                                                                                              |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Problem**           | Binary classification: predict whether an appointment will result in a no-show (`is_no_show`)                                                                                |
| **Data**              | 5,000 appointments and 18 raw columns covering demographics, booking details, reminders, distance, prior history, and outcomes. Cancellations were excluded from the target. |
| **Model**             | Logistic Regression with `class_weight='balanced'`, wrapped in a scikit-learn `Pipeline` with one-hot encoding                                                               |
| **Validation**        | 5-fold `GroupKFold` using `patient_id`, ensuring the same patient never appears in both training and test folds                                                              |
| **Week 6 CV result**  | Accuracy: **0.629** vs **0.624** for the Week 5 baseline; ROC-AUC: **0.680** for both                                                                                        |
| **Working threshold** | **0.45**, pending information about the cost of the clinic's intervention                                                                                                    |

---

## What I Tested in Week 7

This week, I focused on a few questions that matter when moving from a working model to something that could actually be used.

| Question                                                       | Method                                              | Finding                                                                                                                                                                                                                                 |
| -------------------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Does `lead_time_bin` add anything beyond raw lead time?        | CV ablation: raw days vs. raw + bins vs. bins only  | **No meaningful lift.** Raw `booking_lead_days` performed almost identically: 0.6276 vs. 0.6274 accuracy and 0.6803 vs. 0.6802 ROC-AUC.                                                                                                 |
| Is the model overfitting?                                      | Train/test comparison using a grouped 80/20 split   | **No obvious overfitting.** The accuracy gap was 0.35 percentage points and the ROC-AUC gap was 1.3 points. Test performance: 63.5% accuracy and 0.679 ROC-AUC.                                                                         |
| Does performance vary across groups?                           | Accuracy by appointment type, gender, and age group | **Yes.** Diagnostic Tests reached 71.6% accuracy compared with 58.3% for Specialist Consultations. Ages 55–64 reached 68.1% compared with 59.2% for patients aged 65+. Segment sizes are small, so some of this variation may be noise. |
| Is 0.5 the right classification threshold?                     | Threshold sweep from 0.40 to 0.65                   | **It depends on the cost of intervention.**                                                                                                                                                                                             |
| Do the interaction features from Data Analytics still hold up? | Coefficient check after model refinement            | `prevnoshow_x_reminder` retained some signal (coef 0.152), while `distance_x_no_reminder` was close to zero (coef -0.019).                                                                                                              |

---

## Threshold Trade-off

The threshold controls how easily the model flags an appointment as a potential no-show.

| Threshold |  Accuracy | Precision |    Recall |
| --------: | --------: | --------: | --------: |
|      0.40 |     0.620 |     0.588 |     0.801 |
|  **0.45** | **0.633** | **0.613** | **0.718** |
|      0.50 |     0.635 |     0.637 |     0.625 |
|      0.55 |     0.623 |     0.655 |     0.522 |
|      0.60 |     0.603 |     0.666 |     0.412 |
|      0.65 |     0.589 |     0.709 |     0.302 |

A lower threshold catches more potential no-shows, but also means more patients will be flagged.

For a relatively low-cost intervention such as an additional reminder call, that trade-off may justify testing a threshold around **0.40–0.45**. However, the final threshold should be chosen using the actual cost of a missed appointment and the cost of the intervention.

For now, **0.45 is treated as a working threshold rather than a final recommendation**.

---

## Key Findings

### 1. A useful pattern does not always make a useful feature

The lead-time bands reflected a real pattern in the data, but adding those bins did not meaningfully improve the model because Logistic Regression was already getting similar information from the raw `booking_lead_days` feature.

I kept the bins because they make the result easier to explain — for example, saying that risk increases beyond 45 days is easier to communicate than explaining a continuous coefficient.

So this was an **interpretability decision, not a performance improvement**.

### 2. The model does not show obvious overfitting, but performance is not consistent across every group

The train/test results were close, which is reassuring.

However, performance varied across appointment types and age groups. Specialist Consultations and patients aged 65+ had lower accuracy than some other groups.

The current dataset is not large enough to confidently explain those differences, so I am treating them as areas that need further investigation rather than drawing conclusions from them.

### 3. The Data Analytics findings did not all survive model testing

The Data Analytics track identified two potentially useful interactions through their Week 6 crosstab analysis:

* Previous no-shows × reminder
* Distance × reminder

I tested both again after refining the model.

`prevnoshow_x_reminder` retained some predictive weight, while `distance_x_no_reminder` was close to zero.

That result was shared back with the Data Analytics track rather than assuming that every exploratory relationship would translate into predictive value.

---

## Model Suitability

Based on the current results, this model is better suited to **low-cost, low-stakes interventions**, such as an additional reminder call, than to decisions requiring high confidence about an individual patient.

The current performance is around 63% accuracy with a ROC-AUC of about 0.68, so this should be treated as a **decision-support tool**, not a definitive predictor of whether a particular patient will attend.

Predictions for Specialist Consultations and patients aged 65+ also deserve additional caution because performance was lower in those groups.

---

## Cross-Track Collaboration

The HealthConnect project is being developed across several tracks, so part of my work involved testing findings produced by other teams.

| Track              | Dependency                                                        | What I did                                                                                                                                                                             |
| ------------------ | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data Analytics** | Two interaction features came from their Week 6 crosstab analysis | Re-tested both after model refinement. One retained predictive weight; the other did not. `distance_x_no_reminder` was downgraded and the finding was shared with the analytics track. |
| **ML Engineering** | Needs a stable feature list and preprocessing process             | Documented the final feature set and kept the classification threshold configurable rather than hard-coding 0.5.                                                                       |

---

## Final Feature Set

The following features were handed off to ML Engineering.

**Numeric**

`age`, `previous_appointments`, `previous_no_shows`, `distance_to_clinic_km`, `distance_missing_flag`, `no_show_rate_history`, `is_first_appointment`, `is_weekend_appt`, `prevnoshow_x_reminder`, `high_risk_no_reminder`, `distance_x_no_reminder`, `booking_lead_days`

**Categorical**

`gender`, `appointment_type`, `appointment_day`, `appointment_time`, `reminder_sent`, `reminder_channel`, `lead_time_bin`

Categorical variables were one-hot encoded with the first category dropped.

---

## Limitations & Open Questions

There are still several things I would want to investigate before treating this as anything more than a prototype:

* Performance gaps for Specialist Consultations and patients aged 65+ have not yet been explained.
* Only the lead-time feature has gone through a formal ablation test. A full one-feature-at-a-time analysis is still needed.
* The final threshold depends on intervention-cost information that is not currently available.
* `distance_x_no_reminder` remains in the feature set but contributes very little.
* Day-of-week effects, which appeared noisy during Week 5, are still unresolved.
* Overall accuracy is around 63%, so this should be treated as a decision-support signal rather than a definitive predictor.

---

## Before Week 8

1. Get intervention-cost information to help choose the final classification threshold.
2. Run a full one-feature-at-a-time ablation across the feature set.
3. Confirm that the feature list handed to ML Engineering matches their implementation pipeline.

---

## Tech Stack

**Python · pandas · NumPy · scikit-learn · Matplotlib · Jupyter**

---

## About the Project

This project was completed as part of the **AnalystLab Africa Experience Lab Internship Programme**, Data Science track.

HealthConnect Clinic is a fictional healthcare provider created for the programme, and the dataset is provided for project and learning purposes.

`#AnalystLabAfrica`
