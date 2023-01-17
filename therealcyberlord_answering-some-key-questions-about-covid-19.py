import pandas as pd

import matplotlib.pyplot as plt 

import numpy as np 

import seaborn as sns

plt.style.use('fivethirtyeight')
patient_admission = pd.read_csv('../input/uncover/UNCOVER/einstein/diagnosis-of-covid-19-and-its-clinical-spectrum.csv')

patient_admission.head()
patient_admission.describe()
patient_admission.sars_cov_2_exam_result.value_counts()
patient_age = patient_admission[patient_admission.sars_cov_2_exam_result=='positive'].patient_age_quantile

patient_icu = patient_admission[patient_admission.sars_cov_2_exam_result=='positive'].patient_addmited_to_intensive_care_unit_1_yes_0_no
patient_icu.value_counts()
plt.figure(figsize=(10, 5))

patient_icu.hist()

plt.title('ICU Admissions Due to COVID-19')

plt.show()
plt.figure(figsize=(10, 5))

plt.xlabel('Age Quantile')

plt.ylabel('Frequency')

patient_age.hist()

plt.title('Age Quantile Distribution')

plt.show()
plt.figure(figsize=(10, 5))

plt.title('ICU Admission by Age Quantile Due to COVID-19')

plt.xlabel('Age Quantile')

plt.ylabel('Number of Admissions')

patient_icu_age = patient_admission[patient_admission.sars_cov_2_exam_result=='positive'][patient_admission.patient_addmited_to_intensive_care_unit_1_yes_0_no=='t'].patient_age_quantile

patient_icu_age.hist()
hospital_capacity = pd.read_csv('../input/uncover/UNCOVER/harvard_global_health_institute/hospital-capacity-by-state-40-population-contracted.csv')

hospital_capacity
max_icu_bed_occupancy_rate = np.max(hospital_capacity.icu_bed_occupancy_rate)

state_highest_icu_rate = hospital_capacity[hospital_capacity.icu_bed_occupancy_rate==max_icu_bed_occupancy_rate]
state_highest_icu_rate
hospital_capacity['ICU Beds Taken'] = np.round(hospital_capacity.icu_bed_occupancy_rate * hospital_capacity.total_icu_beds)

max_icu_bed_occupancy = np.max(hospital_capacity['ICU Beds Taken'] )

state_highest_icu = hospital_capacity[hospital_capacity['ICU Beds Taken']==max_icu_bed_occupancy]
state_highest_icu