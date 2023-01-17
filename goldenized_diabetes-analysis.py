# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from matplotlib import lines
from scipy import stats

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

dataset.head()
# Add some extra columns which will be useful in the future analyses.
dataset["Total"] = 0
dataset["ID"] = [i + 1 for i in range(len(dataset.Outcome))]
dataset.head()

outcome_summary = dataset.groupby("Outcome").Total.count()
print(outcome_summary)

plt.bar(["Healthy", "Affected"], outcome_summary)
plt.ylabel("Frequency")
plt.title("Number of patients with or without diabetes")
plt.show()
healthy = dataset[dataset.Outcome == 0]
affected = dataset[dataset.Outcome == 1]
plt.figure(figsize=(20, 5))
plt.subplots_adjust(hspace=0.7)

# No. of Pregnancies
healthy_pregnancies = healthy.Pregnancies
affected_pregnancies = affected.Pregnancies

plt.subplot(2, 4, 1)
plt.hist(healthy_pregnancies, alpha=0.5, density=True) # Normalise both hists
plt.hist(affected_pregnancies, alpha=0.5, density=True)

plt.title("Distribution of no. of pregnancies", wrap=True)
plt.legend(["Healthy", "Affected"])
plt.xlabel("Number of Pregnancies")

# Glucose
healthy_glucose = healthy.Glucose[healthy.Glucose != 0] # Remove the = 0 outliers
affected_glucose = affected.Glucose[affected.Glucose != 0]

plt.subplot(2, 4, 2)
plt.hist(healthy_glucose, alpha=0.5, density=True) # Normalise both hists
plt.hist(affected_glucose, alpha=0.5, density=True)

plt.title("Distribution of blood glucose level")
plt.legend(["Healthy", "Affected"])
plt.xlabel("Blood Glucose Level")

# Blood Pressure
healthy_bp = healthy.BloodPressure[healthy.BloodPressure != 0] # Remove the = 0 outliers
affected_bp = affected.BloodPressure[affected.BloodPressure != 0]

plt.subplot(2, 4, 3)
plt.hist(healthy_bp, alpha=0.5, density=True) # Normalise both hists
plt.hist(affected_bp, alpha=0.5, density=True)

plt.title("Distribution of blood pressure")
plt.legend(["Healthy", "Affected"])
plt.xlabel("Blood Pressure")

# Skin Thickness
healthy_st = healthy.SkinThickness[healthy.SkinThickness != 0] # Remove the = 0 outliers
affected_st = affected.SkinThickness[affected.SkinThickness != 0]

plt.subplot(2, 4, 4)
plt.hist(healthy_st, alpha=0.5, density=True) # Normalise both hists
plt.hist(affected_st, alpha=0.5, density=True)

plt.title("Distribution of skin thickness")
plt.legend(["Healthy", "Affected"])
plt.xlabel("Skin Thickness")

# Insulin
healthy_insulin = healthy.Insulin[healthy.Insulin != 0] # Remove the = 0 outliers
affected_insulin = affected.Insulin[affected.Insulin != 0]

plt.subplot(2, 4, 5)
plt.hist(healthy_insulin, alpha=0.5, density=True) # Normalise both hists
plt.hist(affected_insulin, alpha=0.5, density=True)

plt.title("Distribution of insulin level")
plt.legend(["Healthy", "Affected"])
plt.xlabel("Insulin Level")

# BMI
healthy_bmi = healthy.BMI[healthy.BMI != 0] # Remove the = 0 outliers
affected_bmi = affected.BMI[affected.BMI != 0]

plt.subplot(2, 4, 6)
plt.hist(healthy_bmi, alpha=0.5, density=True) # Normalise both hists
plt.hist(affected_bmi, alpha=0.5, density=True)

plt.title("Distribution of BMI values")
plt.legend(["Healthy", "Affected"])
plt.xlabel("BMI Value")

# Diabetes Pedigree Function (DPF)
healthy_dpf = healthy.DiabetesPedigreeFunction[healthy.DiabetesPedigreeFunction != 0]
affected_dpf = affected.DiabetesPedigreeFunction[affected.DiabetesPedigreeFunction != 0]

plt.subplot(2, 4, 7)
plt.hist(healthy_dpf, alpha=0.5, density=True) # Normalise both hists
plt.hist(affected_dpf, alpha=0.5, density=True)

plt.title("Distribution of DPF values")
plt.legend(["Healthy", "Affected"])
plt.xlabel("DPF Value")

# Age
healthy_age = healthy.Age[healthy.Age != 0]
affected_age = affected.Age[affected.Age != 0]

plt.subplot(2, 4, 8)
plt.hist(healthy_age, alpha=0.5, density=True) # Normalise both hists
plt.hist(affected_age, alpha=0.5, density=True)

plt.title("Distribution of ages")
plt.legend(["Healthy", "Affected"])
plt.xlabel("Age")

plt.show()
plt.figure(figsize=(20, 5))

# _id: Filter out all the = 0 outliers, but each patient still has the unique ID assigned to them (Allows for merging).
healthy_st_id = healthy[healthy.SkinThickness != 0]
healthy_bmi_id = healthy[healthy.BMI != 0]
healthy_st_bmi_id = pd.merge(healthy_st_id, healthy_bmi_id)

affected_st_id = affected[affected.SkinThickness != 0]
affected_bmi_id = affected[affected.BMI != 0]
affected_st_bmi_id = pd.merge(affected_st_id, affected_bmi_id)
# pd.merge the st and bmi so we get same no. of shit for scatter, then access it

plt.subplot(1, 3, 1)
plt.scatter(healthy_st_bmi_id['SkinThickness'], healthy_st_bmi_id['BMI'], alpha=0.5)
plt.scatter(affected_st_bmi_id['SkinThickness'], affected_st_bmi_id['BMI'], alpha=0.5)
plt.legend(["Healthy", "Affected"])
plt.xlabel("Skin Thickness")
plt.ylabel("BMI")
plt.title("BMI against Skin Thickness")


# merge them both and get the medians
all_st_bmi_id = pd.merge(healthy_st_bmi_id, affected_st_bmi_id, how='outer')

# medians
st_median = all_st_bmi_id.SkinThickness.median()
bmi_median = all_st_bmi_id.BMI.median()
plt.plot([st_median, st_median], [0, 70], linewidth=1, color='black')
plt.plot([0, 100], [bmi_median, bmi_median], linewidth=1, color='black')

# annotations
plt.annotate("High ST, High BMI", (70, 55))
plt.annotate("Quadrant I", (75, 50))

plt.annotate("High ST, Low BMI", (50, 20))
plt.annotate("Quadrant II", (55, 15))

plt.annotate("Low ST, Low BMI", (-3, 10))
plt.annotate("Quadrant III", (0, 5))

plt.annotate("Low ST, High BMI", (-3, 60))
plt.annotate("Quadrant IV", (0, 55))

# obtain dataframes of interest: logic
low_st_high_bmi = all_st_bmi_id[(all_st_bmi_id.SkinThickness < st_median) & (all_st_bmi_id.BMI > bmi_median)]
low_st_high_bmi_outcome = low_st_high_bmi.groupby("Outcome").Total.count()

high_st_low_bmi = all_st_bmi_id[(all_st_bmi_id.SkinThickness > st_median) & (all_st_bmi_id.BMI < bmi_median)]
high_st_low_bmi_outcome = high_st_low_bmi.groupby("Outcome").Total.count()

high_st_high_bmi = all_st_bmi_id[(all_st_bmi_id.SkinThickness > st_median) & (all_st_bmi_id.BMI > bmi_median)]
high_st_high_bmi_outcome = high_st_high_bmi.groupby("Outcome").Total.count()

low_st_low_bmi = all_st_bmi_id[(all_st_bmi_id.SkinThickness < st_median) & (all_st_bmi_id.BMI < bmi_median)]
low_st_low_bmi_outcome = low_st_low_bmi.groupby("Outcome").Total.count()

# plot pie charts
plt.subplot(2, 3, 2)
plt.pie(high_st_high_bmi_outcome, autopct='%0.1f%%')
plt.title("Quadrant I")
plt.legend(["Healthy", "Affected"])
plt.axis('equal')
plt.annotate("Total: {}".format(high_st_high_bmi_outcome.sum()), (1.5, 0))

plt.subplot(2, 3, 3)
plt.pie(high_st_low_bmi_outcome, autopct='%0.1f%%')
plt.title("Quadrant II")
plt.legend(["Healthy", "Affected"])
plt.axis('equal')
plt.annotate("Total: {}".format(high_st_low_bmi_outcome.sum()), (1.5, 0))

plt.subplot(2, 3, 5)
plt.pie(low_st_low_bmi_outcome, autopct='%0.1f%%')
plt.title("Quadrant III")
plt.legend(["Healthy", "Affected"])
plt.axis('equal')
plt.annotate("Total: {}".format(low_st_low_bmi_outcome.sum()), (1.5, 0))

plt.subplot(2, 3, 6)
plt.pie(low_st_high_bmi_outcome, autopct='%0.1f%%')
plt.title("Quadrant IV")
plt.legend(["Healthy", "Affected"])
plt.axis('equal')
plt.annotate("Total: {}".format(low_st_high_bmi_outcome.sum()), (1.5, 0))

plt.show()
plt.figure(figsize=(20, 5))

# _id: Filter out all the = 0 outliers, but each patient still has the unique ID assigned to them (Allows for merging).
healthy_pregnancies_id = healthy
healthy_age_id = healthy[healthy.Age != 0]
healthy_pregnancies_age_id = pd.merge(healthy_pregnancies_id, healthy_age_id)


affected_pregnancies_id = affected
affected_age_id = affected[affected.Age != 0]
affected_pregnancies_age_id = pd.merge(affected_pregnancies_id, affected_age_id)
# pd.merge the st and bmi so we get same no. of shit for scatter, then access it

ax = plt.subplot(1, 3, 1)
plt.scatter(healthy_pregnancies_age_id['Pregnancies'], healthy_pregnancies_age_id['Age'], alpha=0.5)
plt.scatter(affected_pregnancies_age_id['Pregnancies'], affected_pregnancies_age_id['Age'], alpha=0.5)
plt.legend(["Healthy", "Affected"])
plt.xlabel("No. of Pregnancies")
plt.ylabel("Age")
plt.title("No. of Pregnancies against Age")
ax.set_xticks(range(0, 20, 2))

# merge them both and get the medians
all_pregnancies_age_id = pd.merge(healthy_pregnancies_age_id, affected_pregnancies_age_id, how='outer')

# medians
pregnancies_median = all_pregnancies_age_id.Pregnancies.median()
age_median = all_pregnancies_age_id.Age.median()
plt.plot([pregnancies_median, pregnancies_median], [20, 80], linewidth=1, color='black')
plt.plot([0, 17], [age_median, age_median], linewidth=1, color='black')

# obtain dataframes of interest: logic
low_pregnancies_high_age = all_pregnancies_age_id[(all_pregnancies_age_id.Pregnancies < pregnancies_median) & 
                                                  (all_pregnancies_age_id.Age > age_median)]
low_pregnancies_high_age_outcome = low_pregnancies_high_age.groupby("Outcome").Total.count()


high_pregnancies_low_age = all_pregnancies_age_id[(all_pregnancies_age_id.Pregnancies > pregnancies_median) & 
                                                  (all_pregnancies_age_id.Age < age_median)]
high_pregnancies_low_age_outcome = high_pregnancies_low_age.groupby("Outcome").Total.count()


high_pregnancies_high_age = all_pregnancies_age_id[(all_pregnancies_age_id.Pregnancies > pregnancies_median) & 
                                                   (all_pregnancies_age_id.Age > age_median)]
high_pregnancies_high_age_outcome = high_pregnancies_high_age.groupby("Outcome").Total.count()


low_pregnancies_low_age = all_pregnancies_age_id[(all_pregnancies_age_id.Pregnancies < pregnancies_median) &
                                                 (all_pregnancies_age_id.Age < age_median)]
low_pregnancies_low_age_outcome = low_pregnancies_low_age.groupby("Outcome").Total.count()

# plot pie charts
plt.subplot(2, 3, 2)
plt.pie(high_pregnancies_high_age_outcome, autopct='%0.1f%%')
plt.title("Quadrant I")
plt.legend(["Healthy", "Affected"])
plt.axis('equal')
plt.annotate("Total: {}".format(high_pregnancies_high_age_outcome.sum()), (1.5, 0))

plt.subplot(2, 3, 3)
plt.pie(high_pregnancies_low_age_outcome, autopct='%0.1f%%')
plt.title("Quadrant II")
plt.legend(["Healthy", "Affected"])
plt.axis('equal')
plt.annotate("Total: {}".format(high_pregnancies_low_age_outcome.sum()), (1.5, 0))

plt.subplot(2, 3, 5)
plt.pie(low_pregnancies_low_age_outcome, autopct='%0.1f%%')
plt.title("Quadrant III")
plt.legend(["Healthy", "Affected"])
plt.axis('equal')
plt.annotate("Total: {}".format(low_pregnancies_low_age_outcome.sum()), (1.5, 0))

plt.subplot(2, 3, 6)
plt.pie(low_pregnancies_high_age_outcome, autopct='%0.1f%%')
plt.title("Quadrant IV")
plt.legend(["Healthy", "Affected"])
plt.axis('equal')
plt.annotate("Total: {}".format(low_pregnancies_high_age_outcome.sum()), (1.5, 0))


plt.show()
plt.figure(figsize=(20, 5))

# _id: Filter out all the = 0 outliers, but each patient still has the unique ID assigned to them (Allows for merging).
healthy_glucose_id = healthy[healthy.Glucose != 0]
healthy_bp_id = healthy[healthy.BloodPressure != 0]
healthy_glucose_bp_id = pd.merge(healthy_glucose_id, healthy_bp_id)

affected_glucose_id = affected[affected.Glucose != 0]
affected_bp_id = affected[affected.BloodPressure != 0]
affected_glucose_bp_id = pd.merge(affected_glucose_id, affected_bp_id)
# pd.merge the st and bmi so we get same no. of shit for scatter, then access it

plt.subplot(1, 3, 1)
plt.scatter(healthy_glucose_bp_id['Glucose'], healthy_glucose_bp_id['BloodPressure'], alpha=0.5)
plt.scatter(affected_glucose_bp_id['Glucose'], affected_glucose_bp_id['BloodPressure'], alpha=0.5)
plt.legend(["Healthy", "Affected"])
plt.xlabel("Blood glucose level")
plt.ylabel("Blood pressure level")
plt.title("Blood Pressure against Blood Glucose")

# merge them both and get the medians
all_glucose_bp_id = pd.merge(healthy_glucose_bp_id, affected_glucose_bp_id, how='outer')

# medians
glucose_median = all_glucose_bp_id.Glucose.median()
bp_median = all_glucose_bp_id.BloodPressure.median()
plt.plot([glucose_median, glucose_median], [20, 120], linewidth=1, color='black')
plt.plot([40, 200], [bp_median, bp_median], linewidth=1, color='black')

# obtain dataframes of interest: logic
low_glucose_high_bp = all_glucose_bp_id[(all_glucose_bp_id.Glucose < glucose_median) & 
                                        (all_glucose_bp_id.BloodPressure > bp_median)]
low_glucose_high_bp_outcome = low_glucose_high_bp.groupby("Outcome").Total.count()


high_glucose_low_bp = all_glucose_bp_id[(all_glucose_bp_id.Glucose > glucose_median) & 
                                        (all_glucose_bp_id.BloodPressure < bp_median)]
high_glucose_low_bp_outcome = high_glucose_low_bp.groupby("Outcome").Total.count()


high_glucose_high_bp = all_glucose_bp_id[(all_glucose_bp_id.Glucose > glucose_median) & 
                                     (all_glucose_bp_id.BloodPressure > bp_median)]
high_glucose_high_bp_outcome = high_glucose_high_bp.groupby("Outcome").Total.count()


low_glucose_low_bp = all_glucose_bp_id[(all_glucose_bp_id.Glucose < glucose_median) & 
                                   (all_glucose_bp_id.BloodPressure < bp_median)]
low_glucose_low_bp_outcome = low_glucose_low_bp.groupby("Outcome").Total.count()


# plot pie charts
plt.subplot(2, 3, 2)
plt.pie(high_glucose_high_bp_outcome, autopct='%0.1f%%')
plt.title("Quadrant I")
plt.legend(["Healthy", "Affected"])
plt.axis('equal')
plt.annotate("Total: {}".format(high_glucose_high_bp_outcome.sum()), (1.5, 0))

plt.subplot(2, 3, 3)
plt.pie(high_glucose_low_bp_outcome, autopct='%0.1f%%')
plt.title("Quadrant II")
plt.legend(["Healthy", "Affected"])
plt.axis('equal')
plt.annotate("Total: {}".format(high_glucose_low_bp_outcome.sum()), (1.5, 0))

plt.subplot(2, 3, 5)
plt.pie(low_glucose_low_bp_outcome, autopct='%0.1f%%')
plt.title("Quadrant III")
plt.legend(["Healthy", "Affected"])
plt.axis('equal')
plt.annotate("Total: {}".format(low_glucose_low_bp_outcome.sum()), (1.5, 0))

plt.subplot(2, 3, 6)
plt.pie(low_glucose_high_bp_outcome, autopct='%0.1f%%')
plt.title("Quadrant IV")
plt.legend(["Healthy", "Affected"])
plt.axis('equal')
plt.annotate("Total: {}".format(low_glucose_high_bp_outcome.sum()), (1.5, 0))

plt.show()
st_tstat, st_p = stats.ranksums(healthy_st, affected_st)
bmi_tstat, bmi_p = stats.ranksums(healthy_bmi, affected_bmi)
age_tstat, age_p = stats.ranksums(healthy_age, affected_age)
glucose_tstat, glucose_p = stats.ranksums(healthy_glucose, affected_glucose)

print(st_p, bmi_p, age_p, glucose_p)