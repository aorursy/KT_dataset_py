!pip install colored --upgrade
import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from colored import fg, bg, attr



sns.set_style("whitegrid")

sns.set_palette("pastel")

%matplotlib inline
basepath = "../input/siim-isic-melanoma-classification/"

train_df = pd.read_csv(basepath + "train.csv")

test_df = pd.read_csv(basepath + "test.csv")



print("Train df shape: %s%s%s%s" % (attr(1), fg(156), str(train_df.shape), attr(0)))

display(train_df.head())

print("Test df shape: %s%s%s%s" % (attr(1), fg(156), str(test_df.shape), attr(0)))

display(test_df.head())
print("%s%s ====== train_df INFO ====== %s" % (attr(1), fg(156), attr(0)))

print(train_df.info())

print("%s%s ====== test_df INFO ====== %s" % (attr(1), fg(156), attr(0)))

print(test_df.info())
missing_train = train_df.isnull().sum()

missing_test = test_df.isnull().sum()



print("%s%sMissing values in %straining data: %s" % (attr(1), fg(197), fg(156), attr(0)))

print(missing_train[missing_train > 0].sort_values(ascending=False))



print("%s%sMissing values in %stest data: %s" % (attr(1), fg(197), fg(156), attr(0)))

print(missing_test[missing_test > 0].sort_values(ascending=False))
print("%sNo. of patients in training data: %s%s%s" % (attr(1), fg(45), train_df.patient_id.nunique(), attr(0)))

print("%sNo. of patients in testing data: %s%s%s" % (attr(1), fg(45), test_df.patient_id.nunique(), attr(0)))
fig, ax = plt.subplots(1,2, figsize=(19,6))



sns.countplot(train_df.sex, ax=ax[0], palette=['dodgerblue','lightpink'], alpha=0.7)

ax[0].set_title("Training data")



sns.countplot(test_df.sex, ax=ax[1], palette=['dodgerblue', 'lightpink'], alpha=0.7)

ax[1].set_title("Test data")



plt.show()
train_patient_count = train_df.groupby('patient_id').image_name.count()

test_patient_count = test_df.groupby('patient_id').image_name.count()



fig, ax = plt.subplots(2,2,figsize=(19,11))



sns.distplot(train_patient_count, bins=30, ax=ax[0,0], color="mediumspringgreen", kde=False)

ax[0,0].set_title("Training Data")

ax[0,0].set_xlabel("")

ax[0,0].set_ylabel("Frequency")



sns.distplot(test_patient_count, bins=30, ax=ax[0,1], color='mediumorchid', kde=False)

ax[0,1].set_title("Test Data")

ax[0,1].set_xlabel("")

ax[0,1].set_ylabel("Frequency")



sns.boxplot(train_patient_count, ax=ax[1,0], color="mediumspringgreen")

ax[1,0].set_xlabel("No. of images")



sns.boxplot(test_patient_count, ax=ax[1,1], color='mediumorchid')

ax[1,1].set_xlabel("No. of images")

plt.show()
train_patient_ages = list(train_df.groupby("patient_id").age_approx.unique())

train_mean_patient_ages = [np.mean(ages) for ages in train_patient_ages]



test_patient_ages = list(test_df.groupby("patient_id").age_approx.unique())

test_mean_patient_ages = [np.mean(ages) for ages in test_patient_ages]



fig, ax = plt.subplots(1,2, figsize=(19,6))



sns.distplot(train_mean_patient_ages, bins=15, ax=ax[0], color='mediumspringgreen')

ax[0].set_title("Age Distribution in Training Data")

ax[0].set_xlabel("Mean age")

ax[0].set_ylabel("Frequency")



sns.distplot(test_mean_patient_ages, bins=15, ax=ax[1], color='mediumorchid')

ax[1].set_title("Age Distribution in Test Data")

ax[1].set_xlabel("Mean age")

ax[1].set_ylabel("Frequency")



plt.show()
fig, ax = plt.subplots(1,2, figsize=(19,6))



train_locations = train_df.anatom_site_general_challenge.value_counts().sort_values(ascending=False)

test_locations = test_df.anatom_site_general_challenge.value_counts().sort_values(ascending=False)



sns.barplot(x=train_locations.index.values, y=train_locations.values, ax=ax[0], color="mediumspringgreen", alpha=0.5);

ax[0].set_xlabel("");

labels = ax[0].get_xticklabels();

ax[0].set_xticklabels(labels, rotation=45);

ax[0].set_title("Image locations in Training data");



sns.barplot(x=test_locations.index.values, y=test_locations.values, ax=ax[1], color="mediumorchid", alpha=0.5);

ax[1].set_xlabel("");

labels = ax[1].get_xticklabels();

ax[1].set_xticklabels(labels, rotation=45);

ax[1].set_title("Image locations in Test data");



plt.show()
train_diagnosis = train_df.diagnosis.value_counts().sort_values(ascending=False)



fig, ax = plt.subplots(1,2, figsize=(19,6))



sns.barplot(x=train_diagnosis.index.values, y=train_diagnosis.values, ax=ax[0], color="mediumspringgreen", alpha=0.5);

ax[0].set_xlabel("");

labels = ax[0].get_xticklabels();

ax[0].set_xticklabels(labels, rotation=45);

ax[0].set_title("Diagnoses' in Training data");



sns.countplot(train_df.benign_malignant, ax=ax[1], palette=['tomato','red'], alpha=0.7)

ax[1].set_title("Type of cancer in Training data")

plt.show()
benign = train_df.groupby('benign_malignant').image_name.count()['benign'] / train_df.shape[0] * 100

malignant = train_df.groupby('benign_malignant').image_name.count()['malignant'] / train_df.shape[0] * 100



print("%sTarget Distribution in Training data%s" % (attr(1), attr(0)))

print("%s%sBenign: %s%s%%%s" % (attr(1), fg(197), fg(156), np.round(benign, 2), attr(0)))

print("%s%sMalignant: %s%s%%%s" % (attr(1), fg(197), fg(156), np.round(malignant, 2), attr(0)))
fig, ax = plt.subplots(1,2, figsize=(19,6))



sns.countplot(x=train_df.age_approx, hue=train_df.sex, palette='Greens', ax=ax[0], alpha=0.7)

ax[0].set_title("Training data")



sns.countplot(x=test_df.age_approx, hue=test_df.sex, palette='Purples', ax=ax[1], alpha=0.7)

ax[1].set_title("Test_data")

plt.show()
fig, ax = plt.subplots(1,2, figsize=(19,6))



sns.boxplot(y=train_df.age_approx, x=train_df.diagnosis, ax=ax[1], palette='PuRd', boxprops=dict(alpha=.7))

ax[1].set_title("Age/Diagnosis in Training data")

ax[1].set_xlabel("");

labels = ax[1].get_xticklabels();

ax[1].set_xticklabels(labels, rotation=45);



sns.boxplot(y=train_df.age_approx, x=train_df.anatom_site_general_challenge, ax=ax[0], palette='GnBu', boxprops=dict(alpha=.7))

ax[0].set_title("Age/Location in Test data")

ax[0].set_xlabel("");

labels = ax[0].get_xticklabels();

ax[0].set_xticklabels(labels, rotation=45);
female_diagnosis = list(train_df.groupby(['sex','diagnosis']).image_name.count()[:7])

z = 0

female_diagnosis = [z]*2 + female_diagnosis



male_diagnosis = list(train_df.groupby(['sex', 'diagnosis']).image_name.count()[7:])



diagnosis = ['atypical melanocytic proliferation', 'cafe-au-lait macule', 'lentigo NOS', 'lichenoid keratosis', 'melanoma', 'nevus', 'seborrheic keratosis', 'solar lentigo', 'unknown']



assert train_df.sex.value_counts()['male'] == np.sum(male_diagnosis)

assert train_df.sex.value_counts()['female'] == np.sum(female_diagnosis)



male_diagnosis = (male_diagnosis / np.sum(male_diagnosis)) * 100

female_diagnosis = (female_diagnosis / np.sum(female_diagnosis)) * 100



print("%s%sMale Diagnosis Statistics%s" % (attr(1), fg(45), attr(0)))

for i in range(len(diagnosis)):

    print("%s%s=> " % (attr(1), fg(14)) + diagnosis[i] + ": %s" % (fg(156)) + str(np.round(male_diagnosis[i],2)) + "%%%s" % (attr(0)))

    

print("%s%s=============================================%s" % (attr(1), fg(250), attr(0)))



print("%s%sFemale Diagnosis Statistics%s" % (attr(1), fg(213), attr(0)))

for i in range(len(diagnosis)):

    print("%s%s=> " % (attr(1), fg(205)) + diagnosis[i] + ": %s" % (fg(156)) + str(np.round(female_diagnosis[i],2)) + "%%%s" % (attr(0)))