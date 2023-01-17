# import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# import data

df = pd.read_csv('../input/heart.csv')

df.head()
df.columns
df.rename(columns={'cp':'chest_pain_type', 'trestbps': 'resting_blood_pressure', 

                   'chol':'serum_cholesterol', 'fbs':'fasting_blood_sugar', 

                   'restecg':'resting_electroc_results', 'thalach':'max_heart_rate_achieved', 

                   'exang':'exercise_induced_angina', 'ca':'no_of_mjrv_cf'}, inplace=True)
df.head()
# shape of the dataset

m = df.shape

m
df.describe()
# patients with or without heart disease

heartdisease = len(df[df['target'] == 1])

noheartdisease = len(df[df['target'] == 0])

print(f"Number of patients with heart disease is: {heartdisease}")

print(f"Number of patients without heart disease is: {noheartdisease}")



no_of_patients = m[0]

percentage_with_hd = (heartdisease / m[0]) * 100

percentage_without_hd = (noheartdisease / m[0]) * 100



print(f"Percentage of patients with heart disease is: {percentage_with_hd:.2f}% in 2 decimal places")

print(f"Percentage of patients without heart disease is: {percentage_without_hd:.2f}% in 2 decimal places")
# 0 - no heart disease 1 - heart disease

# Visualization of the number of patients with or without heart diseases

sns.countplot(y="target", data=df, color="b")
# Count of male and female

male_count = len(df[df['sex'] == 1])

female_count = len(df[df['sex'] == 0])



print(f"Total number of men in the dataset is: {male_count}")

print(f"Total number of women in the dataset is: {female_count}")
# Visualization of the number of men and women in the data sample

# 1 - Male, 0 - Female

sns.countplot(y='sex', data=df, color='c')
# Male Patients with heart disease

mp_with_hd_count = len(df[(df['target'] == 1) & (df['sex'] == 1)])

print(f"Number of male Patients with heart disease is: {mp_with_hd_count}")

# Female patients with heart disease

print(f"Number of Female Patients with heart disease is: {heartdisease - mp_with_hd_count}")



# Male patients without heart disease

mp_without_hd_count = len(df[(df['target'] == 0) & (df['sex'] == 1)])

print(f"Number of Male Patients without heart disease is: {mp_without_hd_count}")

# Female patients without heart disease

print(f"Number of Female Patients without heart disease is: {noheartdisease - mp_without_hd_count}")
# Percentage of Men with heart disease

print(f"Percentage of male patients with heart disease is: {(mp_with_hd_count / male_count) * 100:.2f}%")

print(f"Percentage of male patients without heart disease is: {100 - ((mp_with_hd_count / male_count) * 100):.2f}%")



print(f"Percentage of female patients with heart disease is: {((heartdisease - mp_with_hd_count) / female_count) * 100:.2f}%")

print(f"Percentage of female patients without heart disease is: {100 - (((heartdisease - mp_with_hd_count) / female_count) * 100):.2f}%")
# Visualization to showcase the number of male/female with/without heart disease

# 0 - no heart disease 1 - heart disease

sns.catplot(y="target", hue="sex", kind="count",

            palette="pastel", edgecolor=".6",

            data=df, legend=False)

plt.legend(title='Sex', loc='upper right', labels=['0 - Female', '1 - Male'])
# Male patients with heart disease above 65

# According to my research, most occurence of heart disease for men is roughly at the age of 65, lets see if that

# theory holds for this data

male_65_and_above = df[(df['sex'] == 1) & (df['age'] >= 65)]

count_male_65_and_above = len(male_65_and_above)

print(f"The number of male patients above 65 is: {count_male_65_and_above}")



male_hd_above65 = df[(df['sex'] == 1) & (df['age'] >= 65) & (df['target'] == 1)]

count_male_hd_above65 = len(male_hd_above65)



print(f"The number of male patients with heart disease and above 65 is: {count_male_hd_above65}")



# Female patients with heart disease above 72

# According to my research, most occurence of heart disease for women is roughly at the age of 72, lets see if that

# theory holds for this data

female_72_and_above = df[(df['sex'] == 0) & (df['age'] >= 72)]

count_female_72_and_above = len(female_72_and_above)

print(f"The number of female patients above 72 is: {count_female_72_and_above}")



female_hd_above72 = df[(df['sex'] == 0) & (df['age'] >= 72) & (df['target'] == 1)]

count_female_hd_above72 = len(female_hd_above72)



print(f"The number of female patients with heart disease and above 72 is: {count_female_hd_above72}")



print(f"{(count_male_hd_above65 / count_male_65_and_above) * 100:.2f}% of the total number of male patients with ages greater than or equal to 65 appear to have a heart disease")

print(f"{(count_female_hd_above72 / count_female_72_and_above) * 100:.2f}% of the total number of female patients with ages greater than or equal to 72 appear to have a heart disease")
bins = [i for i in range(9, 100, 10)]

df['age_range'] = pd.cut(df['age'], bins)

df.head()
df['age_range'].value_counts()
x = df.groupby(['target', 'sex'])['age_range'].value_counts()

x
x.plot(kind='barh', figsize=(12, 12))
df.head()
df['chest_pain_type'].unique()
plt.hist(df['chest_pain_type'])
cpt = df.groupby('target')['chest_pain_type'].value_counts()

cpt
cpt.plot(kind='barh', color='g')
sns.catplot(x="sex", y="target", hue="chest_pain_type", kind="bar", data=df)
cptts = df.groupby(['target', 'sex'])['chest_pain_type'].value_counts()

cptts
cptts.plot(kind='barh', figsize=(12, 12), color='y')
sns.catplot(x="sex", y="chest_pain_type", hue="target", kind="swarm", data=df, height=11, aspect=11.7/8.27)
# Check the resting blood pressure 

sns.catplot(x="target", y="resting_blood_pressure", data=df)
bins = [i for i in range(90, 210, 9)]

df['rbp_range'] = pd.cut(df['resting_blood_pressure'], bins)

df.head()
df['rbp_range'].value_counts()
rbps = df.groupby(['target', 'sex'])['rbp_range'].value_counts()

rbps
rbps.plot(kind='barh', figsize=(12, 12), color='y')
# Scatter plot showing the correlation between the age and the resting blood pressure based on the target

sns.relplot(x="age", y="resting_blood_pressure", hue="target", data=df)
sns.relplot(x="age", y="resting_blood_pressure", hue="chest_pain_type", palette="ch:r=-.5,l=.75", data=df)
sns.relplot(x="age", y="resting_blood_pressure", hue="sex", data=df)
sns.catplot(x="rbp_range", y="sex", hue="target", kind="swarm", data=df, height=11, aspect=11.7/8.27)
bins = [i for i in range(125, 600, 75)]

df['serum_normal'] = pd.cut(df['serum_cholesterol'], bins)

df.head()
df.tail()
# Normal serum cholesterol range for both male and female should be between 125mmg/dl to 200mmg/dl

df.groupby(['target','sex'])['serum_normal'].value_counts()
df.groupby(['target','sex'])['serum_normal'].value_counts().plot(kind='bar', figsize=(12, 12))
# Number of patients with serum_cholesterol greater than 200mmg/dl

serum_chol_gr8_than_200 = df[df['serum_cholesterol'] > 200]

len(serum_chol_gr8_than_200)
serum_chol_gr8_than_200.head()
count_s = serum_chol_gr8_than_200.groupby('target')['serum_normal'].count()

count_s
print(f"{(count_s[1]/len(serum_chol_gr8_than_200)) * 100:.2f}% of patients with serum cholesterol greater than 200 have a heart disease")
df['serum_chol'] = ((df['serum_cholesterol']>125) & (df['serum_cholesterol']<200)).astype('int')

df.head()
df.groupby(['target', 'serum_chol'])['rbp_range'].value_counts().plot(kind='barh', figsize=(12, 12), color='black')
df.groupby(['target', 'serum_chol'])['age_range'].value_counts().plot(kind='barh', figsize=(12, 12), color='g')
df.isnull().sum()
# Fasting blood sugar

df.groupby(['target', 'sex'])['fasting_blood_sugar'].value_counts()
df.groupby(['target', 'sex'])['resting_electroc_results'].value_counts()
df.groupby(['target', 'sex', 'age_range'])['resting_electroc_results'].value_counts()
# Maximum heart rate 

df['approp_max_heart_rate'] = 220 - df['age']

df['target_heart_rate'] = ["({0:.2f} - {1:.2f}]".format(0.6 * i, 0.8 * i) for i in df['approp_max_heart_rate']]
df.head()
# Maximum heart rate at 85%

df['peak_max_heart_rate'] = 0.85 * df['approp_max_heart_rate']

df.head()
# Maximum heart rate achieved greater than 85% of the maximum heart rate 

# 1 - achieved more than the 85% of the appropriate heart rate 0 - falls in the appropriate range

df['max_heart_rate_danger'] = ((df['max_heart_rate_achieved']>df['peak_max_heart_rate'])).astype('int')
df.head()
df.groupby(['target', 'sex'])['max_heart_rate_danger'].value_counts()
df.groupby(['target', 'sex'])['max_heart_rate_danger'].value_counts().plot(kind='bar', figsize=(12, 12))
df.groupby('target')['max_heart_rate_danger'].value_counts().plot(kind='bar', figsize=(12, 12))
sns.relplot(x="age", y="max_heart_rate_achieved", hue="target", data=df)
sns.relplot(x="max_heart_rate_achieved", y="max_heart_rate_danger", hue="target", data=df, height=11, aspect=11.7/8.27)
sns.relplot(x="sex", y="max_heart_rate_achieved", hue="target", data=df, height=11, aspect=11.7/8.27)
# Exercise Induced Angina

df.head()
eind = df.groupby('target')['exercise_induced_angina'].value_counts()

eind
exercise_ind = df.groupby('target')['exercise_induced_angina'].count()

exercise_ind
print(f"About {(eind[1][1]/exercise_ind[1]) * 100:.2f}% of patients that suffered exercise induced angina have a heart disease")
eind.plot(kind='barh', color='r', figsize=(12, 12))
# oldpeak: ST depression induced by exercise relative to rest

df.head()
sns.relplot(x="oldpeak", y="age", hue="target", data=df)
bins = [i for i in range(0, 8, 2)]

df['old_peak_range'] = pd.cut(df['oldpeak'], bins)

df.head()
sns.catplot(x="old_peak_range", y="sex", hue="target", kind="swarm", data=df, height=11, aspect=11.7/8.27)
df.groupby('target')['old_peak_range'].value_counts()
df.groupby(['target', 'sex'])['old_peak_range'].value_counts()
df.groupby(['target', 'sex'])['old_peak_range'].value_counts().plot(kind='bar', figsize=(12, 12))
# Slope of the peak exercise ST segment

df.head()
df.columns
df.groupby('target')['slope'].value_counts()
df.groupby('target')['slope'].value_counts().plot(kind='bar', figsize=(12, 12))
df.groupby(['target', 'sex'])['slope'].value_counts()
sns.catplot(y="target", hue="slope", kind="count",

            palette="pastel", edgecolor=".6",

            data=df)
g = sns.relplot(x="age", y="resting_blood_pressure", kind="line", data=df)
sns.relplot(x="age", y="max_heart_rate_achieved", kind="line", data=df)
sns.relplot(x="max_heart_rate_danger", y="max_heart_rate_achieved", kind="line", data=df)
sns.catplot(x="max_heart_rate_achieved", y="age_range", hue="target", kind="swarm", data=df)
# Number of major vessels colored by fluorosopy

df.head()
df.columns
df.groupby('target')['no_of_mjrv_cf'].value_counts()
df.groupby(['target','sex', 'age_range'])['no_of_mjrv_cf'].value_counts()
df.groupby('target')['thal'].value_counts()
df.groupby('target')['thal'].value_counts().plot(kind='bar', figsize=(12, 12))
df.groupby(['target', 'sex'])['thal'].value_counts().plot(kind='barh', figsize=(12, 12))
df[['age', 'sex', 'chest_pain_type', 'max_heart_rate_achieved', 'oldpeak', 'slope', 'thal', 'target']].corr()
mean = df.mean()

std = df.std()

df_norm = df.copy()

df_norm = df_norm[['age', 'sex', 'chest_pain_type', 'max_heart_rate_achieved', 'oldpeak', 'slope', 'thal', 'target']]
df_norm.head()
X = df_norm[['age', 'sex', 'chest_pain_type', 'max_heart_rate_achieved', 'oldpeak', 'slope', 'thal']]

y = df_norm.target
X.shape
y.shape
X_norm = (X - X.mean()) / X.std()
X_norm.head()
print(X['sex'].mean())
print(X['sex'].std())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=0)
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# using Linear Regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.predict(X_test[0: 10])
predictions = lr.predict(X_test)
# Measuring the models performance

score = lr.score(X_test, y_test)

print(f"{score}")
print(f"The accuracy for the logistic regression is: {score * 100:.2f}%")
# Confusion matrix

from sklearn import metrics

cm = metrics.confusion_matrix(y_test, predictions)

print(f"{cm}")
plt.figure(figsize=(12, 12))

sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')

plt.xlabel('Predicted Label')

plt.ylabel('Actual Label')

title = 'Accuracy score: {0}'.format(score)

plt.title(title, size=12)
X_p = df[['age', 'sex', 'chest_pain_type', 'max_heart_rate_danger', 'oldpeak', 'slope', 'thal']]

y_p = df.target
X_norm_p = (X_p - X_p.mean()) / X_p.std()
from sklearn.model_selection import train_test_split

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_norm_p, y_p, test_size=0.3, random_state=0)
lr1 = LogisticRegression()
lr1.fit(X_train_p, y_train_p)
lr1.predict(X_test_p[0: 10])
predictions = lr1.predict(X_test_p)
# Measuring the models performance

score1 = lr1.score(X_test_p, y_test_p)

print(f"{score1}")
# Using support vector machine (svm)

from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
score1 = metrics.accuracy_score(y_test, pred)

score1
print(f"The accuracy for the svm linear is: {score1 * 100:.2f}%")
cm_svc = metrics.confusion_matrix(y_test, pred)

print(f"{cm_svc}")
plt.figure(figsize=(12, 12))

sns.heatmap(cm_svc, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')

plt.xlabel('Predicted Label')

plt.ylabel('Actual Label')

title = 'Accuracy score: {0}'.format(score1)

plt.title(title, size=12)
# Using Decision Trees

from sklearn.tree import DecisionTreeClassifier

clftree = DecisionTreeClassifier()
clftree.fit(X_train, y_train)
clftreepred = clftree.predict(X_test)
scoretree = metrics.accuracy_score(y_test, clftreepred)

scoretree
print(f"The accuracy for Decision Tree Classifier is: {scoretree * 100:.2f}%")
cmtree = metrics.confusion_matrix(y_test, clftreepred)

print(f"{cmtree}")
# Using K-nearest neighbors (KNN)

from sklearn.neighbors import KNeighborsClassifier as knn
k = knn(n_neighbors=8)
k.fit(X_train, y_train)
knn_pred = k.predict(X_test)
knnscore = metrics.accuracy_score(y_test, knn_pred)

knnscore
knn_cm = metrics.confusion_matrix(y_test, knn_pred)

print(f"{knn_cm}")
plt.figure(figsize=(12, 12))

sns.heatmap(knn_cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')

plt.xlabel('Predicted Label')

plt.ylabel('Actual Label')

title = 'Accuracy score: {0}'.format(knnscore)

plt.title(title, size=12)