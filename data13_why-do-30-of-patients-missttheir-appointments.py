import re

import sys



import time

import datetime as dt

from datetime import datetime



import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



from sklearn import metrics, preprocessing

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report, accuracy_score, f1_score
df = pd.read_csv('../input/KaggleV2-May-2016.csv')

df.head()
# check if any missing values

df.isnull().any()
sns.heatmap(df.isnull(), cmap='viridis')
# check for doublications

df.duplicated().any()
df['Age'].unique()
# clean Age

df = df[(df['Age'] > 0) & (df['Age'] < 95)]
# clean AppointmentDay

df['AppointmentDay'] = df['AppointmentDay'].apply(lambda x : x.replace('T00:00:00Z', ''))

df['AppointmentDay'] = pd.to_datetime(df['ScheduledDay'])
# extract days & months from the appointments

df['AppointmentDays'] = df['AppointmentDay'].apply(lambda x : dt.datetime.strftime(x, '%A'))

df['AppointmentMonths'] = df['AppointmentDay'].apply(lambda x : dt.datetime.strftime(x, '%B'))
# summary statistics

df.describe()
# who visits the doctor more often?

visits = df[df['No-show'] == 'No']

sns.countplot(x='Gender', data=visits)
# what is the proportion of male & female?

df['Gender'].value_counts(normalize = True)
# visualise male vs female distribution

fig, pie = plt.subplots()

df['Gender'].value_counts(normalize = True).plot(kind='pie', autopct='%1.1f%%')

pie.axis('equal')
# what is the proportion of no-shows

df['No-show'].value_counts(normalize = True)
# no. of patients who miss their appointments

len(df[df['No-show'] == 'Yes'].index)
# overall no-show percentage

# from __future__ import division

len(df[df['No-show'] == 'Yes'].index) / len(df.index)
# visualise prcentage of no-shows

sns.countplot(x='No-show', data=df)
# male vs female age and its effect on showing?

sns.violinplot(df['No-show'], df['Age'], hue=df['Gender'])
# what is the patient age distribution for no-shows versus shows?

df.groupby('No-show')['Age'].mean()
# age distribution for show & no show

# use FacetGrid to plot multiple kdeplots on one plot

fig = sns.FacetGrid(df, hue='No-show', aspect=4)

# call FacetGrid.map() to use sns.kdeplot() to show age distribution

fig.map(sns.kdeplot, 'Age', shade=True)

fig.add_legend()
# what is the proportion of missing the appointment whether the patient is male or female?

x = sns.countplot(x='No-show', hue='Gender', data=df)

# to plot the values over the labels

total = float(len(df))

for p in x.patches:

    height = p.get_height()

    x.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format(height/total),

            ha='center')
# what is the relationship between no-show and diabetes?

sns.countplot(x='No-show', hue='Diabetes', data=df)
# what is the relationship between no-show and each of scholarship, handicap, hipertension and alcoholism respectively?

fig, ax =plt.subplots(2,2, figsize=(15,10))

sns.countplot(x='No-show', hue='Scholarship', data=df, ax=ax[0][0]).set_title('No-show vs Scholarship')

sns.countplot(x='No-show', hue='Handcap', data=df, ax=ax[0][1]).set_title('No-show vs Handicap')

sns.countplot(x='No-show', hue='Hipertension', data=df, ax=ax[1][0]).set_title('No-show vs Hipertension')

sns.countplot(x='No-show', hue='Alcoholism', data=df, ax=ax[1][1]).set_title('No-show vs Alcoholism')
# what is the relationship between neighbourhoods and the number of shows?

plt.figure(figsize=(35,4))

sns.countplot(x='Neighbourhood', hue='No-show', data=df)

plt.xticks(rotation=45)
# which neighbourhoods have the highest number of shows?

shows = df[df['No-show'] == 'No']

shows.groupby('Neighbourhood')['No-show'].count().sort_values(ascending=False).head(20)
# visualise neighbourhoods vs shows

plt.figure(figsize=(35,4))

sns.countplot(x='Neighbourhood', data=shows)

plt.xticks(rotation=45)
# in which month most patients miss their appointments

sns.countplot(x='AppointmentMonths', hue='No-show', data=df, order = df['AppointmentMonths'].value_counts().index)

plt.xticks(rotation=20)
# in which day the patients visit their doctors most frequently?

sns.countplot(x='AppointmentDays', data=visits)

plt.xticks(rotation=15)
# is there specific weekday when most patients miss their appointments?

miss = df[df['No-show'] == 'Yes']

miss.groupby('AppointmentDays')['No-show'].count().sort_values(ascending=False)
# what is the weekly no-show count

sns.countplot(x='AppointmentDays', hue='No-show', data=df)
# do men and women visit the doctor on the same days?

sns.countplot(x='AppointmentDays', hue='Gender', data=visits)
# how likely patients comes to their scheduled appointment if they have received a sms

sns.catplot('SMS_received', hue='No-show', data=df, kind='count')
# what is the proportion of patients who has received familia scholarship?

sns.catplot('Scholarship', hue='Gender', data=df, kind='count')
# based on neighbourhood and scholarship, how showing-up is affected?

nbrhd_schlrshp_nshw = pd.DataFrame(df[['Neighbourhood','Scholarship','No-show']].groupby( ['Neighbourhood', 'No-show','Scholarship']).size().reset_index(name = 'Count'))

nbrhd_schlrshp_nshw.head(30)
# visualise showing up based on Neighbourhood vs Scholarship

nbrhd_schlrshp_nshw['Neighbourhood'] = nbrhd_schlrshp_nshw['Neighbourhood'].apply(lambda x: x)

sns.catplot(x = 'Count', y = 'Neighbourhood', hue = 'No-show', data = nbrhd_schlrshp_nshw, col = 'Scholarship', kind = 'swarm', height = 20, aspect = 0.25)
# which factors can help to predict the showing up of a patient?

# first, let's look at how much each independent variable correlates with No-show (dependent variable)

from IPython import display

df['No-show'] = pd.get_dummies(df['No-show'])

independent_variables = ['SMS_received', 'Scholarship', 'Hipertension', 'Alcoholism', 'Diabetes', 'Handcap']

for variable in independent_variables :

    display.display(df.groupby(variable)['No-show'].mean())
# converting categorical data to numerical data

df['Gender'] = pd.get_dummies(df['Gender'])



le = preprocessing.LabelEncoder()

df['Age'] = le.fit_transform(df['Age'])



le = preprocessing.LabelEncoder()

df['ScheduledDay'] = le.fit_transform(df['ScheduledDay'])



le = preprocessing.LabelEncoder()

df['Neighbourhood'] = le.fit_transform(df['Neighbourhood'])



le = preprocessing.LabelEncoder()

df['AppointmentDay'] = le.fit_transform(df['AppointmentDay'])
# split data

features = ['ScheduledDay', 'AppointmentDay', 'PatientId', 'AppointmentID', 'Gender', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']

X = df[features]

y = df['No-show']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
lr = LogisticRegression(solver='lbfgs')

lr.fit(X_train, y_train)



lr_y_pred = lr.predict(X_test)

lr_conf_mat = confusion_matrix(y_test, lr_y_pred)
sns.heatmap(lr_conf_mat, cmap='PuBu', annot=True, fmt='d')

plt.ylabel('Actual class')

plt.xlabel('Predicted class')

plt.title('Confusion Matrix')
TN = lr_conf_mat[0][0]

TP = lr_conf_mat[1][1]

accuracy = 100*float(TP+TN)/float(np.sum(lr_conf_mat))

'Accuracy: ' + str(np.round(accuracy, 2)) + '%'
knn = KNeighborsClassifier(n_neighbors=5)  

knn.fit(X_train, y_train)
knn_y_pred = knn.predict(X_test)

knn_conf_mat = confusion_matrix(y_test, knn_y_pred)
sns.heatmap(knn_conf_mat, cmap='YlGnBu', annot=True, fmt='d')

plt.ylabel('Actual class')

plt.xlabel('Predicted class')

plt.title('Confusion Matrix')
# classification_report(y_test, knn_y_pred)

'Accuracy: ' + str(metrics.accuracy_score(y_test, knn_y_pred))
error = []

# calculating error for K values between 1 and 50

for i in range(1, 50):  

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    pred_i = knn.predict(X_test)

    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))  

plt.plot(range(1, 50), error, color='red', linestyle='dashed', marker='o',  

         markerfacecolor='blue', markersize=10)

plt.title('Error Rate K Value')  

plt.xlabel('K Value')  

plt.ylabel('Mean Error')
rf = RandomForestClassifier(n_estimators=250, max_depth=100, min_samples_split=50, min_samples_leaf=100, n_jobs=-1)

rf.fit(X_train, y_train)



rf_y_pred = rf.predict(X_test)

rn_conf_mat = confusion_matrix(y_test, rf_y_pred)
sns.heatmap(rn_conf_mat, cmap='RdPu', annot=True, fmt='d')

plt.ylabel('Actual class')

plt.xlabel('Predicted class')

plt.title('Confusion Matrix')
'Accuracy: ' + str(accuracy_score(y_test, rf_y_pred))
'F1: ' +  str(f1_score(y_test, rf_y_pred))