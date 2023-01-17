#import data
import pandas as pd
import numpy as np
import datetime
from time import strftime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns


ApptData = pd.read_csv('../input/kaggle-v3/KaggleV2-May-2016v3.csv' ,encoding="ISO-8859-1")

print(ApptData.head())



#Renaming some fields

ApptData.rename(columns = {'Hipertension': 'Hypertension',
                         'Neighbourhood': 'Neighborhood',
                         'No-show': 'ApptMade',
                         'Handcap': 'Handicap'}, inplace = True)  

print(ApptData.columns)
#Take a little peek at the data
ApptData.head()
#Look at the features

#Features in the Datfram
print("Features in the DataFrame => {}".format(ApptData.columns.ravel()))
# Print Unique Values
print("Unique Values in `Gender` => {}".format(ApptData.Gender.unique()))
print("Unique Values in `Scholarship` => {}".format(ApptData.Scholarship.unique()))
print("Unique Values in `Hypertension` => {}".format(ApptData.Hypertension.unique()))
print("Unique Values in `Diabetes` => {}".format(ApptData.Diabetes.unique()))
print("Unique Values in `Alcoholism` => {}".format(ApptData.Alcoholism.unique()))
print("Unique Values in `Handicap` => {}".format(ApptData.Handicap.unique()))
print("Unique Values in `SMSReceived` => {}".format(ApptData.SMS_received.unique()))
print("Unique Values in `ApptMade` => {}".format(ApptData.ApptMade.unique()))
print("Unique Values in `AppointmentDay_DOW` => {}".format(ApptData.AppointmentDay_DOW.unique()))
print("Unique Values in `ScheduledDay` => {}".format(ApptData.ScheduledDay_DOW.unique()))
print("Unique Values in `Age` => {}".format(np.sort(ApptData.Age.unique())))
print("Unique Values in `WaitDays` => {}".format(np.sort(ApptData.WaitDays.unique())))
ApptData.info()
#Dataviz by Features and Gender

ax = sns.countplot(x=ApptData.Gender, hue=ApptData.ApptMade, data=ApptData)
ax.set_title("Appointments made Made by Gender")
x_ticks_labels=['Female', 'Male']
ax.set_xticklabels(x_ticks_labels)
plt.show()

ax = sns.countplot(x=ApptData.Diabetes, hue=ApptData.ApptMade, data=ApptData)  
ax.set_title("Appointments made Made by Diabetes & Gender")
x_ticks_labels=['No Diabetes','Diabetes']
ax.set_xticklabels(x_ticks_labels)
plt.show()

ax = sns.countplot(x=ApptData.Hypertension, hue=ApptData.ApptMade, data=ApptData)  
ax.set_title("Appointments made Made by Hypertension & Gender")
x_ticks_labels=['No Hypertension', 'Hypertension']
ax.set_xticklabels(x_ticks_labels)
plt.show()



ax = sns.countplot(x=ApptData.Alcoholism, hue=ApptData.ApptMade, data=ApptData)  
ax.set_title("Appointments made Made by Hypertension & Gender")
x_ticks_labels=['No Alcoholism', 'Alcoholism']
ax.set_xticklabels(x_ticks_labels)
plt.show()
# Counts by Gender/ApptMade
print("No Show and Show Count of Patients\n")
print(ApptData.groupby(['ApptMade']).size())

print("\nNo Show and Show '%' of Patients\n")
show = ApptData.groupby(['ApptMade']).size()[0]/(ApptData.groupby(['ApptMade']).size()[0]+ApptData.groupby(['ApptMade']).size()[1])
print("Percent of Patients who `Showed Up` => {:.2f}%".format(show*100))
noshow = ApptData.groupby(['ApptMade']).size()[1]/(ApptData.groupby(['ApptMade']).size()[0]+ApptData.groupby(['ApptMade']).size()[1])
print("Percent of Patients who `Didn't Show Up` => {:.2f}%".format(noshow*100))
#Convert some dates to datetime
import datetime as dt

ApptData['ScheduledDay'] = pd.to_datetime(ApptData['ScheduledDay'])
ApptData['AppointmentDay'] = pd.to_datetime(ApptData['AppointmentDay'])
ApptData['Timedelta'] = ApptData['AppointmentDay'] - ApptData['ScheduledDay']
ApptData['Timedelta'] = ApptData['Timedelta'].dt.days
# Use `LabelEncoder` to encode labels with value between 0 and n_classes-1.
#Gender
le = LabelEncoder()
ApptData['Gender'] = le.fit_transform(ApptData['Gender'])
#Neighbourhood
le = LabelEncoder()
ApptData['Neighborhood'] = le.fit_transform(ApptData['Neighborhood'])
#ScheduledDay_DOW
le = LabelEncoder()
ApptData['ScheduledDay_DOW'] = le.fit_transform(ApptData['ScheduledDay_DOW'])
#AppointmentDay_DOW
le = LabelEncoder()
ApptData['AppointmentDay_DOW'] = le.fit_transform(ApptData['AppointmentDay_DOW'])
print("LabelEncoder Completed")
#Changing fields to integers
ApptData['ScheduledDay_Y'] = ApptData['ScheduledDay'].dt.year
ApptData['ScheduledDay_M'] = ApptData['ScheduledDay'].dt.month
ApptData['ScheduledDay_D'] = ApptData['ScheduledDay'].dt.day
ApptData.drop(['ScheduledDay'], axis=1, inplace=True)

ApptData['AppointmentDay_Y'] = ApptData['AppointmentDay'].dt.year
ApptData['AppointmentDay_M'] = ApptData['AppointmentDay'].dt.month
ApptData['AppointmentDay_D'] = ApptData['AppointmentDay'].dt.day
ApptData.drop(['AppointmentDay'], axis=1, inplace=True)
ApptData.head()
#ApptMade
le = LabelEncoder()
ApptData['ApptMade'] = le.fit_transform(ApptData['ApptMade'])
ApptData.head()
# Get the Dependent and Independent Features.
X = ApptData.drop(['ApptMade'], axis=1)
y = ApptData['ApptMade']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.49, random_state=0)
#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

dt_clf = DecisionTreeClassifier(random_state=0)
dt_clf.fit(X_train, y_train)
print("Feature Importance:\n")
for name, importance in zip(X.columns, np.sort(dt_clf.feature_importances_)[::-1]):
    print("{} -- {:.2f}".format(name, importance))

dt_clf.score(X_test, y_test)
#Randon Tree Classifier
rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(X_train, y_train)
print("Feature Importance:\n")
for name, importance in zip(X.columns, np.sort(rf_clf.feature_importances_)[::-1]):
    print("{} -- {:.2f}".format(name, importance))

rf_clf.score(X_test, y_test)
#splitting data into training and test, first by even split of records
features_train = ApptData[['Age','Diabetes','Alcoholism','Hypertension','Gender']].iloc[:55262]

labels_train = ApptData.ApptMade[:55262]

features_test = ApptData[['Age','Diabetes','Alcoholism','Hypertension','Gender']].iloc[55262:]

labels_test = ApptData.ApptMade[55262:]
#using naive bayes for model accuracy #1, 81.0%
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

clf =  MultinomialNB().fit(features_train, labels_train)
print('Accuracy:', round(accuracy_score(labels_test, 
                                        clf.predict(features_test)), 2) * 100, '%')
#Again with slipt from above for Recall and Precision
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

clf = RandomForestClassifier(n_estimators=100, max_depth = 6, random_state=47).fit(X_train, y_train)

y_predicted = clf.predict(X_test)

print('Recall: {:.3f}'.format(recall_score(y_test, y_predicted)))
print('Precision: {:.3f}'.format(precision_score(y_test, y_predicted)))
print('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_predicted)))
print('F1: {:.3f}'.format(f1_score(y_test, y_predicted)))
confusion = confusion_matrix(y_test, y_predicted)
print(confusion)
print('Feature importances: {}'.format(clf.feature_importances_))