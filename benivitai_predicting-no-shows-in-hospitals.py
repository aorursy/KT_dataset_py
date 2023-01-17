import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from pprint import pprint



%matplotlib inline
df_raw = pd.read_csv('../input/noshowappointments/KaggleV2-May-2016.csv')
df_raw.info()
# PatientId should be int64, not float64

df_raw['PatientId'] = df_raw['PatientId'].astype('int64')



# Convert ScheduledDay and AppointmentDay to datetime64[ns]

df_raw['ScheduledDay'] = pd.to_datetime(df_raw['ScheduledDay']).dt.date.astype('datetime64[ns]')

df_raw['AppointmentDay'] = pd.to_datetime(df_raw['AppointmentDay']).dt.date.astype('datetime64[ns]')
# check the head of dataset

df_raw.head()
# rename typo columns

df_raw.rename(columns={"Hipertension": "Hypertension","Handcap":"Handicap",

                      "SMS_received": "SMSReceived", "No-show": "NoShow"},inplace=True)
# check for typos

print(sorted(df_raw['Neighbourhood'].unique()))
# Check Age

print(sorted(df_raw['Age'].unique()))
df_raw[df_raw['Age'] == -1]
df_raw[df_raw['Age'] == 115]
# Remove erroneous entries

df_raw = df_raw[(df_raw['Age'] < 115) & (df_raw['Age'] > 0)]
df_raw = df_raw.drop(['PatientId','AppointmentID'],axis=1)
df_raw['ScheduledMonth'] = df_raw['ScheduledDay'].dt.month

df_raw['ScheduledDayofWeek'] = df_raw['ScheduledDay'].dt.day_name()

df_raw['ScheduledHour'] = df_raw['ScheduledDay'].dt.hour
df_raw['AppointmentMonth'] = df_raw['AppointmentDay'].dt.month

df_raw['AppointmentDayofWeek'] = df_raw['AppointmentDay'].dt.day_name()

df_raw['AppointmentHour'] = df_raw['AppointmentDay'].dt.hour
sns.countplot(x='Gender', hue='NoShow', data=df_raw)
plt.figure(figsize=(30,12))

fig = sns.countplot(x='Neighbourhood',hue='NoShow',data=df_raw)

fig.set_xticklabels(fig.get_xticklabels(), rotation=90);
sns.heatmap(df_raw.corr(), vmin=-0.9, vmax=0.9,cmap='coolwarm')
df_raw['AppointmentDayofWeek'] = df_raw['AppointmentDay'].dt.weekday

df_raw['ScheduledDayofWeek'] = df_raw['ScheduledDay'].dt.weekday
df_raw['NoShow'] = pd.get_dummies(df_raw['NoShow'])['Yes']
no_show = len(df_raw[df_raw['NoShow'] == 1])

print(f'No-shows: {no_show}')



total = len(df_raw)

print(f'Percentage no-show: {(no_show/total) * 100}')
# skewed towards female entries

print(f"Gender entries: {df_raw['Gender'].unique()}")

print(df_raw['Gender'].describe())

df_raw['Male'] = pd.get_dummies(df_raw['Gender'])['M']

      

df_raw = df_raw.drop('Gender',axis=1)
# get dummy variables for neighbourhood

neighbourhoods = pd.get_dummies(df_raw['Neighbourhood'])



# join dummy neighbourhood columns and drop string neighbourhood column

df_raw = df_raw.join(neighbourhoods).drop('Neighbourhood',axis=1)
df = df_raw.drop(['AppointmentDay','ScheduledDay'],axis=1)
# import StandardScaler from Scikit learn

from sklearn.preprocessing import StandardScaler



# create StandardScaler object

scaler = StandardScaler()



# fit scaler to features

scaler.fit(df.drop(['NoShow'],axis=1))
# use .transform() to transform features to scaled version

scaled_features = scaler.transform(df.drop('NoShow',axis=1))
df_feat = pd.DataFrame(scaled_features)

df_feat.head()
# Import train_test_split function

from sklearn.model_selection import train_test_split



X = df_feat  # Features

y = df['NoShow']  # Labels



# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

from sklearn.tree import DecisionTreeClassifier



dtree = DecisionTreeClassifier()



# fit to data

dtree.fit(X_train,y_train)



# get predictions

dtree_pred = dtree.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,dtree_pred))
print("Confusion matrix:\n",confusion_matrix(y_test, dtree_pred))
#Import Random Forest Model

from sklearn.ensemble import RandomForestClassifier



#Create a Gaussian Classifier

rfc = RandomForestClassifier(n_estimators=100,verbose=5)



#Train the model using the training sets y_pred=clf.predict(X_test)

rfc.fit(X_train,y_train)



rfc_pred = rfc.predict(X_test)
print(classification_report(y_test,rfc_pred))
# Model Accuracy, how often is the classifier correct?

print("Confusion matrix:\n",confusion_matrix(y_test, rfc_pred))
from sklearn.linear_model import LogisticRegression



# Instantiate model

logmodel = LogisticRegression(max_iter=1000)



# Train model

logmodel.fit(X_train,y_train)



# Get predictions

log_pred = logmodel.predict(X_test)
print(classification_report(y_test,log_pred))
print("Confusion matrix:\n",confusion_matrix(y_test, log_pred))