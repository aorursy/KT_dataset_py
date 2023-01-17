import numpy as np

import pandas as pd
raw_data = pd.read_csv('../input/KaggleV2-May-2016.csv')
raw_data.info()
raw_data.describe()
raw_data.head(10)
raw_data['PatientId'].nunique()
raw_data['Neighbourhood'].nunique()
raw_data['AppointmentID'].nunique()
raw_data['AppointmentDay'].nunique()
raw_data['ScheduledDay'].nunique()
raw_data.drop('AppointmentID', axis=1, inplace=True)



raw_data.drop('ScheduledDay', axis=1, inplace=True)



raw_data.head()
raw_data['AppointmentDay'] = pd.to_datetime(raw_data['AppointmentDay'], infer_datetime_format=True)



raw_data['AppointmentDay'] = raw_data['AppointmentDay'].map(lambda x: x.dayofweek)



raw_data.head()
raw_data['Gender'] = raw_data['Gender'].map({'F': 0, 'M': 1})



raw_data.head()
from sklearn.preprocessing import LabelEncoder
encoder_neighbourhood = LabelEncoder()



raw_data['neighbourhood_enc'] = encoder_neighbourhood.fit_transform(raw_data['Neighbourhood'])
raw_data.head()
raw_data = raw_data.rename(columns={'PatientId': 'pid', 'Gender': 'sex', 'AppointmentDay': 'appointment_day', 

                                    'Age': 'age', 'Neighbourhood': 'neighbourhood', 'Scholarship': 'scholarship', 

                                    'Hipertension': 'hypertension', 'Diabetes': 'diabetic', 'Alcoholism': 'alcoholic', 

                                    'Handcap': 'handicap', 'No-show': 'show_up'})



raw_data['show_up'] = raw_data['show_up'].map({'No': 1, 'Yes': 0})



raw_data.head()
data = raw_data[['sex', 'appointment_day', 'age', 'neighbourhood_enc', 'scholarship', 

                 'hypertension', 'diabetic', 'alcoholic', 'handicap', 'SMS_received', 'show_up']]



data.head()
import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline

sns.set_style('whitegrid')
sns.countplot(x='show_up', data=data, palette='coolwarm')
sns.countplot(x='show_up', hue='sex', data=data, palette='hls')
sns.countplot(x='appointment_day', data=data, hue='show_up', palette='Set1')
sns.barplot(x='appointment_day', y='show_up', hue='alcoholic', data=data, palette='Set1')
sns.barplot(x='scholarship', y='show_up', data=data)
sns.countplot(x='show_up', hue='hypertension', data=data, palette='Paired')
sns.countplot(x='show_up', hue='diabetic', data=data, palette='Paired')
sns.countplot(x='show_up', hue='alcoholic', data=data, palette='cubehelix')
sns.barplot(x='alcoholic', y='show_up', data=data, palette='cubehelix')
sns.countplot(x='show_up', hue='handicap', data=data, palette='Set2')
sns.barplot(x='handicap', y='show_up', data=data, palette='Set2')
sns.countplot(x='show_up', hue='SMS_received', data=data, palette='Set3')
sns.barplot(x='SMS_received', y='show_up', data=data, palette='Set3')
sns.violinplot(x='show_up', y='age', data=data)
from sklearn.model_selection import train_test_split
data.columns
X = data[data.columns[:-1]]

y = pd.DataFrame(data[data.columns[-1]])
X.columns
y.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, np.ravel(y_train))
rfc.score(X_test, y_test)
from sklearn.metrics import confusion_matrix, classification_report
predictions = rfc.predict(X_test)



print(confusion_matrix(y_test, predictions))



print(classification_report(y_test, predictions))
from sklearn.naive_bayes import MultinomialNB
nbc = MultinomialNB()
nbc.fit(X_train, np.ravel(y_train))
nbc.score(X_test, y_test)
predictions = nbc.predict(X_test)



print(confusion_matrix(y_test, predictions))



print(classification_report(y_test, predictions))
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, np.ravel(y_train))
gbc.score(X_test, y_test)
predictions = gbc.predict(X_test)



print(confusion_matrix(y_test, predictions))



print(classification_report(y_test, predictions))