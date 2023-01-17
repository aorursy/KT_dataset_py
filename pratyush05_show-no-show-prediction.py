import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline 

sns.set_style('whitegrid')
raw_data = pd.read_csv('../input/KaggleV2-May-2016.csv')



raw_data.head()
raw_data.info()
print(raw_data['PatientId'].nunique(), raw_data['AppointmentID'].nunique(), raw_data['Gender'].nunique(), 

      raw_data['ScheduledDay'].nunique(), raw_data['AppointmentDay'].nunique(), raw_data['Age'].nunique(), 

      raw_data['Neighbourhood'].nunique(), raw_data['Scholarship'].nunique(), raw_data['Hipertension'].nunique(), 

      raw_data['Diabetes'].nunique(), raw_data['Alcoholism'].nunique(), raw_data['Handcap'].nunique(), 

      raw_data['SMS_received'].nunique())
raw_data.drop(['PatientId', 'AppointmentID'], axis=1, inplace=True)



raw_data.head()
raw_data = raw_data.rename(columns={'Gender': 'sex', 'ScheduledDay': 'scheduled_day', 'AppointmentDay': 'appointment_day', 

                                    'Age': 'age', 'Neighbourhood': 'neighbourhood', 'Scholarship': 'scholarship', 

                                    'Hipertension': 'hypertension', 'Diabetes': 'diabetic', 'Alcoholism': 'alcoholic', 

                                    'Handcap': 'handicap', 'No-show': 'show_up'})



raw_data['show_up'] = raw_data['show_up'].map({'No': 1, 'Yes': 0})



raw_data.head()
raw_data['sex'] = raw_data['sex'].map({'F': 0, 'M': 1})
raw_data['scheduled_day'] = pd.to_datetime(raw_data['scheduled_day'], infer_datetime_format=True)



raw_data['appointment_day'] = pd.to_datetime(raw_data['appointment_day'], infer_datetime_format=True)
raw_data['waiting_time'] = list(map(lambda x: x.days, raw_data['appointment_day'] - raw_data['scheduled_day']))



raw_data.head()
from collections import Counter



plt.figure(figsize=(18, 10))

for x in raw_data['waiting_time'].unique():

    plt.scatter(x, (Counter(raw_data[raw_data['waiting_time'] == x]['show_up'])[0]/ len(raw_data[raw_data['waiting_time'] == x])), c='black', s=50)
raw_data.drop(raw_data[raw_data['waiting_time'] < -1].index, inplace=True)



raw_data['waiting_time'] = raw_data['waiting_time'].apply(lambda x: 1 if(x > 1) else 0)



raw_data.head()
raw_data['appointment_dayofweek'] = raw_data['appointment_day'].map(lambda x: x.dayofweek)
raw_data.drop(raw_data[raw_data['age'] < 0].index, inplace=True)
from sklearn.preprocessing import LabelEncoder

from sklearn.externals import joblib



encoder_neighbourhood = LabelEncoder()



raw_data['neighbourhood_enc'] = encoder_neighbourhood.fit_transform(raw_data['neighbourhood'])
raw_data['handicap'] = raw_data['handicap'].apply(lambda x: 1 if x != 0 else x)
raw_data.drop(['scheduled_day', 'appointment_day', 'neighbourhood'], axis=1, inplace=True)



raw_data.head()
raw_data.info()
raw_data = raw_data.select_dtypes(['int64']).apply(pd.Series.astype, dtype='category')



raw_data['age'] = raw_data['age'].astype('int64')



raw_data.info()
sns.countplot(x='show_up', data=raw_data, palette='Set1')
sns.countplot(x='show_up', hue='sex', data=raw_data, palette='RdBu')
sns.countplot(x='appointment_dayofweek', data=raw_data, palette='GnBu_r')
sns.distplot(raw_data['age'])
sns.violinplot(x='show_up', y='age', data=raw_data, palette='BuGn_r')
fig, ax = plt.subplots(2, 3, figsize=(15, 12))



sns.countplot(x='show_up', data=raw_data, hue='scholarship', ax=ax[0, 0], palette='Set2')

sns.countplot(x='show_up', data=raw_data, hue='hypertension', ax=ax[0, 1], palette='Set2')

sns.countplot(x='show_up', data=raw_data, hue='diabetic', ax=ax[0, 2], palette='Set2')

sns.countplot(x='show_up', data=raw_data, hue='alcoholic', ax=ax[1, 0], palette='Set2')

sns.countplot(x='show_up', data=raw_data, hue='handicap', ax=ax[1, 1], palette='Set2')

sns.countplot(x='show_up', data=raw_data, hue='SMS_received', ax=ax[1, 2], palette='Set2')
sns.countplot(x='show_up', hue='waiting_time', data=raw_data, palette='RdBu')
X = raw_data.drop(['show_up'], axis=1)

y = raw_data['show_up']
X.head()
y.head()
Counter(y)
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=101)
X_res, y_res = sm.fit_sample(X, y)



Counter(y_res)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, random_state=101)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200)
clf.fit(X_train, y_train)
clf.feature_importances_
clf.score(X_test, y_test)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, clf.predict(X_test)))

print(classification_report(y_test, clf.predict(X_test)))