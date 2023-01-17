import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")

df
df.info()
df.isnull().sum()
df.drop('company',axis=1,inplace=True)
df['hotel'].value_counts(normalize=True).plot(kind='bar')

plt.show()
df.groupby('hotel')['is_canceled'].value_counts(normalize=True).plot(kind='bar')

plt.show()
df.groupby('is_canceled')['lead_time'].value_counts()
plt.figure(figsize=(8,5))

df.groupby(['hotel'])['arrival_date_year'].value_counts(normalize=True).plot(kind='bar')

plt.show()
plt.figure(figsize=(8,5))

df.groupby(['hotel'])['stays_in_weekend_nights'].value_counts(normalize=True).plot(kind='bar')

plt.show()
df.groupby(['arrival_date_month'])['stays_in_weekend_nights','stays_in_week_nights'].sum().plot(kind='bar')

plt.show()
plt.figure(figsize=(8,5))

df.groupby(['hotel'])['deposit_type'].value_counts(normalize=True).plot(kind='bar')

plt.show()
plt.figure(figsize=(8,5))

df[df['stays_in_week_nights']<=10].groupby(['hotel'])['stays_in_week_nights'].value_counts(normalize=True).plot(kind='bar')

plt.show()
plt.figure(figsize=(8,5))

df[df['adults']<=5].groupby(['hotel','adults'])['children'].value_counts(normalize=True).plot(kind='bar')

plt.show()
df.groupby('hotel')['is_repeated_guest'].value_counts(normalize=True).plot(kind='bar')

plt.show()
df['reservation_status'].value_counts()
df['reservation_status_date'].value_counts()
df['customer_type'].value_counts()
df['days_in_waiting_list'].value_counts()
df[df['days_in_waiting_list']>5].groupby('hotel')['days_in_waiting_list'].value_counts()
df[df['reserved_room_type'] != df['assigned_room_type']]['assigned_room_type'].count()
df[df['previous_cancellations'] != df['previous_bookings_not_canceled']]['previous_bookings_not_canceled'].count()
print(df[df['previous_cancellations']==1]['customer_type'].value_counts())

df[df['previous_cancellations']==1].groupby('hotel')['customer_type'].value_counts().plot(kind='bar')

plt.show()
df.info()
from sklearn.preprocessing import LabelEncoder

lr = LabelEncoder()
df['hotel'] = lr.fit_transform(df['hotel'])
df = pd.get_dummies(df,columns=['reservation_status'],drop_first=True)
df = pd.get_dummies(df,columns=['deposit_type','meal','market_segment','distribution_channel'],drop_first=True)
df = pd.get_dummies(df,columns=['reserved_room_type','assigned_room_type','customer_type'],drop_first=True)
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
df['agent'].mode()
df['children'].mode()
df['agent'] = df['agent'].fillna(9.0)

df['children'] = df['children'].fillna(0.0)
df1 = df.drop(['reservation_status_date'],axis=1)
df1.info()
df2 = df1[df1['country']=='PRT']
df2 = df2.drop('country',axis=1)
df2 = pd.get_dummies(df2,columns=['arrival_date_month'],drop_first=True)
X = df2.drop(['is_canceled'],axis=1)

y = df2['is_canceled']
X.info()
from sklearn.ensemble import RandomForestClassifier

rfc= RandomForestClassifier()

from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint as sp_randint

params={'n_estimators':sp_randint(5,150), 'max_features':sp_randint(10,40),'max_depth':sp_randint(2,10),'min_samples_leaf':sp_randint(1,50),'min_samples_split':sp_randint(2,50),'criterion':['gini','entropy']}

rsearch=RandomizedSearchCV(rfc,param_distributions=params,n_jobs=-1,scoring='accuracy',n_iter=100,cv=3)

rsearch.fit(X,y)
rsearch.best_params_
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)

rfc= RandomForestClassifier(**rsearch.best_params_)

rfc.fit(X_train,y_train)

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, classification_report

y_train_pred=rfc.predict(X_train)

#y_train_prob=rfc.predict_proba(X_train)[:,1]

print('Accuracy score of train: ', accuracy_score(y_train,y_train_pred))

print('Confusion Matrix of train: ', confusion_matrix(y_train,y_train_pred))

#print('Auc of train: ', roc_auc_score(y_train,y_train_prob))

y_test_pred=rfc.predict(X_test)

#y_test_prob=rfc.predict_proba(X_test)[:,1]

print('Accuracy score of test: ', accuracy_score(y_test,y_test_pred))

print('Confusion Matrix of test: ', confusion_matrix(y_test,y_test_pred))

#print('Auc of test: ', roc_auc_score(y_test,y_test_prob))
df1['country'].mode()
df1['country'] = df1['country'].fillna('PRT')
df3 = pd.get_dummies(df1,columns=['arrival_date_month','country'],drop_first=True)
X = df3.drop(['is_canceled'],axis=1)

y = df3['is_canceled']
params={'n_estimators':sp_randint(5,150), 'max_features':sp_randint(10,40),'max_depth':sp_randint(2,10),'min_samples_leaf':sp_randint(1,50),'min_samples_split':sp_randint(2,50),'criterion':['gini','entropy']}

rsearch=RandomizedSearchCV(rfc,param_distributions=params,n_jobs=-1,scoring='accuracy',n_iter=100,cv=3)

rsearch.fit(X,y)
rsearch.best_params_
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)

rfc= RandomForestClassifier(**rsearch.best_params_)

rfc.fit(X_train,y_train)

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, classification_report

y_train_pred=rfc.predict(X_train)

#y_train_prob=rfc.predict_proba(X_train)[:,1]

print('Accuracy score of train: ', accuracy_score(y_train,y_train_pred))

print('Confusion Matrix of train: ', confusion_matrix(y_train,y_train_pred))

#print('Auc of train: ', roc_auc_score(y_train,y_train_prob))

y_test_pred=rfc.predict(X_test)

#y_test_prob=rfc.predict_proba(X_test)[:,1]

print('Accuracy score of test: ', accuracy_score(y_test,y_test_pred))

print('Confusion Matrix of test: ', confusion_matrix(y_test,y_test_pred))

#print('Auc of test: ', roc_auc_score(y_test,y_test_prob))