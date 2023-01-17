import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv")
df.isnull().any().any()
df.head()
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors = 'coerce')

df["HighSpeed"].replace({'No internet' : 0,'No' : 1 ,'Yes' : 2},inplace = True)

df['gender'].replace({'Male' : 0, 'Female' : 1},inplace=True)

df[["Married","Children","AddedServices"]] = df[["Married","Children","AddedServices"]].replace({ 'No' : 0 , 'Yes' : 1})

df[["Channel1","Channel2","Channel3","Channel4","Channel5","Channel6"]]=df[["Channel1","Channel2","Channel3","Channel4","Channel5","Channel6"]].replace({'No tv connection' : 0 , 'No' : 1 , 'Yes' : 2 })

df["Subscription"].replace({'Monthly':1,'Biannually':6,'Annually':12},inplace=True)

df = pd.get_dummies(data = df,columns=['TVConnection','PaymentMethod'])
df.isnull().any().any()
df.dropna(axis = 0,inplace=True)
df.isnull().any().any()
df_t = pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")
df_t["TotalCharges"] = pd.to_numeric(df_t["TotalCharges"],errors = 'coerce')

df_t["HighSpeed"].replace({'No internet' : 0,'No' : 1 ,'Yes' : 2},inplace = True)

df_t['gender'].replace({'Male' : 0, 'Female' : 1},inplace=True)

df_t[["Married","Children","AddedServices"]] = df_t[["Married","Children","AddedServices"]].replace({ 'No' : 0 , 'Yes' : 1})

df_t[["Channel1","Channel2","Channel3","Channel4","Channel5","Channel6"]]=df_t[["Channel1","Channel2","Channel3","Channel4","Channel5","Channel6"]].replace({'No tv connection' : 0 , 'No' : 1 , 'Yes' : 2 })

df_t["Subscription"].replace({'Monthly':1,'Biannually':6,'Annually':12},inplace=True)

df_t = pd.get_dummies(data = df_t,columns=['TVConnection','PaymentMethod'])
df_t["TotalCharges"].fillna(value = df_t["TotalCharges"].mean(),inplace=True)
df_t.isnull().any().any()
X = df[['gender', 'SeniorCitizen', 'Married', 'Children',

       'Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5',

       'Channel6', 'HighSpeed', 'AddedServices',

       'Subscription', 'tenure', 'MonthlyCharges', 'TotalCharges', 'TVConnection_Cable', 'TVConnection_DTH',

       'TVConnection_No', 'PaymentMethod_Bank transfer',

       'PaymentMethod_Cash', 'PaymentMethod_Credit card',

       'PaymentMethod_Net Banking']]

y = df['Satisfied']
X_test = df_t[['gender', 'SeniorCitizen', 'Married', 'Children',

       'Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5',

       'Channel6', 'HighSpeed', 'AddedServices',

       'Subscription', 'tenure', 'MonthlyCharges', 'TotalCharges', 'TVConnection_Cable', 'TVConnection_DTH',

       'TVConnection_No', 'PaymentMethod_Bank transfer',

       'PaymentMethod_Cash', 'PaymentMethod_Credit card',

       'PaymentMethod_Net Banking']]
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()



X = scaler.fit_transform(X)

X_test = scaler.fit_transform(X_test)
from sklearn.cluster import KMeans

from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score



kmeans = KMeans(n_clusters=2).fit(X)



y_pred = kmeans.predict(X)



acc = roc_auc_score(y,y_pred)

print(acc)
y_test = kmeans.predict(X_test)
df_f  =pd.DataFrame(df_t['custId'])

df_f['Satisfied'] = y_test

df_f.to_csv('out1.csv',index=False)