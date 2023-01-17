import numpy as np

import seaborn as sns

import pandas as pd

pd.set_option('display.max_columns',50)

from sklearn.metrics import f1_score

import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE

from sklearn.model_selection import train_test_split

train = pd.read_csv("../input/predict-the-churn-for-customer-dataset/Train File.csv")

test = pd.read_csv("../input/predict-the-churn-for-customer-dataset/Test File.csv")
train.info()
train.isnull().sum()
sns.distplot(train['TotalCharges'].dropna())
train.fillna(train['TotalCharges'].median(), inplace=True)

test.fillna(train['TotalCharges'].median(), inplace=True)
for i in train.columns :

    print(i,' :')

    print(train[i].value_counts())
train['gender'] = train['gender'].map({'Female':0, 'Male':1})

test['gender'] = test['gender'].map({'Female':0, 'Male':1})



train['Partner'] = train['Partner'].map({'Yes':1,'No':0})

test['Partner'] = test['Partner'].map({'Yes':1,'No':0})



train['Dependents'] = train['Dependents'].map({'Yes':1,'No':0})

test['Dependents'] = test['Dependents'].map({'Yes':1,'No':0})



train['PhoneService'] = train['PhoneService'].map({'Yes':1,'No':0})

test['PhoneService'] = test['PhoneService'].map({'Yes':1,'No':0})



train['MultipleLines'] = train['MultipleLines'].map({'Yes':1,'No':0, 'No phone service':0})

test['MultipleLines'] = test['MultipleLines'].map({'Yes':1,'No':0, 'No phone service':0})



train['OnlineSecurity'] = train['OnlineSecurity'].map({'Yes':1,'No':0, 'No internet service':0})

test['OnlineSecurity'] = test['OnlineSecurity'].map({'Yes':1,'No':0, 'No internet service':0})



train['OnlineBackup'] = train['OnlineBackup'].map({'Yes':1,'No':0, 'No internet service':0})

test['OnlineBackup'] = test['OnlineBackup'].map({'Yes':1,'No':0, 'No internet service':0})



train['DeviceProtection'] = train['DeviceProtection'].map({'Yes':1,'No':0, 'No internet service':0})

test['DeviceProtection'] = test['DeviceProtection'].map({'Yes':1,'No':0, 'No internet service':0})



train['TechSupport'] = train['TechSupport'].map({'Yes':1,'No':0, 'No internet service':0})

test['TechSupport'] = test['TechSupport'].map({'Yes':1,'No':0, 'No internet service':0})



train['StreamingTV'] = train['StreamingTV'].map({'Yes':1,'No':0, 'No internet service':0})

test['StreamingTV'] = test['StreamingTV'].map({'Yes':1,'No':0, 'No internet service':0})



train['StreamingMovies'] = train['StreamingMovies'].map({'Yes':1,'No':0, 'No internet service':0})

test['StreamingMovies'] = test['StreamingMovies'].map({'Yes':1,'No':0, 'No internet service':0})



train['PaperlessBilling'] = train['PaperlessBilling'].map({'Yes':1,'No':0})

test['PaperlessBilling'] = test['PaperlessBilling'].map({'Yes':1,'No':0})



train['Churn'] = train['Churn'].map({'Yes':1,'No':0})
train.drop('customerID', axis=1, inplace=True)

ans_id = test['customerID']

test.drop('customerID', axis=1, inplace=True)
train = pd.get_dummies(train)

test = pd.get_dummies(test)
x = train.drop('Churn', axis=1)

y = train['Churn'].values
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x = scaler.fit_transform(x)

test = scaler.transform(test)
from xgboost import XGBClassifier



xgb = XGBClassifier(n_estimators=1000, learning_rate=0.01)

from sklearn.feature_selection import RFE

rfe = RFE(xgb, 10)

rfe.fit(x,y)

x = x[:,rfe.get_support()]

test = test[:,rfe.get_support()]
xgb.fit(x,y)

xgb.score(x,y)
ans=xgb.predict(test)
ans = pd.DataFrame(ans, columns=['Churn'])
ans['customerID'] = ans_id
ans['Churn'] = ans['Churn'].map({0:'No', 1:'Yes'})
ans.set_index('customerID', inplace=True)
ans.to_csv('Raj Naruka Submission 1.csv')