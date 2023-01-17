from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn import preprocessing

import matplotlib.pylab as pylab

import matplotlib.pyplot as plt

from pandas import get_dummies

import matplotlib as mpl

import xgboost as xgb

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib

import warnings

import sklearn

import scipy

import numpy

import json

import sys

import csv

import os
warnings.filterwarnings('ignore')

%matplotlib inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#loading the dataset

df_train = pd.read_csv('/kaggle/input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')

df_test = pd.read_csv('/kaggle/input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv')
#function for missing data

def missing_data(df_train):

    total = df_train.isnull().sum().sort_values(ascending=False)

    percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return(missing_data.head(20))
df_train.head()
df_train['Gender'] = df_train['Gender'].fillna('Others')

df_train['Married'] = df_train['Married'].fillna('Yes')

df_train['Dependents'] = df_train['Dependents'].fillna('0')

df_train['Self_Employed'] = df_train['Self_Employed'].fillna('No')

df_train['Loan_Amount_Term'] = df_train['Loan_Amount_Term'].fillna('360')

df_train['Credit_History']=df_train['Credit_History'].fillna('1')

df_train['LoanAmount']=df_train['LoanAmount'].fillna('147')
df_train = df_train.drop(['Loan_ID'],axis=1)
df_train.dtypes
missing_data(df_train)
df_train['ApplicantIncome']=df_train['ApplicantIncome'].astype('int')

df_train['LoanAmount']=df_train['LoanAmount'].astype('int')

df_train['Loan_Amount_Term']=df_train['Loan_Amount_Term'].astype('int')

df_train['CoapplicantIncome']=df_train['CoapplicantIncome'].astype('int')

df_train['Credit_History']=df_train['Credit_History'].astype('int')
df_train.Loan_Status[df_train.Loan_Status == 'Y'] = 1

df_train.Loan_Status[df_train.Loan_Status == 'N'] = 0
df_train['Loan_Status']=df_train['Loan_Status'].astype('int')
encoded = pd.get_dummies(df_train)
encoded.head()
dependent_all=encoded['Loan_Status']
independent_all=encoded.drop(['Loan_Status'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(independent_all,dependent_all,test_size=0.3,random_state=100)
xgboost = xgb.XGBClassifier(max_depth=3,n_estimators=300,learning_rate=0.05)
xgboost.fit(x_train,y_train)
#XGBoost modelon the train set

XGB_prediction = xgboost.predict(x_train)

XGB_score= accuracy_score(y_train,XGB_prediction)

print('accuracy score on train using XGBoost :',XGB_score)
#XGBoost model on the test

XGB_prediction = xgboost.predict(x_test)

XGB_score= accuracy_score(y_test,XGB_prediction)

print('accuracy score on test using XGBoost :',XGB_score)
rfc2=RandomForestClassifier()

rfc2.fit(x_train,y_train)

#model on train using all the independent values in df

rfc_prediction = rfc2.predict(x_train)

rfc_score= accuracy_score(y_train,rfc_prediction)

print('accuracy score on train using random forest : ',rfc_score)

#model on test using all the indpendent values in df

rfc_prediction = rfc2.predict(x_test)

rfc_score= accuracy_score(y_test,rfc_prediction)

print('accuracy score on test using random forest:',rfc_score)
dec=DecisionTreeClassifier()

dec.fit(x_train,y_train)

#model on train using all the independent values in df

dec_prediction = dec.predict(x_train)

dec_score= accuracy_score(y_train,dec_prediction)

print('accuracyscore on train using decision tree: ',dec_score)

#model on test using all the independent values in df

dec_prediction = dec.predict(x_test)

dec_score= accuracy_score(y_test,dec_prediction)

print('accuracy score on test using decision tree: ',dec_score)
log =LogisticRegression()

log.fit(x_train,y_train)

#model on train using all the independent values in df

log_prediction = log.predict(x_train)

log_score= accuracy_score(y_train,log_prediction)

print('Accuracy score using logisitic regression on train :',log_score)

#model on train using all the independent values in df

log_prediction = log.predict(x_test)

log_score= accuracy_score(y_test,log_prediction)

print('Accuracy score using logisitic regression on test :',log_score)