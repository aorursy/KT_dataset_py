from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import ExtraTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn import preprocessing

from scipy.stats import norm

import matplotlib.pylab as pylab

import matplotlib.pyplot as plt

from pandas import get_dummies

import matplotlib as mpl

from scipy import stats

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
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train=pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")
df_train.head()
#function for missing data

def missing_data(df_train):

    total = df_train.isnull().sum().sort_values(ascending=False)

    percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return(missing_data.head(20))
missing_data(df_train)
df_train.info()
df_train["class"]= df_train["class"].replace("p", 1)

df_train["class"]= df_train["class"].replace("e", 0)
df_train['class']=df_train['class'].astype('int')
df_train_new=df_train.drop(['class'],axis=1)
encoded = pd.get_dummies(df_train_new)

encoded.head()
dependent_all=df_train['class']

independent_all=encoded
x_train,x_test,y_train,y_test=train_test_split(independent_all,dependent_all,test_size=0.3,random_state=100)
xgboost = xgb.XGBClassifier(max_depth=3,n_estimators=300,learning_rate=0.05)
xgboost.fit(x_train,y_train)
#XGBoost modelon the train set

XGB_prediction = xgboost.predict(x_train)

XGB_score= accuracy_score(y_train,XGB_prediction)

XGB_score
#XGBoost model on the test

XGB_prediction = xgboost.predict(x_test)

XGB_score= accuracy_score(y_test,XGB_prediction)

XGB_score
rfc2=RandomForestClassifier()

rfc2.fit(x_train,y_train)

#model on train using all the independent values in df

rfc_prediction = rfc2.predict(x_train)

rfc_score= accuracy_score(y_train,rfc_prediction)

print(rfc_score)

#model on test using all the indpendent values in df

rfc_prediction = rfc2.predict(x_test)

rfc_score= accuracy_score(y_test,rfc_prediction)

print(rfc_score)
log =LogisticRegression()

log.fit(x_train,y_train)

#model on train using all the independent values in df

log_prediction = log.predict(x_train)

log_score= accuracy_score(y_train,log_prediction)

print(log_score)

#model on train using all the independent values in df

log_prediction = log.predict(x_test)

log_score= accuracy_score(y_test,log_prediction)

print(log_score)