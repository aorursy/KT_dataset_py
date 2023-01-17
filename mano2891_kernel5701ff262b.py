# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd 



import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import csv

import numpy as np

import operator

import random

import datetime as dt





import sklearn.discriminant_analysis

import sklearn.linear_model as skl_lm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import confusion_matrix, classification_report, precision_score

from sklearn import preprocessing

from sklearn import neighbors

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import PolynomialFeatures

from sklearn import metrics

from datetime import timedelta

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.metrics import hamming_loss, accuracy_score 

from pandas import DataFrame

from datetime import datetime

from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error



%matplotlib inline

import matplotlib.pyplot as plt



import statsmodels.api as sm

import statsmodels.formula.api as smf
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
date = pd.to_datetime(train["Date"])

datet = pd.to_datetime(test["Date"])

print (date)

train['Date'] = pd.to_datetime(train.Date)

test['Date'] = pd.to_datetime(test.Date)
ldate = int(len(date))

ldatet = int(len(datet))

print("Length of training- date is", ldate)

print("Length of test- date is", ldatet)
train['month'] = train['Date'].dt.month

train['day'] = train['Date'].dt.day

test['month'] = test['Date'].dt.month

test['day'] = test['Date'].dt.day
train.head()
print("Datatrain")

traindays = train['Date'].nunique()

print("Number of Country_Region: ", train['Country_Region'].nunique())

print("Number of Province_State: ", train['Province_State'].nunique())

print("Number of Days: ", traindays)



notrain = train['Id'].nunique()

print("Number of datapoints in train:", notrain)

lotrain = int(notrain/traindays)

print("L Trains:", lotrain)

print("Datatest")

testdays = test['Date'].nunique()

print("Number of Days: ", testdays)

notest = test['ForecastId'].nunique()

print("Number of datapoints in test:", notest)

lotest = int(notest/testdays)

print("L Test:", lotest)
zt = datet[0]

daycount = []

for i in range(0,lotrain):

    for j in range(1,traindays+1):

        daycount.append(j)



print(daycount)

print(zt)
for i in range(traindays):

    if(zt == date[i]):

        zx = i

        print(zx)

        

daytest = []

for i in range(0,lotest):

    for j in range(1,testdays+1):

        jr = zx + j

        daytest.append(jr)

print(daytest)
train.insert(8,"DayCount",daycount,False)

test.insert(6,"DayCount",daytest,False)
traincount = int(len(train["Date"]))

testcount = int(len(test["Date"]))

print(traincount,testcount)
train.Province_State = train.Province_State.fillna(0)

empty = 0

for i in range(0,traincount):

    if(train.Province_State[i] == empty):

        train.Province_State[i] = train.Country_Region[i]
test.Province_State = test.Province_State.fillna(0)

empty = 0

for i in range(0,testcount):

    if(test.Province_State[i] == empty):

        test.Province_State[i] = test.Country_Region[i]
label = preprocessing.LabelEncoder()

train.Country_Region = label.fit_transform(train.Country_Region)

train.Province_State = label.fit_transform(train.Province_State)
test.Country_Region = label.fit_transform(test.Country_Region)

test.Province_State = label.fit_transform(test.Province_State)

train.rename({'Id': 'ForecastId'}, axis=1, inplace=True)
forecastid = test['ForecastId']

cases = train.ConfirmedCases

fatalities = train.Fatalities

train = train.drop(['ForecastId', 'Date','ConfirmedCases','Fatalities'], axis = 1)

test = test.drop(['ForecastId', 'Date'], axis = 1)
lr_model = LinearRegression()

lr_modelfat = LinearRegression()

lr_model.fit(train, cases)

lr_modelfat.fit(train,fatalities)
ypredtr = lr_model.predict(train)

ypredtrft = lr_modelfat.predict(train)

err = mean_squared_error(ypredtr,cases)

err1 = mean_squared_error(ypredtrft,fatalities)

print(err,err1)
regr_cs = XGBRegressor(n_estimators = 2500 , gamma = 0, learning_rate = 0.04,  random_state = 42 , max_depth = 23)

regr_ft = XGBRegressor(n_estimators = 2500 , gamma = 0, learning_rate = 0.04,  random_state = 42 , max_depth = 23)
regr_cs.fit(train,cases)

ypred = regr_cs.predict(train)

err = mean_squared_error(ypred,cases)

print("Training - Mean Squared Error is: ",err)
ypred1 = regr_cs.predict(test)
submission = pd.DataFrame({"ForecastId": forecastid, "ConfirmedCases": ypred1},

                          columns=["ForecastId", "ConfirmedCases"])
regr_ft.fit(train,fatalities)

ypred2= regr_ft.predict(train)

yptest = regr_ft.predict(test)

error = mean_squared_error(ypred2,fatalities)

print("Training - (Fatalities) Mean Squared Error is", error)
submission['Fatalities'] = yptest

submission = round(submission)

submission.head(10)
submission.to_csv('submission.csv', index=False)