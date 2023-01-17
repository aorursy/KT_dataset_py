# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from sklearn.linear_model import LinearRegression

import xgboost

from sklearn.svm import SVR

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

htrans = pd.read_csv("../input/historical_transactions.csv")

merc = pd.read_csv("../input/merchants.csv")

nmerc = pd.read_csv("../input/new_merchant_transactions.csv")

test = pd.read_csv("../input/test.csv")

sub = pd.read_csv("../input/sample_submission.csv")
train.head()
train.describe()
train.isnull().sum()
train.info()
train.head()
train["first_active"]=pd.to_datetime(train["first_active_month"])

train["day"]=train["first_active"].dt.month

train["year"]=train["first_active"].dt.year
test.head()
test=test.fillna("2017-09")
test["first_active"]=pd.to_datetime(test["first_active_month"])

test["day"]=test["first_active"].dt.month

test["year"]=test["first_active"].dt.year
train = train.drop(['first_active','first_active_month','card_id'],axis=1)

test  = test.drop(['first_active','first_active_month','card_id'],axis=1)

y=train["target"]

x=train.drop(["target"],axis=1)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4, random_state=11)
from math import *

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=100,random_state = 42)

rfr.fit(x_train, y_train)

y_pred=rfr.predict(x_test)

print(math.sqrt(mean_squared_error(y_test, y_pred)))
params = {}

params["objective"] = "reg:linear"

params["eta"] = 0.03

params["min_child_weight"] = 10

params["subsample"] = 0.8

params["colsample_bytree"] = 0.7

params["silent"] = 1

params["max_depth"] = 18

#params["max_delta_step"]=2

params["seed"] = 0

 #params['eval_metric'] = "auc"

plst1 = list(params.items())

num_rounds1 = 1100

import xgboost as xgb

xgdmat=xgb.DMatrix(x_train,y_train)



final_gb1=xgb.train(plst1,xgdmat,num_rounds1)



tesdmat=xgb.DMatrix(x_test)

y_pred=final_gb1.predict(tesdmat)

print(np.sqrt(mean_squared_error(y_test, y_pred)))
testmat=xgb.DMatrix(test)

answer=final_gb1.predict(testmat)
sub.head()
sub["target"]=answer
sub.to_csv("sample_submission.csv",index=False)