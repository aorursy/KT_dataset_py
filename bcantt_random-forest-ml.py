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
center_info = pd.read_csv('/kaggle/input/av-genpact-hack-dec2018/fulfilment_center_info.csv')

train = pd.read_csv('/kaggle/input/av-genpact-hack-dec2018/train.csv')

test = pd.read_csv('/kaggle/input/av-genpact-hack-dec2018/test.csv')

meal_info = pd.read_csv('/kaggle/input/av-genpact-hack-dec2018/meal_info.csv')
center_info
train = train.merge(center_info)

train = train.merge(meal_info)



test = test.merge(center_info)

test = test.merge(meal_info)
train
from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn import linear_model

from sklearn import svm

from sklearn import tree

import xgboost as xgb

from sklearn.ensemble import BaggingRegressor

import numpy as np 

import pandas as pd 

import random
train_df = train.drop('num_orders',axis = 1)
data = pd.concat([train_df,test])
le = preprocessing.LabelEncoder()

for name in data.columns:

    if data[name].dtypes == "O":

        print(name)

        data[name] = data[name].astype(str)

        train[name] = train[name].astype(str)

        test[name] = test[name].astype(str)

        le.fit(data[name])

        train[name] = le.transform(train[name])

        test[name] = le.transform(test[name])
X = train.drop('num_orders',axis = 1)

y = train['num_orders']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
regr = RandomForestRegressor()

regr.fit(X_train, y_train)



predictions = regr.predict(X_test)
mean_squared_error(predictions, y_test)
pd.DataFrame(predictions, y_test).reset_index().corr()
train