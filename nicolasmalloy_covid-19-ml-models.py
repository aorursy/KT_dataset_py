# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import io

import requests

import seaborn as sns

import time

import datetime

from sklearn.preprocessing import OneHotEncoder# creating instance of one-hot-encoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import make_regression

from sklearn import linear_model

import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.stattools import adfuller



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



train_data = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv", encoding= 'unicode_escape', parse_dates = ['Date'])

test_data = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv", encoding= 'unicode_escape', parse_dates = ['Date'])

submission_data = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv", encoding= 'unicode_escape')
#Function that extracts year and month to create columns in dataframe

def extract_date(df,column):

    '''INPUT: df takes dataframe and column takes string'''

    df[column+"_year"] = df[column].apply(lambda x: x.year)

    df[column+"_month"] = df[column].apply(lambda x: x.month)

    df[column+"_day"] = df[column].apply(lambda x: x.day)
extract_date(train_data,'Date')

extract_date(test_data,'Date')
train_data.head()
train_df = train_data.drop('Province_State',axis=1)

train_df = train_df.drop('Date',axis=1)

train_df = train_df.drop('Id',axis=1)

train_df.tail(1)
test_df = test_data.drop('ForecastId',axis=1)

test_df = test_df.drop('Province_State',axis=1)

test_df = test_df.drop('Date',axis=1)

test_df.tail(1)
# creating initial dataframe

train_coded_df = pd.DataFrame(train_df, columns=['Country_Region'])# generate binary values using get_dummies

dum_df = pd.get_dummies(train_coded_df, columns=["Country_Region"])# merge with main df bridge_df on key values

train_df = train_df.join(dum_df)

train_df = train_df.drop('Country_Region',axis=1)

train_df.tail(1)
# creating initial dataframe

test_coded_df = pd.DataFrame(test_df, columns=['Country_Region'])# generate binary values using get_dummies

dum_df = pd.get_dummies(test_coded_df, columns=["Country_Region"])# merge with main df bridge_df on key values

test_df = test_df.join(dum_df)

test_df = test_df.drop('Country_Region',axis=1)

test_df.tail(1)
train_X = train_df.drop('ConfirmedCases',axis=1)

train_X = train_X.drop('Fatalities',axis=1)

train_X.tail(1)
y1 = pd.DataFrame(train_df['ConfirmedCases'])

y2 = pd.DataFrame(train_df['Fatalities'])

train_y = y1.join(y2)

train_y.tail(1)
regr = RandomForestRegressor()

regr.fit(train_X, train_y)

regr.predict(test_df)
submission = submission_data.join(pd.DataFrame(regr.predict(test_df)))

submission = submission.drop('ConfirmedCases',axis=1)

submission = submission.drop('Fatalities',axis=1)

submission = submission.rename(columns={0:'ConfirmedCases',1:'Fatalities'})

submission.head()
submission.to_csv('submission.csv',index=False)