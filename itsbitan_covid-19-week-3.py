# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')

df_sub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')
df_train.info()
df_test.info()
df_sub.info()
df_train.shape, df_test.shape, df_sub.shape
df_train.head()
df_train['Province_State'] = df_train['Province_State'].fillna('unknown')

df_test['Province_State'] = df_test['Province_State'].fillna('unknown')
test_id = df_test['ForecastId']

df_train.drop(['Id'], axis=1, inplace=True)

df_test.drop('ForecastId', axis=1, inplace=True)
import datetime

df_train['Date'] = pd.to_datetime(df_train['Date'], infer_datetime_format=True)

df_test['Date'] = pd.to_datetime(df_test['Date'], infer_datetime_format=True)

df_train.loc[:, 'Date'] = df_train.Date.dt.strftime("%m%d")

df_train["Date"]  = df_train["Date"].astype(int)

df_test.loc[:, 'Date'] = df_test.Date.dt.strftime("%m%d")

df_test["Date"]  = df_test["Date"].astype(int)
#Lets take our target variable

y_train_cc = df_train['ConfirmedCases']

y_train_ft = df_train['Fatalities']
df_train.drop(['ConfirmedCases'], axis=1, inplace=True)

df_train.drop(['Fatalities'], axis=1, inplace=True)
#Now lets encode the catagorical variable

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

df_train['Country_Region'] = labelencoder.fit_transform(df_train['Country_Region'])

df_train['Province_State'] = labelencoder.fit_transform(df_train['Province_State'])

df_test['Country_Region'] = labelencoder.fit_transform(df_test['Country_Region'])

df_test['Province_State'] = labelencoder.fit_transform(df_test['Province_State'])
x_train = df_train.iloc[:,:].values

x_test = df_test.iloc[:,:].values
from xgboost import XGBRegressor

regressor1 = XGBRegressor(n_estimators = 1000)

regressor1.fit(x_train, y_train_cc)

y_pred_cc= regressor1.predict(x_test)
regressor2 = XGBRegressor(n_estimators = 1000)

regressor2.fit(x_train, y_train_ft)

y_pred_ft= regressor2.predict(x_test)
#Sumbmission the result

df_sub = pd.DataFrame()

df_sub['ForecastId'] = test_id

df_sub['ConfirmedCases'] = y_pred_cc

df_sub['Fatalities'] = y_pred_ft

df_sub.to_csv('submission.csv', index=False)
df_sub.head()