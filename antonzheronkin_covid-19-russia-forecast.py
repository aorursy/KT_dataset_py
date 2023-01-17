# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly

import datetime



from sklearn import linear_model

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression



from sklearn.metrics import mean_squared_error

from math import sqrt



import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")
train.describe()
train.isna().sum()
train.fillna('None')
train = train[train['Country_Region'] == 'Russia']

train.drop(['Province_State', 'Country_Region'], axis=1)
train['Date_label'] = train['Date']

train['Date'] = pd.to_datetime(train['Date'], format="%Y-%m-%d")
start_date = train.iloc[0]['Date']

train['NumberOfDays'] = [ (x - start_date).days for x in train['Date']]
test = test[test['Country_Region'] == 'Russia']

test = test.drop(['Province_State', 'Country_Region'], axis=1)
test['Date_label'] = test['Date']

test['Date'] = pd.to_datetime(test['Date'], format="%Y-%m-%d")

test['NumberOfDays'] = [ (x - start_date).days for x in test['Date']]
test
x = train['NumberOfDays'].to_numpy().reshape((-1,1))

y = train['ConfirmedCases']

model = Pipeline([('poly', PolynomialFeatures(degree=7)),

    ('linear', LinearRegression(fit_intercept=False))])

model = model.fit(x, y)

predict_x = test['NumberOfDays'].to_numpy().reshape((-1,1))

test['ConfirmedCases'] = model.predict(predict_x)
plt.figure(figsize=(15,5))

sns.set()

g = sns.lineplot(x='Date_label', y='ConfirmedCases', data=test[['Date_label', 'ConfirmedCases']])

g.set_xticklabels(test['Date_label'], rotation = 60);
x = train['NumberOfDays'].to_numpy().reshape((-1,1))

y = train['Fatalities']

model = Pipeline([('poly', PolynomialFeatures(degree=5)),

    ('linear', LinearRegression(fit_intercept=False))])

model = model.fit(x, y)

predict_x = test['NumberOfDays'].to_numpy().reshape((-1,1))

test['Fatalities'] = model.predict(predict_x)
plt.figure(figsize=(15,5))

g = sns.lineplot(x='Date_label', y='Fatalities', data=test[['Date_label', 'Fatalities']])

g.set_xticklabels(test['Date_label'], rotation = 60);
Actual = train.set_index(['Date'])

Actual = Actual.loc['2020-3-19':'2020-3-31']





Predict = test.set_index(['Date'])

Predict = Predict.loc['2020-3-19':'2020-3-31']
Y_actual = Actual['ConfirmedCases']

Y_predict = Predict['ConfirmedCases']



rmse = sqrt(mean_squared_error(Y_actual, Y_predict))
rmse
Y_actual = Actual['Fatalities']

Y_predict = Predict['Fatalities']



rmse = sqrt(mean_squared_error(Y_actual, Y_predict))
rmse