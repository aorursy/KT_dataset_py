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

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from sklearn.neighbors import KNeighborsClassifier

from sklearn import preprocessing

from warnings import filterwarnings

from sklearn import preprocessing

from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.arima_model import ARIMA

from random import random



path_week4 = '/kaggle/input/covid19-global-forecasting-week-4'

train_dataset = pd.read_csv(f'{path_week4}/train.csv',parse_dates=['Date'], engine='python')

test_dataset = pd.read_csv(f'{path_week4}/test.csv')

train_dataset['Date'] = pd.to_datetime(train_dataset['Date'], infer_datetime_format=True)

test_dataset['Date'] = pd.to_datetime(test_dataset['Date'], infer_datetime_format=True)

y1_Train = train_dataset.iloc[:, -2]

y2_Train = train_dataset.iloc[:, -1]

EMPTY_VALUE = "EMPTY_VALUE"

def fillState(state, country):

    if state == EMPTY_VALUE: return country

    return state

X_Train = train_dataset.copy()

X_Train['Province_State'].fillna(EMPTY_VALUE, inplace=True)

X_Train['Province_State'] = X_Train.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)

X_Train.loc[:, 'Date'] = X_Train.Date.dt.strftime("%m%d")

X_Train["Date"]  = X_Train["Date"].astype(int)

X_Test = test_dataset.copy()

X_Test['Province_State'].fillna(EMPTY_VALUE, inplace=True)

X_Test['Province_State'] = X_Test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)

X_Test.loc[:, 'Date'] = X_Test.Date.dt.strftime("%m%d")

X_Test["Date"]  = X_Test["Date"].astype(int)

label_encoder = preprocessing.LabelEncoder()

X_Train.Country_Region = label_encoder.fit_transform(X_Train.Country_Region)

X_Train['Province_State'] = label_encoder.fit_transform(X_Train['Province_State'])

X_Train.head()

X_Test.Country_Region = label_encoder.fit_transform(X_Test.Country_Region)

X_Test['Province_State'] = label_encoder.fit_transform(X_Test['Province_State'])

train_dataset.loc[train_dataset.Country_Region == 'Afghanistan', :]

filterwarnings('ignore')

label_encoder = preprocessing.LabelEncoder()

countries = X_Train.Country_Region.unique()

final_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

for country in countries:

    states = X_Train.loc[X_Train.Country_Region == country, :].Province_State.unique()

    for state in states:

        X_Train_CS = X_Train.loc[(X_Train.Country_Region == country) & (X_Train.Province_State == state), ['Province_State', 'Country_Region', 'Date', 'ConfirmedCases', 'Fatalities']]

        y1_Train_CS = X_Train_CS.loc[:, 'ConfirmedCases']

        y2_Train_CS = X_Train_CS.loc[:, 'Fatalities']        

        X_Train_CS = X_Train_CS.loc[:, ['Province_State', 'Country_Region', 'Date']]

        X_Train_CS.Country_Region = label_encoder.fit_transform(X_Train_CS.Country_Region)

        X_Train_CS['Province_State'] = label_encoder.fit_transform(X_Train_CS['Province_State'])

        X_Test_CS = X_Test.loc[(X_Test.Country_Region == country) & (X_Test.Province_State == state), ['Province_State', 'Country_Region', 'Date', 'ForecastId']]

        X_Test_CS_Id = X_Test_CS.loc[:, 'ForecastId']

        X_Test_CS = X_Test_CS.loc[:, ['Province_State', 'Country_Region', 'Date']]

        X_Test_CS.Country_Region = label_encoder.fit_transform(X_Test_CS.Country_Region)

        X_Test_CS['Province_State'] = label_encoder.fit_transform(X_Test_CS['Province_State'])

        #model1 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

        model1 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

        model1.fit(X_Train_CS, y1_Train_CS)

        y1_pred = model1.predict(X_Test_CS)

        #model2 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

        model2= KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

        model2.fit(X_Train_CS, y2_Train_CS)

        y2_pred = model2.predict(X_Test_CS)

        df = pd.DataFrame({'ForecastId': X_Test_CS_Id, 'ConfirmedCases': y1_pred, 'Fatalities': y2_pred})

        final_out = pd.concat([final_out, df], axis=0)

    # Done for state loop

# Done for country Loop

final_out.ForecastId = final_out.ForecastId.astype('int')

final_out.tail()

final_out.to_csv('submission.csv', index=False)