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

from sklearn import preprocessing

from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.arima_model import ARIMA

from random import random

from warnings import filterwarnings

from sklearn import preprocessing

from sklearn.ensemble.forest import RandomForestClassifier



path_week4 = '/kaggle/input/covid19-global-forecasting-week-4'

df_train = pd.read_csv(f'{path_week4}/train.csv',parse_dates=['Date'], engine='python')

df_test = pd.read_csv(f'{path_week4}/test.csv')

#df_train = pd.read_csv('train.csv')

#df_test = pd.read_csv('test.csv')

df_train.head()

df_test.head()

df_train.rename(columns={'Country_Region':'Country'}, inplace=True)

df_test.rename(columns={'Country_Region':'Country'}, inplace=True)



df_train.rename(columns={'Province_State':'State'}, inplace=True)

df_test.rename(columns={'Province_State':'State'}, inplace=True)



df_train['Date'] = pd.to_datetime(df_train['Date'], infer_datetime_format=True)

df_test['Date'] = pd.to_datetime(df_test['Date'], infer_datetime_format=True)



df_train.info()

df_test.info()



y1_Train = df_train.iloc[:, -2]

y1_Train.head()

y2_Train = df_train.iloc[:, -1]

y2_Train.head()



EMPTY_VAL = "EMPTY_VAL"



def fillState(state, country):

    if state == EMPTY_VAL: return country

    return state

#X_Train = df_train.loc[:, ['State', 'Country', 'Date']]

X_Train = df_train.copy()



X_Train['State'].fillna(EMPTY_VAL, inplace=True)

X_Train['State'] = X_Train.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)



X_Train.loc[:, 'Date'] = X_Train.Date.dt.strftime("%m%d")

X_Train["Date"]  = X_Train["Date"].astype(int)



X_Train.head()



#X_Test = df_test.loc[:, ['State', 'Country', 'Date']]

X_Test = df_test.copy()



X_Test['State'].fillna(EMPTY_VAL, inplace=True)

X_Test['State'] = X_Test.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)



X_Test.loc[:, 'Date'] = X_Test.Date.dt.strftime("%m%d")

X_Test["Date"]  = X_Test["Date"].astype(int)



X_Test.head()





le = preprocessing.LabelEncoder()



X_Train.Country = le.fit_transform(X_Train.Country)

X_Train['State'] = le.fit_transform(X_Train['State'])



X_Train.head()



X_Test.Country = le.fit_transform(X_Test.Country)

X_Test['State'] = le.fit_transform(X_Test['State'])



X_Test.head()



df_train.head()

df_train.loc[df_train.Country == 'Afghanistan', :]

df_test.tail()



filterwarnings('ignore')







le = preprocessing.LabelEncoder()



countries = X_Train.Country.unique()



#models_C = {}

#models_F = {}



df_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})



for country in countries:

    states = X_Train.loc[X_Train.Country == country, :].State.unique()

    #print(country, states)

    # check whether string is nan or not

    for state in states:

        X_Train_CS = X_Train.loc[(X_Train.Country == country) & (X_Train.State == state), ['State', 'Country', 'Date', 'ConfirmedCases', 'Fatalities']]

        

        y1_Train_CS = X_Train_CS.loc[:, 'ConfirmedCases']

        y2_Train_CS = X_Train_CS.loc[:, 'Fatalities']

        

        X_Train_CS = X_Train_CS.loc[:, ['State', 'Country', 'Date']]

        

        X_Train_CS.Country = le.fit_transform(X_Train_CS.Country)

        X_Train_CS['State'] = le.fit_transform(X_Train_CS['State'])

        

        X_Test_CS = X_Test.loc[(X_Test.Country == country) & (X_Test.State == state), ['State', 'Country', 'Date', 'ForecastId']]

        

        X_Test_CS_Id = X_Test_CS.loc[:, 'ForecastId']

        X_Test_CS = X_Test_CS.loc[:, ['State', 'Country', 'Date']]

        

        X_Test_CS.Country = le.fit_transform(X_Test_CS.Country)

        X_Test_CS['State'] = le.fit_transform(X_Test_CS['State'])

        

        #models_C[country] = gridSearchCV(model, X_Train_CS, y1_Train_CS, param_grid, 10, 'neg_mean_squared_error')

        #models_F[country] = gridSearchCV(model, X_Train_CS, y2_Train_CS, param_grid, 10, 'neg_mean_squared_error')

        #classifier1 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

        classifier1 = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)

        classifier1.fit(X_Train_CS, y1_Train_CS)

        Y1_pred = classifier1.predict(X_Test_CS)

        

        classifier2 = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)

        classifier2.fit(X_Train_CS, y2_Train_CS)

        Y2_pred = classifier2.predict(X_Test_CS)

        

        df = pd.DataFrame({'ForecastId': X_Test_CS_Id, 'ConfirmedCases': Y1_pred, 'Fatalities': Y2_pred})

        df_out = pd.concat([df_out, df], axis=0)

    # Done for state loop

# Done for country Loop

df_out.ForecastId = df_out.ForecastId.astype('int')

df_out.tail()

df_out.to_csv('submission.csv', index=False)