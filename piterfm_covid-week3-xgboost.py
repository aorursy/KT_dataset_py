import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from itertools import cycle, islice

import seaborn as sb

import matplotlib.dates as dates

import datetime as dt

from sklearn import preprocessing

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go

import plotly_express as px

from sklearn.preprocessing import OrdinalEncoder
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/test.csv")

train.head()
train.info()
train['Date'] = pd.to_datetime(train['Date'], format = '%Y-%m-%d')

test['Date'] = pd.to_datetime(test['Date'], format = '%Y-%m-%d')
def create_features(df):

    df['day'] = df['Date'].dt.day

    df['month'] = df['Date'].dt.month

    df['dayofweek'] = df['Date'].dt.dayofweek

    df['dayofyear'] = df['Date'].dt.dayofyear

    df['quarter'] = df['Date'].dt.quarter

    df['weekofyear'] = df['Date'].dt.weekofyear

    return df
def categoricalToInteger(df):

    #convert NaN Province State values to a string

    df.Province_State.fillna('NaN', inplace=True)

    #Define Ordinal Encoder Model

    oe = OrdinalEncoder()

    df[['Province_State','Country_Region']] = oe.fit_transform(df.loc[:,['Province_State','Country_Region']])

    return df
df_train = categoricalToInteger(train)

df_train.info()

df_train = create_features(df_train)
columns = ['day','month','dayofweek','dayofyear','quarter','weekofyear','Province_State', 'Country_Region','ConfirmedCases','Fatalities']

df_train = df_train[columns]
df_test = categoricalToInteger(test)

df_test = create_features(test)

#Columns to select

columns = ['day','month','dayofweek','dayofyear','quarter','weekofyear','Province_State', 'Country_Region']
df_test
df_train.Country_Region.unique().tolist()[0]
submission = []

#Loop through all the unique countries

for country in df_train.Country_Region.unique():

    #Filter on the basis of country

    df_train1 = df_train[df_train["Country_Region"]==country]

    #Loop through all the States of the selected country

    for state in df_train1.Province_State.unique():

        #Filter on the basis of state

        df_train2 = df_train1[df_train1["Province_State"]==state]

        #Convert to numpy array for training

        train = df_train2.values

        #Separate the features and labels

        X_train, y_train = train[:,:-2], train[:,-2:]

        #model1 for predicting Confirmed Cases

        model1 = XGBRegressor(n_estimators=1100)

        model1.fit(X_train, y_train[:,0])

        #model2 for predicting Fatalities

        model2 = XGBRegressor(n_estimators=1100)

        model2.fit(X_train, y_train[:,1])

        #Get the test data for that particular country and state

        df_test1 = df_test[(df_test["Country_Region"]==country) & (df_test["Province_State"] == state)]

        #Store the ForecastId separately

        ForecastId = df_test1.ForecastId.values

        #Remove the unwanted columns

        df_test2 = df_test1[columns]

        #Get the predictions

        y_pred1 = np.round(model1.predict(df_test2.values),5)

        y_pred2 = np.round(model2.predict(df_test2.values),5)

        #Append the predicted values to submission list

        for i in range(len(y_pred1)):

            d = {'ForecastId':ForecastId[i], 'ConfirmedCases':y_pred1[i], 'Fatalities':y_pred2[i]}

            submission.append(d)
df_submit = pd.DataFrame(submission)



df_submit.to_csv(r'submission.csv', index=False)