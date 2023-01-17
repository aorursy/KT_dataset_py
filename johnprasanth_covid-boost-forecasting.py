#Libraries to import

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt

import pycountry

import plotly_express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots



sns.set_style('darkgrid')

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import OrdinalEncoder

from sklearn import metrics

#import xgboost as xgb

from xgboost import XGBRegressor

from sklearn import preprocessing

#from xgboost import plot_importance, plot_tree

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import AdaBoostRegressor

import lightgbm as lgb

from catboost import CatBoostRegressor
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv') 

df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
display(df_train.head())

display(df_train.describe())

display(df_train.info())
df_train['Date'] = pd.to_datetime(df_train['Date'], format = '%Y-%m-%d')

df_test['Date'] = pd.to_datetime(df_test['Date'], format = '%Y-%m-%d')
print('Minimum date from training set: {}'.format(df_train['Date'].min()))

print('Maximum date from training set: {}'.format(df_train['Date'].max()))
print('Minimum date from test set: {}'.format(df_test['Date'].min()))

print('Maximum date from test set: {}'.format(df_test['Date'].max()))
def add_daily_measures(df):

    df.loc[0,'Daily Cases'] = df.loc[0,'ConfirmedCases']

    df.loc[0,'Daily Deaths'] = df.loc[0,'Fatalities']

    for i in range(1,len(df_world)):

        df.loc[i,'Daily Cases'] = df.loc[i,'ConfirmedCases'] - df.loc[i-1,'ConfirmedCases']

        df.loc[i,'Daily Deaths'] = df.loc[i,'Fatalities'] - df.loc[i-1,'Fatalities']

    #Make the first row as 0 because we don't know the previous value

    df.loc[0,'Daily Cases'] = 0

    df.loc[0,'Daily Deaths'] = 0

    return df
df_world = df_train.copy()

df_world = df_world.groupby('Date',as_index=False)['ConfirmedCases','Fatalities'].sum()

df_world = add_daily_measures(df_world)
df_world.head()
fig = go.Figure(data=[

    go.Bar(name='Cases', x=df_world['Date'], y=df_world['Daily Cases']),

    go.Bar(name='Deaths', x=df_world['Date'], y=df_world['Daily Deaths'])

])

# Change the bar mode

fig.update_layout(barmode='overlay', title='Worldwide daily Case and Death count')

fig.show()
df_map = df_train.copy()

df_map['Date'] = df_map['Date'].astype(str)

df_map = df_map.groupby(['Date','Country_Region'], as_index=False)['ConfirmedCases','Fatalities'].sum()
df_map.head()
df_test.head()
df_train[df_train.Country_Region=='India'].Date.min()
def create_features(df):

    df['day'] = df['Date'].dt.day

    df['month'] = df['Date'].dt.month

    df['dayofweek'] = df['Date'].dt.dayofweek

    df['dayofyear'] = df['Date'].dt.dayofyear

    df['quarter'] = df['Date'].dt.quarter

    df['weekofyear'] = df['Date'].dt.weekofyear

    return df
def train_dev_split(df, days):

    #Last days data as dev set

    date = df['Date'].max() - dt.timedelta(days=days)

    return df[df['Date'] <= date], df[df['Date'] > date]
#df_train = categoricalToInteger(df_train)

df_train = create_features(df_train)
df_train, df_dev = train_dev_split(df_train,0)
df_train.head()
columns = ['day','month','dayofweek','dayofyear','quarter','weekofyear','Province_State', 'Country_Region','ConfirmedCases','Fatalities']

df_train = df_train[columns]

df_dev = df_dev[columns]
df_train.Province_State.fillna('NaN', inplace=True)

df_test.Province_State.fillna('NaN', inplace=True)

#Apply the same transformation to test set that were applied to the training set

df_test = create_features(df_test)

#Columns to select

columns = ['day','month','dayofweek','dayofyear','quarter','weekofyear']
df_train.dtypes
submission = []

#Loop through all the unique countries

for country in df_train.Country_Region.unique():

    #Filter on the basis of country

    df_train1 = df_train[df_train["Country_Region"]==country]

    #Loop through all the States of the selected country

    for state in df_train1.Province_State.unique():

        #Filter on the basis of state

        df_train2 = df_train1[df_train1["Province_State"]==state]

        #Drop unwanted columns

        df_train3 = df_train2.drop(['Country_Region','Province_State'], axis=1)

        #Convert to numpy array for training

        train = df_train3.values

        #Separate the features and labels

        X_train, y_train = train[:,:-2], train[:,-2:]

        #model1 for predicting Confirmed Cases

        #model1 = lgb.LGBMRegressor(random_state=1,n_estimators=1000)

        model1 = XGBRegressor(random_state=1,n_estimators=1000)

        model1.fit(X_train, y_train[:,0])

        #model2 for predicting Fatalities

        model2 = XGBRegressor(random_state=1,n_estimators=1000)

        #model2 = CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE')

        model2.fit(X_train, y_train[:,1])

        #Get the test data for that particular country and state

        df_test1 = df_test[(df_test["Country_Region"]==country) & (df_test["Province_State"] == state)]

        #Store the ForecastId separately

        ForecastId = df_test1.ForecastId.values

        #Remove the unwanted columns

        df_test2 = df_test1[columns]

        #Get the predictions

        y_pred1 = model1.predict(df_test2.values).astype(int)

        y_pred2 = model2.predict(df_test2.values).astype(int)

        #Append the predicted values to submission list

        for i in range(len(y_pred1)):

            d = {'ForecastId':ForecastId[i], 'ConfirmedCases':y_pred1[i], 'Fatalities':y_pred2[i]}

            submission.append(d)
df_submit = pd.DataFrame(submission)
len(submission)
df_submit.to_csv(r'submission.csv', index=False)