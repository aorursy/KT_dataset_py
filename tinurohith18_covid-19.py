import pandas as pd

from pandas import DataFrame, Series

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')
# import os

# dir = 'D:\Covid-19\covid19-global-forecasting-week-1'

# os.chdir(dir)

# data = pd.read_csv('train.csv')
data.shape
data.columns
data.head()
data.isnull().sum()/data.shape[0]*100
data = data.drop('Province/State', axis=1)
data.head()
data.tail()
data['Date'] = pd.to_datetime(data['Date'])
data.index = data.Date
data = data.drop('Date', axis =1)

data = data.reset_index()

data.head()
import plotly 

import plotly.express as px

import plotly.graph_objects as go
grp = data.groupby('Country/Region')['Country/Region', 'ConfirmedCases', 'Fatalities'].sum().reset_index()

fig = px.bar(grp, x = 'Country/Region', y = 'ConfirmedCases', title = 'Confirmed rate acroos country', log_y = True)

fig.show()
fig = px.bar(grp, x = 'Country/Region', y = 'Fatalities', title = 'Fatalities rate acroos country', log_y = True)

fig.show()
grp = data.groupby('Date')['Date', 'ConfirmedCases', 'Fatalities'].sum().reset_index()

fig = px.line(grp, x = 'Date', y = 'ConfirmedCases', title = 'Confirmed rate across country')

fig.show()
grp = data.groupby('Date')['Date', 'ConfirmedCases', 'Fatalities'].sum().reset_index()

fig = px.line(grp, x = 'Date', y = 'ConfirmedCases', title = 'Fatalities rate across country')

fig.show()
fig = go.Figure([go.Bar(x=pd.DataFrame(data[data['Country/Region']==i].groupby('Date')['ConfirmedCases'].sum()).index,

                        y=pd.DataFrame(data[data['Country/Region']==i].groupby('Date')['ConfirmedCases'].sum()).ConfirmedCases, name=str(i)) 

                 for i in data['Country/Region'].unique()],

               go.Layout(yaxis={'title': 'Total ConfirmedCases'},

                        title= ' ConfirmedCases for each Country/Region across days '))

fig.show()
fig = go.Figure([go.Bar(x=pd.DataFrame(data[data['Country/Region']==i].groupby('Date')['Fatalities'].sum()).index,

                        y=pd.DataFrame(data[data['Country/Region']==i].groupby('Date')['Fatalities'].sum()).Fatalities, name=str(i)) 

                 for i in data['Country/Region'].unique()],

               go.Layout(yaxis={'title': 'Total Fatalities'},

                        title= ' Fatalities for each Country/Region across days '))

fig.show()
data.head()
import xgboost as xgb

data.index = data.Date

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data['Country/Region'] = le.fit_transform(data['Country/Region'])

data1 = data

data = data.drop(['Date','ConfirmedCases'], axis =1)

train = data[0:int(data.shape[0]*0.8)]

val = data[int(data.shape[0]*0.8): ]
data1 = data1.drop(['Date','Fatalities'], axis =1)

train1 = data1[0:int(data1.shape[0]*0.8)]

val1 = data1[int(data1.shape[0]*0.8): ]
data.dtypes
print(data.shape)

print(train.shape)

print(val.shape)
print(data1.shape)

print(train1.shape)

print(val1.shape)
train_x = train.drop('Fatalities', axis=1)

val_x = val.drop('Fatalities', axis=1)

train_y = train[['Fatalities']]

val_y = val[['Fatalities']]
train1.head()
train1_x = train1.drop('ConfirmedCases', axis=1)

val1_x = val1.drop('ConfirmedCases', axis=1)

train1_y = train1[['ConfirmedCases']]

val1_y = val1[['ConfirmedCases']]
model = xgb.XGBRegressor()

model1 = xgb.XGBRegressor()
train1.head()
model.fit(train_x, train_y, eval_set = [(val_x, val_y)])

model1.fit(train1_x, train1_y, eval_set = [(val1_x, val1_y)])
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')

test.Date = pd.to_datetime(test.Date)

test = test.drop(['Province/State', 'Date'], axis =1)

test['Country/Region'] = le.transform(test['Country/Region'])
test.shape
train.head()
test.head()
test.rename(columns = {"ForecastId": 'Id'}, inplace=True)
test.head()
a = model.predict(test)

b = model1.predict(test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mean_absolute_error(model.predict(test), data[0:12212]['Fatalities'])
mean_absolute_error(model1.predict(test), data1[0:12212]['ConfirmedCases'])
test['Fatalities'] = a

test['ConfirmedCases'] = b
test.rename(columns = {"Id": 'ForecastId'}, inplace=True)

test.head()
test['Country/Region'] = le.inverse_transform(test['Country/Region'])
test.head()
test = test[['ForecastId', 'ConfirmedCases', 'Fatalities']]
test.head()
test.shape
test = test[['ForecastId', 'ConfirmedCases', 'Fatalities']]
test.to_csv('submission.csv',index= False)