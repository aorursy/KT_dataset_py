#Importign libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import plotly.express as px
#Read Files block

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')

#Rename Columns

train = train.rename(columns={'Province/State' : 'State', 'Country/Region' : 'Country'})

test = test.rename(columns={'Province/State' : 'State', 'Country/Region' : 'Country'})
# Groupby and adding lag and active new columns

train = train.groupby(['Country','Date'])[['ConfirmedCases','Fatalities']].sum().reset_index()

train['Prev_day_cc'] = train.groupby('Country')['ConfirmedCases'].shift(1)

train['Prev_day_cc'] = train['Prev_day_cc'].fillna(0)

train['New_active'] = train['ConfirmedCases'] - train['Prev_day_cc']

train['Date'] = pd.to_datetime( train['Date'])
train['day_since_1000_cc'] = train[train.ConfirmedCases>=1000].groupby('Country')['Date'].rank()

train['day_since_10_d'] = train[train.Fatalities>=10].groupby('Country')['Date'].rank()
train['Country'].unique()
df_korea = train[train['Country']=='Korea, South']

df_korea
fig = px.line(train[train.day_since_1000_cc>0], x="Date", y="ConfirmedCases")

fig.update_layout(title='Confirmed cases since they crossed 1000 cases',

                   xaxis_title='Date',

                   yaxis_title='No. of Deaths')

fig.show()
sub = pd.read_csv("/kaggle/input/submission/submission.csv")

sub
sub = pd.read_csv("/kaggle/input/submission/submission.csv")

submission.head()

submission.to_csv("submission.csv", index=False)