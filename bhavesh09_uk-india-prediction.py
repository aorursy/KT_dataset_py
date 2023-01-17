#Importign libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import plotly.express as px

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot, plot
#Read Files block

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')

#Rename Columns

train = train.rename(columns={'Province/State' : 'State', 'Country/Region' : 'Country'})

test = test.rename(columns={'Province/State' : 'State', 'Country/Region' : 'Country'})
'''

train_uk_dict={"Id" : train.Id.max()+1,

              'State' : 'United Kingdom'   , 

              'Country' : 'United Kingdom',

               'Lat' : '', 

               'Long' : '', 

               'Date' : '2020-03-24', 

               'ConfirmedCases' : 8077,

                'Fatalities' : 422}

train = train.append([train_uk_dict])                

'''
# Groupby and adding lag and active new columns

train = train.groupby(['Country','Date'])[['ConfirmedCases','Fatalities']].sum().reset_index()

train['Prev_day_cc'] = train.groupby('Country')['ConfirmedCases'].shift(1)

train['Prev_day_cc'] = train['Prev_day_cc'].fillna(0)

train['New_active'] = train['ConfirmedCases'] - train['Prev_day_cc']



train['Prev_day_deaths'] = train.groupby('Country')['Fatalities'].shift(1)

train['Prev_day_deaths'] = train['Prev_day_deaths'].fillna(0)

train['New_deaths'] = train['Fatalities'] - train['Prev_day_deaths']



train['Date'] = pd.to_datetime( train['Date'])
train['day_since_1000_cc'] = train[train.ConfirmedCases>=1000].groupby('Country')['Date'].rank()

train['day_since_10_d'] = train[train.Fatalities>=10].groupby('Country')['Date'].rank()
'''train[(train.Country=='United Kingdom') & (train.day_since_10_d>=1)][['Country','Date','day_since_10_d','Fatalities']].merge(

    train[(train.Country=='Switzerland') & (train.day_since_10_d>=1)][['Country','Date','day_since_10_d','Fatalities']],

    how='right',

    on='day_since_10_d')

'''
trace1 = go.Scatter(x=train[train.Country=='United Kingdom'].Date,

                   y=train[train.Country=='United Kingdom'].ConfirmedCases,

                   mode="lines+markers",

                   name="Total Number of Confirmed Cases")

trace2 = go.Bar(x=train[train.Country=='United Kingdom'].Date,

                y=train[train.Country=='United Kingdom'].New_active,

                name="Confirmed Cases of Respactive Day")

layout = {'template' : "plotly_dark",

         "legend.orientation" : "h"}

fig = dict(data=[trace1,trace2], layout=layout)

iplot(fig)
trace1 = go.Scatter(x=train[train.Country=='United Kingdom'].Date,

                   y=train[train.Country=='United Kingdom'].Fatalities,

                   mode="lines+markers",

                   name="Total Number of Fatalities")

trace2 = go.Bar(x=train[train.Country=='United Kingdom'].Date,

                y=train[train.Country=='United Kingdom'].New_deaths,

                name="Confirmed Cases of Respactive Day")

layout = {'template' : "plotly_dark",

         "legend.orientation" : "h"}

fig = dict(data=[trace1,trace2], layout=layout)

iplot(fig)
trace1 = go.Scatter(x=train[train.Country=='India'].Date,

                   y=train[train.Country=='India'].Fatalities,

                   mode="lines+markers",

                   name="Total Number of Fatalities")

trace2 = go.Bar(x=train[train.Country=='India'].Date,

                y=train[train.Country=='India'].New_deaths,

                name="Confirmed Cases of Respactive Day")

layout = {'template' : "plotly_dark",

         "legend.orientation" : "h"}

fig = dict(data=[trace1,trace2], layout=layout)

iplot(fig)
trace1 = go.Scatter(x=train[train.Country=='India'].Date,

                   y=train[train.Country=='India'].ConfirmedCases,

                   mode="lines+markers",

                   name="Total Number of Confirmed Cases")

trace2 = go.Bar(x=train[train.Country=='India'].Date,

                y=train[train.Country=='India'].New_active,

                name="Confirmed Cases of Respactive Day")

layout = {'template' : "plotly_dark",

         "legend.orientation" : "h"}

fig = dict(data=[trace1,trace2], layout=layout)

iplot(fig)
'''

trace1 = go.Scatter(x=train[train.day_since_1000_cc>0].day_since_1000_cc,

                   y=train[train.day_since_1000_cc>0].ConfirmedCases,

                   mode="lines+markers",

                   name="Total Number of Confirmed Cases",

                   color='Country')

layout = {'template' : "plotly_dark",

         "legend.orientation" : "h",

         "title" : "Confirmed cases since they crossed 1000 cases for every country",

         "xaxis_title" : "Day since confirmed cases crossed 1000",

         "yaxis_title" : "No. of Deaths"}

fig = dict(data=[trace1], layout=layout)

iplot(fig)

'''



fig = px.line(train[train.day_since_1000_cc>0], x="day_since_1000_cc", y="ConfirmedCases", color='Country')

fig.update_layout(title='Confirmed cases since they crossed 1000 cases for every country',

                   xaxis_title='Day since confirmed cases crossed 1000',

                   yaxis_title='No. of Deaths',

                   template = "plotly_dark")

fig.show()
poi_countries = train[train.day_since_10_d>=1].Country.unique() # Only considerign countries which crossed death toll 10 before 7 days at least

fig = px.line(train[(train.day_since_10_d.between(1,30)) & (train.Country.isin(poi_countries))], 

              x="day_since_10_d", y="Fatalities", color='Country')

fig.update_layout(title='VIRUS PANDEMIC',

                   xaxis_title='Number of Days Since 10th Death',

                   yaxis_title='No. of Deaths',

                   template = "plotly_dark")

fig.show()
poi_countries = train[train.day_since_1000_cc>7].Country.unique()

fig = px.scatter(train[(train.day_since_1000_cc<=7) & train.Country.isin(poi_countries)], 

              x="day_since_1000_cc", y="Fatalities", color='Country',

             animation_frame="day_since_1000_cc", range_y=[0,300], range_x=[0,7])

fig.show()