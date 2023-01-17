import numpy as np

import pandas as pd

import seaborn as sb

import plotly.express as px

import matplotlib.pyplot as plt

from matplotlib import rcParams

from plotly.subplots import make_subplots

import plotly.graph_objects as go

import datetime as dt

import folium

from folium.plugins import MarkerCluster

from plotly.offline import iplot

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')

plt.style.use('ggplot')



import warnings

warnings.filterwarnings('ignore')







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv")

usa = pd.read_csv('../input/covid19-in-usa/us_covid19_daily.csv')
usa.head(3)
data.head(3)
data = data.rename(columns={'Country/Region':'Country'})
data.isnull().sum()
del data['Province/State']
data.shape
usa.shape
usa.columns
data.columns
usa.info()
data.info()
data['Date'] = pd.to_datetime(data['Date'])
usa['date'] = pd.to_datetime(usa['date'], format="%Y%m%d", errors='ignore')
# data_us = data[data['Country'] == 'US']

data_aus = data[data['Country'] == 'Australia']

data_china = data[data['Country'] == 'China']

data_italy = data[data['Country'] == 'Italy']

data_jp = data[data['Country'] == 'Japan']

data_swiss = data[data['Country'] == 'Switzerland']
usa[['date', 'positive']].iplot(kind='bar',

                                x='date',

                                y='positive',

                                color='blue',

                                gridcolor='white',

                                linecolor='black',

                                theme='pearl',

                                title='Confirmed Caess in USA',

                                yTitle='Count',

                                bargap=0.4,

                                opacity=0.7,

                                xTitle='Progression over Time')
usa[['date', 'positive']].iplot(kind='line',

                                x='date',

                                y='positive',

                                color='blue',

                                gridcolor='white',

                                linecolor='black',

                                theme='pearl',

                                title='Confirmed Caess in USA',

                                yTitle='Count',

                                bargap=0.4,

                                opacity=0.7,

                                xTitle='Progression over Time')
usa['active'] = usa['positive'] - (usa['death'] + usa['recovered'])
usa[['date', 'active']].iplot(kind='bar',

                                x='date',

                                y='active',

                                color='green',

                                gridcolor='white',

                                linecolor='black',

                                theme='pearl',

                                title='Active Cases in United States',

                                yTitle='Count',

                                bargap=0.4,

                                opacity=0.7,

                                xTitle='Progression over Time')
usa[['date', 'active']].iplot(kind='line',

                                x='date',

                                y='active',

                                color='green',

                                gridcolor='white',

                                linecolor='black',

                                theme='pearl',

                                title='Active Cases in United States',

                                yTitle='Count',

                                bargap=0.4,

                                opacity=0.7,

                                xTitle='Progression over Time')
usa[['date', 'death']].iplot(kind='bar',

                                  x='date',

                                  y='death',

                                color='red',

                                gridcolor='white',

                                linecolor='black',

                                theme='pearl',

                                title='Deaths in USA',

                                yTitle='Count',

                                bargap=0.4,

                                opacity=0.7,

                                xTitle='Progression over Time')
usa[['date', 'death']].iplot(kind='line',

                                  x='date',

                                  y='death',

                                color='red',

                                gridcolor='white',

                                linecolor='black',

                                theme='pearl',

                                title='Deaths in USA',

                                yTitle='Count',

                                bargap=0.4,

                                opacity=0.7,

                                xTitle='Progression over Time')
data_italy[['Date', 'Active']].iplot(kind='bar',

                                     x='Date',

                                     y='Active',

                                    color='green',

                                    gridcolor='white',

                                    linecolor='black',

                                    theme='pearl',

                                    title='Active Cases in Italy',

                                    yTitle='Count',

                                    bargap=0.4,

                                    opacity=0.7,

                                    xTitle='Progression over Time')
data_italy[['Date', 'Active']].iplot(kind='line',

                                     x='Date',

                                     y='Active',

                                    color='green',

                                    gridcolor='white',

                                    linecolor='black',

                                    theme='pearl',

                                    title='Active Cases in Italy',

                                    yTitle='Count',

                                    bargap=0.4,

                                    opacity=0.7,

                                    xTitle='Progression over Time')
data_italy[['Date', 'Deaths']].iplot(kind='bar',

                                     x='Date',

                                     y='Deaths',

                                    color='red',

                                    gridcolor='white',

                                    linecolor='black',

                                    theme='pearl',

                                    title='Deaths in Italy',

                                    yTitle='Count',

                                    bargap=0.4,

                                    opacity=0.7,

                                    xTitle='Progression over Time')
data_italy[['Date', 'Deaths']].iplot(kind='line',

                                     x='Date',

                                     y='Deaths',

                                    color='red',

                                    gridcolor='white',

                                    linecolor='black',

                                    theme='pearl',

                                    title='Deaths in Italy',

                                    yTitle='Count',

                                    bargap=0.4,

                                    opacity=0.7,

                                    xTitle='Progression over Time')
data_jp[['Date', 'Active']].iplot(kind='bar',

                                  x='Date',

                                  y='Active',

                                color='green',

                                gridcolor='white',

                                linecolor='black',

                                theme='pearl',

                                title='Active Cases in Japan',

                                yTitle='Count',

                                bargap=0.4,

                                opacity=0.7,

                                xTitle='Progression over Time')
data_jp[['Date', 'Active']].iplot(kind='line',

                                x='Date',

                                y='Active',

                                color='green',

                                gridcolor='white',

                                linecolor='black',

                                theme='pearl',

                                title='Active Cases in Japan',

                                yTitle='Count',

                                bargap=0.4,

                                opacity=0.7,

                                xTitle='Progression over Time')
data_jp[['Date', 'Deaths']].iplot(kind='bar',

                                x='Date',

                                y='Deaths',

                                color='red',

                                gridcolor='white',

                                linecolor='black',

                                theme='pearl',

                                title='Deaths in Japan',

                                yTitle='Count',

                                bargap=0.4,

                                opacity=0.7,

                                xTitle='Progression over Time')
data_jp[['Date', 'Deaths']].iplot(kind='line',

                                x='Date',

                                y='Deaths',

                                color='red',

                                gridcolor='white',

                                linecolor='black',

                                theme='pearl',

                                title='Deaths in Japan',

                                yTitle='Count',

                                bargap=0.4,

                                opacity=0.7,

                                xTitle='Progression over Time')
data_china[['Date', 'Active']].iplot(kind='bar',

                                x='Date',

                                y='Active',

                                color='green',

                                gridcolor='white',

                                linecolor='black',

                                theme='pearl',

                                title='Active Cases in China',

                                yTitle='Count',

                                bargap=0.4,

                                opacity=0.7,

                                xTitle='Progression over Time')
data_china[['Date', 'Deaths']].iplot(kind='bar',

                                x='Date',

                                y='Deaths',

                                color='red',

                                gridcolor='white',

                                linecolor='black',

                                theme='pearl',

                                title='Deaths in China',

                                yTitle='Count',

                                bargap=0.4,

                                opacity=0.7,

                                xTitle='Progression over Time')
data_aus[['Date', 'Active']].iplot(kind='bar',

                                x='Date',

                                y='Active',

                                color='green',

                                gridcolor='white',

                                linecolor='black',

                                theme='pearl',

                                title='Active Cases in Australia',

                                yTitle='Count',

                                bargap=0.4,

                                opacity=0.7,

                                xTitle='Progression over Time')
data_aus[['Date', 'Deaths']].iplot(kind='bar',

                                x='Date',

                                y='Deaths',

                                color='red',

                                gridcolor='white',

                                linecolor='black',

                                theme='pearl',

                                title='Deaths in Australia',

                                yTitle='Count',

                                bargap=0.4,

                                opacity=0.7,

                                xTitle='Progression over Time')
f, axes = plt.subplots(3, 2, figsize=(17, 17), sharex=True)



sb.lineplot(x='date', y='active', data=usa[['date', 'active']], color='red', ax=axes[0, 0]).set_title('Unites States Active Cases')

sb.lineplot(x='Date', y='Active', data=data_jp[['Date', 'Active']], color='green', ax=axes[0, 1]).set_title('Japan Active Cases')

sb.lineplot(x='Date', y='Active', data=data_aus[['Date', 'Active']], color='green', ax=axes[1, 0]).set_title('Australia Active Cases')

sb.lineplot(x='Date', y='Active', data=data_italy[['Date', 'Active']], color='green', ax=axes[1, 1]).set_title('Italy Active Cases')

sb.lineplot(x='Date', y='Active', data=data_china[['Date', 'Active']], color='green', ax=axes[2, 0]).set_title('China Active Cases')

sb.lineplot(x='Date', y='Active', data=data_swiss[['Date', 'Active']], color='green', ax=axes[2, 1]).set_title('Switzerland Active Cases')