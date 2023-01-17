# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

from datetime import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
countries = ['France', 'Italy', 'Germany', 'Spain' ] ## Countries in the analysis

start_date = pd.Timestamp('2020-02-28')
# Read dataset

df = pd.read_csv('/kaggle/input/coronavirus-2019ncov/covid-19-all.csv', names=['Country', 'Province', 'Lat', 'Long', 'Confirmed', 'Recovered', 'Deaths', 'Date'], header=0)



# Convert Date to datetime object

df['Date'] = df['Date'].apply(pd.Timestamp)

# exclude irrelevant countries and dates 

df = df.loc[(df['Country'].isin(countries)) & (df['Date'] >= start_date)].sort_values(['Country', 'Date'])

# Sum all provinces

df_eur = df.groupby(['Country', 'Date']).sum().reset_index()

# pd.set_option('display.max_rows', df.shape[0]+1)

# df
# Add new confirmed cases

df_eur['NewConfirmed'] =  df_eur.groupby('Country')['Confirmed'].diff()

# Add ratio

df_eur['ConfirmedRt'] = df_eur['Confirmed'].div(df_eur['Confirmed'].shift(1)).replace([np.inf, -np.inf], np.nan)

df_eur['NewRt'] = df_eur['NewConfirmed'].div((df_eur['NewConfirmed'].shift(1) + df_eur['NewConfirmed'].shift(2) + df_eur['NewConfirmed'].shift(3) + df_eur['NewConfirmed'].shift(4)) / 4).replace([np.inf, -np.inf], np.nan)

# Exclude first date (divided by different country)

df_eur = df_eur.loc[df_eur['Date'] != start_date]

# df_eur
# pd.set_option('display.max_rows', df_eur.shape[0]+1) # Show all df in 1 print

# df_eur.columns





## All countries together

# Plot total cases Rt

fig = px.line(df_eur.loc[df_eur['Date'] > start_date], x="Date", y="ConfirmedRt", color='Country', labels={'x': 'Date', 'y':'Ratio'}, title='The rate of *total* confirmed cases for each country')

fig.update_xaxes(tick0=2)

fig.show()

# Plot new cases Rt

# fig = px.line(df_eur.loc[df_eur['Date'] > start_date], x="Date", y="NewRt", color='Country', labels={'x': 'Date', 'y':'Ratio'}, title='The rate of *new* confirmed cases for each country')

# fig.update_xaxes(tick0=2)

# fig.show()

## Each country seperatly

for country in countries:

    fig = px.bar(df_eur.loc[df_eur['Country'] == country], x="Date", y="NewConfirmed", labels={'x': 'Date', 'y':'Ratio'}, title='%s: New confirmed cases'%country)

    fig.show()
