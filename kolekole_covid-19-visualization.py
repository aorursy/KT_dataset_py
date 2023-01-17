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
N = 100 # timeline start at N confirmed cases
countries = ['Israel', 'Czech Republic']
# countries = ['Israel', 'France', 'Italy', 'Germany', 'Czech Republic', 'Norway', 'Iceland', 'Sweden', 'Taiwan','Singapore', 'Hong Kong', 'Spain', 'Russia', 'United Kingdom', 'UK', 'South Korea' ] ## Countries in the analysis
# countries = ['Israel', 'France', 'Italy', 'Germany', 'Czech Republic',  'Spain', 'United Kingdom', 'UK' ] ## Countries in the analysis
# countries = ['Israel', 'France', 'Italy', 'Germany', 'Spain', 'Czech Republic'] ## Countries in the analysis
df = pd.read_csv(r'/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', header=0, names=['SNo', 'Date', 'Province', 'Country', 'Update', 'Confirmed', 'Deaths', 'Recovered'])
df = df.loc[df['Country'].isin(countries)]
df = df.groupby(['Country', 'Date']).sum()
df.head(10)
dfm = df
dfm['New'] = dfm.groupby(level='Country').diff()['Confirmed']
dfm = dfm.reset_index()
dfm['Date'] = dfm['Date'].apply(pd.Timestamp)
dfm = dfm.loc[dfm['Date'] >= pd.Timestamp('03/09/2020')]
dfm['Ratio'] = dfm['New'] / dfm['Confirmed'].shift(1)
dfm = dfm.loc[dfm['Date'] >= pd.Timestamp('03/10/2020')]

dfm
fig = px.bar(dfm, x='Date', y='Ratio', color='Country', barmode='group', title='New cases ratio by date')
fig.show()
dfd = df.loc[df['Confirmed'] >= N].reset_index('Date', drop=True)
dfd = dfd.set_index(dfd.groupby(level=0).cumcount().rename('Day'), append=True)#.reset_index() # date_of_N_cases
# dfd['New'] = dfd.groupby(level='Country').diff()['Confirmed']
dfd = dfd.reset_index()
dfd['Ratio'] = dfd['New'] / dfd['Confirmed'].shift(1)
dfd['Ratio_ma'] = dfd.groupby('Country').apply(lambda x: x.rolling(window=7).mean())['Ratio']
dfd['Ratio_ma_diff'] = dfd['Ratio_ma'].diff()
dfd
fig = px.line(dfd, x='Day', y='Ratio_ma', color='Country', title='New cases ratio by day since %d cases'%N)
fig.show()
dfn = df.loc[df['Confirmed'] >= N].reset_index('Date', drop=True)
dfn = dfn.set_index(dfn.groupby(level=0).cumcount().rename('Day'), append=True)#.reset_index() # date_of_N_cases
pd.set_option('display.max_rows', dfn.shape[0]+1)
dfn.head()
dfn['Confirmed_m'] = dfn.groupby(level='Country').apply(lambda x: x.rolling(window=4).mean())['Confirmed']
dfn['New'] = dfn.groupby(level='Country').diff()['Confirmed']
dfn['New_m'] = dfn.groupby(level='Country').apply(lambda x: x.rolling(window=4).mean())['New']
dfn.head(10)
fig = go.Figure()
for country in countries:
    x = dfn.reset_index().loc[dfn.reset_index()['Country'] == country]['Confirmed_m']
    y = dfn.reset_index().loc[dfn.reset_index()['Country'] == country]['New_m']
    fig.add_trace(go.Scatter(x=x, y=y, name=country))
    fig.update_layout(xaxis_type="log", yaxis_type="log")
fig.update_layout(
    title="New cases vs. Total cases",
    xaxis_title="Total cases",
    yaxis_title="New cases")
fig.show()
fig = go.Figure()
for country in countries:
    x = dfn.reset_index().loc[dfn.reset_index()['Country'] == country]['Day']
    y = dfn.reset_index().loc[dfn.reset_index()['Country'] == country]['New_m']
    fig.add_trace(go.Scatter(x=x, y=y, name=country))
    fig.update_layout(yaxis_type="log")
fig.update_layout(
    title="New cases vs. Days",
    xaxis_title="Days since %d cases"%N,
    yaxis_title="New cases")
fig.show()
