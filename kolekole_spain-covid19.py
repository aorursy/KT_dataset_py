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
## Read and clean data

df = pd.read_excel('/kaggle/input/covid19spain/COVID_19_SPAIN_OFFICIAL_DATA.xlsx', names=['Date', 'Region', 'TotalCases', 'TotalDeaths', 'URL'], header=0)

df = df[['Date', 'Region', 'TotalCases', 'TotalDeaths']] # Drop URL

pd.set_option('display.max_rows', df.shape[0]+1)

df = df.sort_values(['Region', 'Date'])

df['NewCases'] = df.sort_values(['Region', 'Date']).groupby('Region')['TotalCases'].diff()



regions = ['Madrid','País Vasco', 'Murcia']

start_date = pd.Timestamp("2020-02-28")

df_early = df.loc[(df['Region'].isin(regions)) & (df['Date'] >= start_date)].groupby('Date').sum().reset_index()

df_late = df.loc[(~df['Region'].isin(regions)) & (df['Date'] >= start_date)].groupby('Date').sum().reset_index()

df_sum = df.loc[df['Date'] >= start_date].groupby('Date').sum().reset_index()

df_early['Isolation'] = 'Early isolation measurements'

df_late['Isolation'] = 'Later isolation measurements'

df_sum['Isolation'] = 'All of Spain'

df_early['NewCasesRate'] = df_early['NewCases'].div((df_early['NewCases'].shift(1) + df_early['NewCases'].shift(2) + df_early['NewCases'].shift(3) + df_early['NewCases'].shift(4)) / 4).replace([np.inf, -np.inf], np.nan)

df_early['TotalCasesRate'] = df_early['TotalCases'].div(df_early['TotalCases'].shift(1)).replace([np.inf, -np.inf], np.nan)

df_early = df_early.loc[df_late['Date'] > start_date]



df_late['NewCasesRate'] = df_late['NewCases'].div((df_late['NewCases'].shift(1) + df_late['NewCases'].shift(2) + df_late['NewCases'].shift(3) + df_late['NewCases'].shift(4)) / 4).replace([np.inf, -np.inf], np.nan)

df_late['TotalCasesRate'] = df_late['TotalCases'].div(df_late['TotalCases'].shift(1)).replace([np.inf, -np.inf], np.nan)

df_late = df_late.loc[df_late['Date'] > start_date]



df_sum['NewCasesRate'] = df_sum['NewCases'].div((df_sum['NewCases'].shift(1) + df_sum['NewCases'].shift(2) + df_sum['NewCases'].shift(3) + df_sum['NewCases'].shift(4)) / 4).replace([np.inf, -np.inf], np.nan)

df_sum['TotalCasesRate'] = df_sum['TotalCases'].div(df_sum['TotalCases'].shift(1)).replace([np.inf, -np.inf], np.nan)

df_sum = df_sum.loc[df_sum['Date'] > start_date]



df_all = pd.concat([df_sum, df_early, df_late])

fig = px.line(df_all, x="Date", y="TotalCasesRate", color='Isolation', labels={'x': 'Date', 'y':'Ratio'}, title='Spain: The rate of *total* confirmed cases in the regions with different isolation measurements')

fig.show()

fig = px.line(df_all, x="Date", y="NewCasesRate", color='Isolation', labels={'x': 'Date', 'y':'Ratio'}, title='Spain: The rate of *new* confirmed cases in the regions with different isolation measurements')

fig.show()