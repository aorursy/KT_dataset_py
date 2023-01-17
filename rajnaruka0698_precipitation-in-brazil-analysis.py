import numpy as np ## For Linear Algebra

import pandas as pd ## For Working with data

import plotly.express as px ## Visualization

import plotly.graph_objects as go ## Visualization

import plotly as py  ## Visulization

import matplotlib.pyplot as plt ## Visulization

from scipy import stats ## for stats.

import os
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

path = os.path.join(dirname, filename)



df = pd.read_csv(path)
df.head()
df.loc[:,'date'] = pd.to_datetime(df['date'])  ## Changing into datetime object helps in many ways, we'll see ahead.
## Now we can extract year and month from datetime object.

df['Year'] = df['date'].dt.year

df['Month'] = df['date'].dt.month
fig = py.subplots.make_subplots(1,2, subplot_titles=['Total Rain', 'Median rain on a Rainy day'])



temp = df.groupby(by=['state'])['precipitation'].sum().reset_index()

trace0 = go.Bar(x=temp['state'], y=temp['precipitation'])



temp = df.groupby(by=['state'])['precipitation'].median().reset_index()

trace1 = go.Bar(x=temp['state'], y=temp['precipitation'])



fig.append_trace(trace0,1,1)

fig.append_trace(trace1,1,2)



fig.show()
temp = df.groupby(by=['state', 'Year'])['precipitation'].sum().reset_index()

temp = temp.groupby(by='state')['precipitation'].median().reset_index()

px.bar(temp, 'state', 'precipitation', title='median Rain in an year', color='precipitation',

      color_continuous_scale=px.colors.sequential.Blugrn)
fig = py.subplots.make_subplots(1,2, subplot_titles=['RJ', 'PI'])



rj = df[df['state']=='RJ']

temp = rj.groupby(by='Year')['precipitation'].sum().reset_index()

trace0 = go.Scatter(x=temp['Year'], y=temp['precipitation'])



pi = df[df['state']=='PI']

temp1 = pi.groupby(by='Year')['precipitation'].sum().reset_index()

trace1 = go.Scatter(x=temp1['Year'], y=temp1['precipitation'])



fig.append_trace(trace0,1,1)

fig.append_trace(trace1,1,2)



fig.show()
fig = py.subplots.make_subplots(1,2, subplot_titles=['Total Rain in Year', 'Median rain on a Rainy day of an Year'])



temp = df.groupby(by=['Year'])['precipitation'].sum().reset_index()

trace0 = go.Scatter(x=temp['Year'], y=temp['precipitation'])



temp = df.groupby(by=['Year'])['precipitation'].median().reset_index()

trace1 = go.Scatter(x=temp['Year'], y=temp['precipitation'])



fig.append_trace(trace0,1,1)

fig.append_trace(trace1,1,2)



fig.show()
fig = py.subplots.make_subplots(1,2, subplot_titles=['Total Rain', 'Median rain on a Rainy day'])



temp = df.groupby(by=['Month'])['precipitation'].sum().reset_index()

trace0 = go.Scatter(name= 'Total Rain', x=temp['Month'], y=temp['precipitation'])



temp = df.groupby(by=['Month'])['precipitation'].median().reset_index()

trace1 = go.Scatter(name='Median Rain', x=temp['Month'], y=temp['precipitation'])



fig.append_trace(trace0,1,1)

fig.append_trace(trace1,1,2)



fig.show()
df1 = df.groupby(by='Year')['precipitation'].sum().reset_index()



df1['change'] = 0

for i in range(1,df1.shape[0]):

    df1.loc[i,'change'] = (df1.loc[i,'precipitation']-df1.loc[i-1,'precipitation'])/df1.loc[i-1,'precipitation']



px.bar(df1, 'Year', 'change', color='change', title = 'Change in rainfall over the years :',

      color_continuous_scale=px.colors.sequential.Cividis)