#import libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

#for advanced ploting

import seaborn as sns

#for interactive visualizations

import plotly.express as px

import plotly.graph_objs as go
#read data 

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

df_germany = (df[df['Country/Region']=='Germany'])

df = df.groupby('ObservationDate').sum()

df.tail(10)
df_germany = df_germany.groupby('ObservationDate').sum()

df_germany.tail(10)
df_germany['daily_confirmed'] = df_germany['Confirmed'].diff()

df_germany['daily_deaths'] = df_germany['Deaths'].diff()

df_germany['daily_recovered'] = df_germany['Recovered'].diff()

df_germany['daily_confirmed'].plot(color=['b'], label='daily confirmed')

df_germany['daily_recovered'].plot(color=['g'], label='daily recovered')

df_germany['daily_deaths'].plot(color='r', label='daily deaths')

plt.ylabel('Number of people')

plt.xticks(rotation=45)

plt.title('Coronavirus cases in Germany 19M58430')

plt.legend()
from plotly.offline import iplot

import plotly.graph_objs as go

df_germany
daily_confirmed_object = go.Scatter(x=df.index, y=df_germany['daily_confirmed'].values, name='Daily Confirmed',mode='markers', marker_line_width=1, marker_size=8)

daily_deaths_object = go.Scatter(x=df.index, y=df_germany['daily_deaths'].values, name='Daily Deaths', mode='markers',marker_line_width=1, marker_size=9)

daily_recovered_object = go.Scatter(x=df.index, y=df_germany['daily_recovered'].values, name='Daily Recovered', mode='markers',marker_line_width=1, marker_size=8)

layout_object = go.Layout(title='Germany Daily Cases 19M58430', xaxis=dict(title='Date'), yaxis=dict(title='Number of people'))

fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovered_object], layout=layout_object)

fig.update_xaxes(tickangle=60,tickfont=dict(family='Arial', color='black',size=12))

iplot(fig)
df_germany
df_germany['Active'] = df_germany['Confirmed']- df_germany['Deaths'] - df_germany['Recovered']

df_germany['NewConfirmed'] = df_germany['Confirmed'].shift(-1) - df_germany['Confirmed']

df_germany['NewRecovered'] = df_germany['Recovered'] - df_germany['Recovered'].shift(periods=1)

df_germany['NewDeaths'] = df_germany['Deaths'] - df_germany['Deaths'].shift(periods=1)


print(df_germany)
active = df_germany['Active'].plot()

plt.xticks(rotation=45)

plt.ylabel('Number of people')

plt.title('Active cases in Germany 19M58430')
new = df_germany['NewConfirmed'].plot()

plt.xticks(rotation=45)

plt.ylabel('Number of people')

plt.title('New Confirmed Cases 19M58430')

newrecov = df_germany['NewRecovered'].plot()

plt.xticks(rotation=45)

plt.ylabel('Number of people')

plt.title('New Recovered Cases 19M58430')
newdeaths = df_germany['NewDeaths'].plot()

plt.xticks(rotation=45)

plt.ylabel('Number of people')

plt.title('New Deaths 19M58430')



df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

df.index=df['ObservationDate']

df = df.drop(['SNo','ObservationDate'],axis=1)

df_Germany = df[df['Country/Region']=='Germany']



latest = df[df.index=='06/16/2020']

latest = latest.groupby('Country/Region').sum()

latest = latest.sort_values(by='Confirmed',ascending=False).reset_index()



print('Germany Rank: ', latest[latest['Country/Region']=='Germany'].index.values[0]+1)