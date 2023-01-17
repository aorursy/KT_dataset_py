import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

np.set_printoptions(threshold=np.inf)



selected_country = 'Singapore'

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', header = 0)

df = df[df['Country/Region']==selected_country]

df = df.groupby('ObservationDate').sum()

print(df)
df['daily_confirmed'] = df['Confirmed'].diff()

df['daily_deaths'] = df['Deaths'].diff()

df['daily_recovery'] = df['Recovered'].diff()

df['daily_confirmed'].plot()

df['daily_recovery'].plot()
print(df)
from plotly.offline import iplot

import plotly.graph_objs as go



daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily confirmed')

daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily deaths')



layout_object = go.Layout(title='Singapore daily cases 20M51843',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))

fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object],layout=layout_object)

iplot(fig)

fig.write_html('Singapore_daily_cases_20M51843.html')
df1 = df#[['daily_confirmed']]

df1 = df1.fillna(0.)

styled_object = df1.style.background_gradient(cmap='gnuplot2').highlight_max('daily_confirmed').set_caption('Daily Summaries')

display(styled_object)

f = open('table_20M51843.html','w')

f.write(styled_object.render())
data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

data.head()

data.index=data['ObservationDate']

data = data.drop(['SNo','ObservationDate'],axis=1)

data.head()

data_Singapore = data[data['Country/Region']=='Singapore']

data_Singapore.tail()

latest = data[data.index=='06/18/2020']

latest = latest.groupby('Country/Region').sum()

latest = latest.sort_values(by='Confirmed',ascending=False).reset_index() 

print('Ranking of Singapore: ', latest[latest['Country/Region']=='Singapore'].index.values[0]+1)