import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

np.set_printoptions(threshold=np.inf)
data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

#let's set ObservationDate as the index and drop the unneeded columns

data.index=data['ObservationDate']

data = data.drop(['SNo','ObservationDate'],axis=1)

data.head()

data_Russia = data[data['Country/Region']=='Russia']

data_Russia = data_Russia.groupby('ObservationDate').sum()

data_Russia
latest=data[data.index=='06/02/2020']

latest=latest.groupby('Country/Region').sum()

latest=latest.sort_values(by='Confirmed',ascending=False).reset_index() 

print('Ranking of Russia: ',latest[latest['Country/Region']=='Russia'].index.values[0]+1)
data_Russia['Daily confirmed'] = data_Russia['Confirmed'].diff()

data_Russia['Daily deaths'] = data_Russia['Deaths'].diff()

data_Russia['Daily recovery'] = data_Russia['Recovered'].diff()

data_Russia
from plotly.offline import iplot

import plotly.graph_objs as go



daily_confirmed_object = go.Scatter(x=data_Russia.index,y=data_Russia['Daily confirmed'].values,name='Daily confirmed')

daily_deaths_object = go.Scatter(x=data_Russia.index,y=data_Russia['Daily deaths'].values,name='Daily deaths')



layout_object = go.Layout(title='Russia Daily Cases 20M53055',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))

fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object],layout=layout_object)

iplot(fig)

fig.write_html('Russia_Daily_Cases_20M53055.html')
df1=data_Russia

df1=df1.fillna(0.)

styled_object=df1.style.background_gradient(cmap='gist_ncar').highlight_max('Daily confirmed').set_caption('Daily Summaries')

display(styled_object)

f=open('table_20M53055.html','w')

f.write(styled_object.render())