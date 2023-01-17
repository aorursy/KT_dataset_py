import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)

selected_country='Tunisia'
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', header=0)

del df['SNo']
df = df[df['Country/Region']==selected_country]
df = df.groupby('ObservationDate').sum()
print(df)
df1 = df.diff()
print(df1.plot())
from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df1.index, y=df1['Confirmed'].values, name='Confirmed')
daily_deaths_object = go.Scatter(x=df1.index, y=df1['Deaths'].values, name='Deaths')
daily_recovered_object = go.Scatter(x=df1.index, y=df1['Recovered'].values, name='Recovered')
layout_object = go.Layout(title='Tunisia Daily Cases',xaxis=dict(title='Date'),yaxis=dict(title='Number of People'))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovered_object],layout=layout_object)
iplot(fig)

df1= df1.fillna(0.)
styled_object = df1.style.background_gradient(cmap='gist_ncar').highlight_max('Confirmed').set_caption('Daily Summaries')
display(styled_object)
f = open('table_19M58624.html','w')
f.write(styled_object.render())
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', header=0)

latest = df[df['ObservationDate']=='06/12/2020']
latest = latest.groupby('Country/Region').sum()
latest = latest.sort_values(by='Confirmed',ascending=False).reset_index()
print('Global Ranking of Tunisia: ', latest[latest['Country/Region']==selected_country].index.values[0]+1)