import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.figure_factory as ff

import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots
data = pd.read_csv('../input/covid19turkeydailydetailsdataset/covid19-Turkey.csv')
data.head()
data.info()
data.date = pd.to_datetime(data.date)
fig = px.line(data, x='date', y='totalCases', title='Progression of Number of Cases')

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=data.date, y=data.totalCases,

                    mode='lines',

                    name='Total Cases'))

fig.add_trace(go.Scatter(x=data.date, y=data.totalDeaths,

                    mode='lines',

                    name='Total Deaths'))

fig.add_trace(go.Scatter(x=data.date, y=data.totalRecovered,

                    mode='lines',

                    name='Total Recovered'))

fig.update_layout(title='Total cases along with recovery and fatality trends')

fig.show()
fig = go.Figure(data=[go.Pie(labels=['Recovered','Deaths'], values=[data.iloc[-1].totalRecovered,data.iloc[-1].totalDeaths],textinfo='label+percent')])

fig.update_layout(title='Comparison of Recoveries and Deaths')

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=data.date, y=data.totalCases,

                    mode='lines',

                    name='Total Cases'))

fig.add_trace(go.Scatter(x=data.date, y=data.totalRecovered,

                    mode='lines',

                    name='Total Recovered'))

fig.add_trace(go.Scatter(x=data.date, y=data.totalTests,

                    mode='lines',

                    name='Total Tested'))

fig.update_layout(title='Has Testing contributed to high recovery rate?')

fig.show()
fig = go.Figure()

# fig.add_trace(go.Scatter(x=data.date, y=data.totalCases,

#                     mode='lines',

#                     name='Total Cases'))

fig.add_trace(go.Scatter(x=data.date, y=data.totalDeaths,

                    mode='lines',

                    name='Total Deaths'))

# fig.add_trace(go.Scatter(x=data.date, y=data.totalRecovered,

#                     mode='lines',

#                     name='Total Recovered'))

fig.add_trace(go.Scatter(x=data.date, y=data.totalIntubated,

                    mode='lines',

                    name='Total Intubated'))

fig.add_trace(go.Scatter(x=data.date, y=data.totalIntensiveCare,

                    mode='lines',

                    name='Total IntensiveCare'))

fig.update_layout(title='Intubation and intensive care trends')

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=data.date, y=data.dailyCases,

                    mode='lines',

                    name='Daily Cases'))

fig.add_trace(go.Scatter(x=data.date, y=data.dailyDeaths,

                    mode='lines',

                    name='Daily Deaths'))

fig.add_trace(go.Scatter(x=data.date, y=data.dailyRecovered,

                    mode='lines',

                    name='Daily Recovered'))

fig.update_layout(title='Daily cases along with recovery and fatality trends')

fig.show()
fig = go.Figure(data=[

    go.Bar(name='Daily Test', x=data.date, y=data.dailyTests),

    go.Bar(name='Daily Cases', x=data.date, y=data.dailyCases)

])

fig.update_layout(title='Test vs Cases Counts')

fig.update_layout(barmode='stack')

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=data.date, y=data.dailyCases,

                    mode='lines',

                    name='Daily Cases'))

fig.add_trace(go.Scatter(x=data.date, y=data.dailyDeaths,

                    mode='lines',

                    name='Daily Deaths'))

fig.add_trace(go.Scatter(x=data.date, y=data.dailyRecovered,

                    mode='lines',

                    name='Daily Recovered'))

fig.add_trace(go.Scatter(x=data.date, y=data.dailyTests,

                    mode='lines',

                    name='Daily Tests'))

fig.update_layout(title='At what scale has Turkey carried out testing?')

fig.show()