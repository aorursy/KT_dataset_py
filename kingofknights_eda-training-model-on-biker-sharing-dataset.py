# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly
import plotly.offline as plt
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from datetime import datetime

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
plotly.offline.init_notebook_mode(connected=True)
import os
print(os.listdir("../input/bike-sharing-dataset"))

# Any results you write to the current directory are saved as output.
days = pd.read_csv('../input/bike-sharing-dataset/day.csv', index_col='instant')
days.head()
days.info()
days.dteday = pd.to_datetime(days.dteday, format='%Y-%m-%d')

for column in ['season', 'mnth', 'weekday', 'workingday', 'weathersit', 'holiday', 'yr']:
    days[column] = days[column].astype('category')
    
days.describe()
data = days[['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']]
data = data.corr()
trace_1 = go.Heatmap(z=data.as_matrix(), x=data.columns, y=data.index)

trace_2 = go.Surface(z=data.as_matrix())

layout = go.Layout(title='Surface map <br> 3D version of HeatMap', autosize=False, width=500, height=500,
    margin=dict(l=65, r=50, b=65, t=90))
trace_2 = go.Figure(data=[trace_2], layout=layout)
plt.iplot([trace_1])
plt.iplot(trace_2)
def plotVariation(by):
    data = days.groupby(by, as_index=False).sum()
    trace_1 = go.Bar(x=data[by], y=data.casual, name='casual')
    trace_2 = go.Bar(x=data[by], y=data.registered, name='registered')
    layout = go.Layout(barmode='group', title='Variation in biker in different ' + by + '<br> casual vs registered')
    figure = go.Figure(data=[trace_1, trace_2], layout=layout)
    plt.iplot(figure)
plotVariation('yr')
data_0 = days[days.yr == 0]
data_1 = days[days.yr == 1]

data_0 = data_0.groupby('workingday', as_index=False).count()
data_1 = data_1.groupby('workingday', as_index=False).count()

print("diffrence in number of workingday in both year is ", data_1.yr[1] - data_0.yr[1])
fig = tools.make_subplots(rows=2, cols=4)
fig['layout'].update( title='side by side comparasion of casual biker in each season of both year')
for seasons in days.season.unique():
    for year in days.yr.unique():
        data = days[days.yr == year]
        data = data[data.season == seasons]
        trace = go.Box(y=data.casual, name='season #' + str(seasons) + ' of year ' + str(year), boxmean='sd', boxpoints='all')
        fig.append_trace(trace, year + 1, seasons)

fig_1 = tools.make_subplots(rows=2, cols=4)
fig_1['layout'].update( title='side by side comparasion of registered biker in each season of both year')
for seasons in days.season.unique():
    for year in days.yr.unique():
        data = days[days.yr == year]
        data = data[data.season == seasons]
        trace = go.Box(y=data.registered, name='season #' + str(seasons) + ' of year ' + str(year), boxmean='sd', boxpoints='all')
        fig_1.append_trace(trace, year + 1, seasons)
plt.iplot(fig)
plt.iplot(fig_1)
season = [1, 2, 3, 4]
year_0_casual = [140, 730.5, 800, 456]
year_1_casual = [269, 1064, 1197, 753]
year_0_reg = [1454, 3203, 3594.5, 3240]
year_1_reg = [3162, 4948.5, 5670.5, 5080]
trace0 = go.Scatter( x = season, y = year_0_casual, name = 'year_0_casual', mode='lines+markers' ,hoverinfo='name', line=dict(shape='vhv'))
trace1 = go.Scatter( x = season, y = year_1_casual, name = 'year_1_casual', mode='lines+markers', hoverinfo='name', line=dict(shape='vhv'))
trace2 = go.Scatter( x = season, y = year_0_reg, name = 'year_0_reg', mode='lines+markers', hoverinfo='name', line=dict(shape='vhv'))
trace3 = go.Scatter( x = season, y = year_1_reg, name = 'year_1_reg', mode='lines+markers', hoverinfo='name', line=dict(shape='vhv'))

layout = dict(title = 'variation in bikers', xaxis = dict(title = 'Seasons'), yaxis = dict(title = 'Median of biker visited'))
fig = dict(data=[trace0, trace1, trace2, trace3], layout=layout)
plt.iplot(fig)
def voilinplot(bikertype):
    for year in [0, 1]:
        data_header = bikertype
        group_header = 'mnth'
        data = days[days.yr == year]
        fig = ff.create_violin(data, data_header=data_header, group_header=group_header, height=500, width=1000)
        plt.iplot(fig)
voilinplot('casual')
voilinplot('registered')
days['weeks'] = days.dteday.dt.strftime('%U')
data = days[days.yr == 0]
data = data.groupby('weeks', as_index=False).sum()
trace1 = go.Bar( x=data.weeks, y=data.casual, name='casual- year 0', text=data.casual, textposition = 'auto',)
trace2 = go.Bar( x=data.weeks, y=data.registered, name='registered-year 0', text=data.registered, textposition = 'auto')

data = days[days.yr == 1]
data = data.groupby('weeks', as_index=False).sum()
trace3 = go.Bar( x=data.weeks, y=data.casual, name='casual-year 1', text=data.casual, textposition = 'auto',)
trace4 = go.Bar( x=data.weeks, y=data.registered, name='registered-year 1', text=data.registered, textposition = 'auto')

layout = go.Layout(title='Variation in bikers through weeks in both year' ,xaxis=dict(tickangle=-45), barmode='group', bargap=0.15, bargroupgap=0.1, yaxis=dict(title='Number of biker visited'))
fig = go.Figure(data=[trace1, trace3, trace2, trace4], layout=layout)
plt.iplot(fig)
trace_1 = go.Scatter(
    x=days.dteday,
    y=days.casual,
    name = "casual",
    line = dict(color = '#6C3483'),
    opacity = 0.8)

trace_2 = go.Scatter(
    x=days.dteday,
    y=days.registered,
    name = "registered",
    line = dict(color = '#1E8449'),
    opacity = 0.8)

trace_3 = go.Scatter(
    x=days.dteday,
    y=days.cnt,
    name = "total",
    line = dict(color = '#EB984E'),
    opacity = 0.8)

title = 'casual bikers vs registered bikers'
layout = dict(
    title=title,
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(),
        type='date'
    )
)

data = [trace_1,trace_2, trace_3]
fig = dict(data=data, layout=layout)
plt.iplot(fig)
fig = tools.make_subplots(rows=2, cols=4)
fig['layout'].update( title='Season wise <br> Temprature')
for seasons in days.season.unique():
    for year in days.yr.unique():
        data = days[days.yr == year]
        data = data[data.season == seasons]
        trace = go.Box(y=data.temp, name='season #' + str(seasons) + ' of year ' + str(year), boxmean='sd', boxpoints='all')
        fig.append_trace(trace, year + 1, seasons)
plt.iplot(fig)
def tempraturevoilin(group_header):
    for year in [0, 1]:
        data_header = 'temp'
        data = days[days.yr == year]
        fig = ff.create_violin(data, data_header=data_header, group_header=group_header, height=500, width=1000)
        plt.iplot(fig)
tempraturevoilin('mnth')
data = days[days.yr == 0]
data = data.groupby('weeks', as_index=False).mean()
trace1 = go.Bar( x=data.weeks, y=data.temp, name='temp- year 0', text=data.temp, textposition = 'auto',)

data = days[days.yr == 1]
data = data.groupby('weeks', as_index=False).mean()
trace4 = go.Bar( x=data.weeks, y=data.temp, name='temp-year 1', text=data.temp, textposition = 'auto')

layout = go.Layout(title='Variation in temprature through weeks in both year' ,xaxis=dict(tickangle=-45), barmode='group', bargap=0.15, bargroupgap=0.1, yaxis=dict(title='mean temprature'))
fig = go.Figure(data=[trace1, trace4], layout=layout)
plt.iplot(fig)
trace_1 = go.Scatter(
    x=days.dteday,
    y=days.temp,
    name = "temp",
    line = dict(color = '#F39C12'),
    opacity = 0.8)

trace_2 = go.Scatter(
    x=days.dteday,
    y=days.atemp,
    name = "atemp",
    line = dict(color = '#3498DB'),
    opacity = 0.8)
title = 'Temprature variation in both year'

data = [trace_1, trace_2]
layout = dict(
    title=title,
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(),
        type='date'
    )
)
fig = go.Figure(data=data, layout=layout)
plt.iplot(fig)
trace_1 = go.Scatter(
    x=days.dteday,
    y=days.hum,
    name = "humidity",
    line = dict(color = '#F1948A'),
    opacity = 0.8)

trace_2 = go.Scatter(
    x=days.dteday,
    y=days.windspeed,
    name = "windspeed",
    line = dict(color = '#3498DB'),
    opacity = 0.8)
title = 'humidity and windspeed variation in both year'

data = [trace_1, trace_2]
layout = dict(
    title=title,
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(),
        type='date'
    )
)
fig = go.Figure(data=data, layout=layout)
plt.iplot(fig)
feature = ['season', 'mnth', 'weekday', 'workingday', 'weathersit', 'holiday', 'yr']
target = ['temp']
x_train, x_test, y_train, y_test = train_test_split(days[feature], days[target], random_state=67, test_size=0.2)
def regressorOutCome(depth):
    regressor = DecisionTreeRegressor(max_depth=depth)
    regressor.fit(x_train, y_train)
    score = regressor.score(x_test, y_test)
    print ('regression score on maxdepth {depth} is {score}'.format(depth=depth, score=score))
    y_pred = regressor.predict(x_test)
    return y_pred
y_pred = regressorOutCome(6)
x = np.arange(0, len(y_pred) + 1).tolist()
trace_1 = go.Scatter(
    x=x,
    y=y_test.temp,
    name = "testing",
    line = dict(color = '#FF0000'),
    opacity = 0.8)

trace_2 = go.Scatter(
    x=x,
    y=y_pred.tolist(),
    name = "prediction",
    line = dict(color = '#123456'),
    opacity = 0.8)
title = 'y_pred and y_test <br> Temprature'

data = [trace_1, trace_2]
layout = dict(
    title=title
)
fig = go.Figure(data=data, layout=layout)
plt.iplot(fig)
