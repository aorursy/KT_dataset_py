import os

import datetime

import numpy as np

import pandas as pd

import plotly

import plotly.offline as py
py.init_notebook_mode()
data_directory = '../input'
participant = 'a'
rolling_mean_window = 30
timeframe = ('2016-04-01', '2017-02-01')
def remove_outliers(series):

    iqr = (series.quantile(0.25) * 1.5, series.quantile(0.75) * 1.5)

    outliers = (series < iqr[0]) | (series > iqr[1])

    return series[~outliers]
def normalize(series):

    min = series.min()

    max = series.max()

    return ((series - min) / (max - min) - 0.5) * 2
data = pd.DataFrame()
lifeslice = pd.read_csv(data_directory + '/' + participant + '.lifeslice.csv', parse_dates=[['date', 'time']], index_col=['date_time']).dropna()



lifeslice.head()
series = lifeslice['emotions.valence']

series = remove_outliers(series)

series = normalize(series)

data = data.merge(series.to_frame('lifeslice'), how='outer', left_index=True, right_index=True)



for dataset in ['imessage', 'facebook', 'dayone', '750words']:

    csv = data_directory + '/' + participant + '.' + dataset + '.csv'

    if (not os.path.exists(csv)):

        continue

    df = pd.read_csv(csv, parse_dates=[['date', 'time']], index_col=['date_time']).dropna()

    series = df['sentiment.comparative']

    series = remove_outliers(series)

    series = normalize(series)

    data = data.merge(series.to_frame(dataset), how='outer', left_index=True, right_index=True)
data.head()
start, end = (data.index.searchsorted(datetime.datetime.strptime(i, '%Y-%m-%d')) for i in timeframe)

data = data[start:end]
fig = plotly.tools.make_subplots(rows=len(data.columns), cols=1)



for index, column in enumerate(data.columns):

    trace = plotly.graph_objs.Histogram(

        name = column,

        x = data[column],

    )

    fig.append_trace(trace, index + 1, 1)



fig['layout'].update(height=len(data.columns) * 250)

plot_url = py.iplot(fig)
for column in data.columns:

    if column == 'lifeslice':

        data = data[data[column] != 1]

        continue

    data = data[data[column] != -1]
fig = plotly.tools.make_subplots(rows=len(data.columns), cols=1)



for index, column in enumerate(data.columns):

    trace = plotly.graph_objs.Histogram(

        name = column,

        x = data[column],

    )

    fig.append_trace(trace, index + 1, 1)



fig['layout'].update(height=len(data.columns) * 250)

plot_url = py.iplot(fig)
rule = '1d'

resampled = data.resample('1d').mean().fillna(data.mean()).rolling(rolling_mean_window, center=True).mean()

resampled.head()
resampled.dropna(inplace=True)

resampled.head()
lower = data.resample(rule).apply(lambda x: x.quantile(q=0.25)).fillna(data.mean()).rolling(rolling_mean_window, center=True).mean().dropna()

upper = data.resample(rule).apply(lambda x: x.quantile(q=0.75)).fillna(data.mean()).rolling(rolling_mean_window, center=True).mean().dropna()
colors = [

    '#50514F',

    '#F25F5C',

    '#FFE066',

    '#247BA0',

    '#70C1B3',

]



datasets = [[

    # Scatterplot

    plotly.graph_objs.Scatter(

        name = column,

        x = data.index,

        y = data[column],

        mode = 'markers',

        marker = {

            'size': 1,

            'color': colors[index],

        },

    ),

    # Moving average

    plotly.graph_objs.Scatter(

        name = column + ' ma',

        x = resampled.index,

        y = resampled[column],

        mode = 'lines',

        fill = 'tonexty',

        fillcolor = 'rgba(68, 68, 68, 0.3)',

        line = {

            'color': colors[index],

        },

    ),

    # Lower quartile

    plotly.graph_objs.Scatter(

        name = column,

        x = resampled.index,

        y = lower[column],

        line = dict(width = 0),

        showlegend = False,

        mode = 'lines',

    ),

    # Upper quartile

    plotly.graph_objs.Scatter(

        name = column,

        x = resampled.index,

        y = upper[column],

        fill='tonexty',

        fillcolor='rgba(68, 68, 68, 0.3)',

        marker=dict(color = '444'),

        line=dict(width = 0),

        showlegend = False,

        mode='lines',

    )

] for index, column in enumerate(data.columns)]
to_plot = [trace for dataset in datasets[0:2] for trace in dataset]

py.iplot(to_plot, filename='chronist-time-series')
fig = plotly.tools.make_subplots(rows=len(datasets), cols=1)



for index, dataset in enumerate(datasets):

    for trace in dataset:

        fig.append_trace(trace, index + 1, 1)



fig['layout'].update(title='Sentiment Comparisons', height=len(datasets) * 250)

plot_url = py.iplot(fig, filename='stacked-subplots')
resampled.corr()
trace = plotly.graph_objs.Scatter(

    name = 'comparison',

    x = resampled['lifeslice'],

    y = resampled['imessage'],

    mode = 'markers',

    marker = {

        'size': 5,

        'color': colors[0],

    },

)

py.iplot([trace])