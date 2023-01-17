import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import plotly as py

from plotly.offline import iplot, plot, init_notebook_mode, download_plotlyjs

import cufflinks as cf

import plotly.graph_objs as go

init_notebook_mode(connected=True)

import plotly.offline as offline

# py.tools.set_credentials_file(username='minhtc2', api_key='ByJVzjrlWsJX5h9SjZfh')



import statsmodels.api as sm

import ipywidgets as widgets

df = pd.read_csv("../input/market_cap.csv", parse_dates=["Date"])

df.head()
#• Fill empty data by using backward or forward method or values
# Using forward method to fill NaN value in co2 column

df_forw = df.fillna(method = "ffill")



#Using Backward method to fill NaN value in co2 column

df_back = df.fillna(method = "bfill")



#Display result after using 2 method

df_forw.head()
df_back.set_index("Date", inplace = True)

df_back.head()
# Shift, roll window, upsampling or downsampling data in co2 column



#Shift data in co2 column

# co2["Shift"] = co2.Backward.shift()

df_shift = df_back.shift()

#Display the result after somme action

df_shift.head(10)
#Rolling window by number that is size of the moving window

df_roll = df_back.rolling(window = 5).mean()

df_roll.iloc[5:10, :]
#Rolling window by offset that is time period

df_roll_1 = df_back.rolling(window = "30D").mean()

df_roll_1.head(10)
#Upsampling data by "D"

df_up = df_back.asfreq("D", method = "bfill")



#Display resutl after upsampling data

df_up.head(10)
#Downsampling data by "Q"

df_down = df_back.resample("Q").mean()



#Display result after downsampling data

df_down.head(10)
a = widgets.Dropdown(

    options=list(df_back.columns),

    value='AAPL',

    description='Kind of market:',

    disabled=False,

)

a
df_up["Average"] = df_up.mean(axis = 1)
# Decomposition data 

descompos = sm.tsa.seasonal_decompose(df_up[["Average"]])

df_obs = descompos.observed

df_tre = descompos.trend

df_sea = descompos.seasonal

df_res = descompos.resid

trace1 = go.Scatter(

    x = df_obs.index,

    y = df_obs.Average,

    name = "Observed"

)



# using graphic_object in plotly to creat trace for Trend

trace2 = go.Scatter(

    x = df_tre.index,

    y = df_tre.Average,

    name = "Trend"

)



# using graphic_object in plotly to creat trace for Seasonal

trace3 = go.Scatter(

    x = df_sea.index,

    y = df_sea.Average,

    name = "Seasonal"

)



# using graphic_object in plotly to creat trace for Residiual

trace4 = go.Scatter(

    x = df_res.index,

    y = df_res.Average,

    name = "Residiual"

)



# creat data from 4 trace

data = [trace1, trace2, trace3, trace4]

#Figure for graph

fig = py.tools.make_subplots(rows=4, cols=1, subplot_titles=('Observed', 'Trend',

                                                          'Seasonal', 'Residiual'))

# append trace into fig

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 2, 1)

fig.append_trace(trace3, 3, 1)

fig.append_trace(trace4, 4, 1)



fig['layout'].update( title='Descompose with TimeSeri')

offline.iplot(fig)
df_up.columns
data1 = []

for i in df_up.columns:

    trace = go.Scatter(

        x = df_up.index,

        y = df_up[i],

        name = i

    )

    data1.append(trace)

    

updatemenus  = list([

    dict(active=-1,

         buttons=list([   

            dict(label = 'AAPL',

                 method = 'update',

                 args = [{'visible': [True, False,False, False,False, False,False, False,False, False,False, False,True,]},

                         {'title': 'AAPL',

                          }]),

            dict(label = 'AMGN',

                 method = 'update',

                 args = [{'visible': [False, True, False,False,False, False,False, False,False, False,False, False,True,]},

                         {'title': 'AMGN',

                          }]),

             dict(label = 'AMZN',

                 method = 'update',

                 args = [{'visible': [False, False, True,False,False, False,False, False,False, False,False, False,True,]},

                         {'title': 'AMZN',

                          }]),

             dict(label = 'CPRT',

                 method = 'update',

                 args = [{'visible': [False, False, False,True,False, False,False, False,False, False,False, False,True,]},

                         {'title': 'CPRT',

                          }]),

             dict(label = 'EL',

                 method = 'update',

                 args = [{'visible': [False, False, False,False,True, False,False, False,False, False,False, False,True,]},

                         {'title': 'EL',

                          }]),

             dict(label = 'GS',

                 method = 'update',

                 args = [{'visible': [False, False, False,False,False, True,False, False,False, False,False, False,True,]},

                         {'title': 'GS',

                          }]),

             dict(label = 'All',

                 method = 'update',

                 args = [{'visible': [True, True, True,True,True, True,True, True, True, True,True, True,True,]},

                         {'title': 'ALLLL',

                          }]),



        ]),

    )

    

])

layout = go.Layout(

    title = "Time Seri %s and Average"%(a.value),

    xaxis = dict(title = "Time",

                rangeslider=dict(

                visible = True

                ),

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

                dict(count=1,

                    label='YTD',

                    step='year',

                    stepmode='todate'),

                dict(count=1,

                    label='1y',

                    step='year',

                    stepmode='backward'),

                dict(step='all')

            ])

        ),

                ),

    yaxis = dict(title = "Value"),

     updatemenus = updatemenus

    

)



fig = go.Figure(data = data1, layout = layout)

offline.iplot(fig)
from statsmodels.graphics import tsaplots
import matplotlib as mpl

with mpl.rc_context():

    mpl.rc("figure", figsize=(20,10))

    tsaplots.plot_acf(df_up.Average, lags = 25)
df_temp = df_up.corr(method = "spearman")

trace = go.Heatmap(

    x = df_temp.columns,

    y = df_temp.columns,

    z = df_temp

)

layout = go.Layout(

    title = "Correlation with spearman by heatmap",

)

data = [trace]

fig  = go.Figure(data = data, layout = layout)

offline.iplot(fig)
trace1 = go.Violin(

    y = df_up.Average,

    x0 = "Value", 

    box = dict(visible = True),

    line = dict(color = "black"),

    meanline = dict(visible = True),

    opacity = 0.6,

    fillcolor = '#8dd3c7'

)

layout  = go.Layout(

    title = "Violin Plot",

    yaxis = dict(title = "Value")

)

data = [trace1]

fig = go.Figure(data = data, layout = layout)

offline.iplot(fig)
data = []

for i in df_up.columns:

    trace = go.Violin(

        x0 = i,

        y = df_up[i],

        name = i,

        box = dict(visible = True),

        meanline = dict(visible = True),

        opacity = 0.6,

    )

    data.append(trace)

    

layout = go.Layout(

    title = "Violin Plot"

)

fig = go.Figure(data = data, layout = layout)

offline.iplot(fig, validate = False)