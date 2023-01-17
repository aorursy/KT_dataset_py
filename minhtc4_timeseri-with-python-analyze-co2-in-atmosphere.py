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



#https://www.kaggle.com/thebrownviking20/everything-you-can-do-with-a-time-series
#Reading dataset co2 by pandas

co2 = pd.read_csv("../input/co2.csv", index_col="datestamp", parse_dates= ["datestamp"])

co2.head()
#• Fill empty data by using backward or forward method or values
# Using forward method to fill NaN value in co2 column

co2["Forward"] = co2.co2.fillna(method = "ffill")



#Using Backward method to fill NaN value in co2 column

co2["Backward"] = co2.co2.fillna(method = "bfill")



#Display result after using 2 method

co2.head(10)

# Shift, roll window, upsampling or downsampling data in co2 column



#Shift data in co2 column

co2["Shift"] = co2.Backward.shift()



#Rolling window by number that is size of the moving window

co2["Rolling_int"] = co2.Backward.rolling(window = 5).mean()



#Rolling window by offset that is time period

co2["Rolling_offset"] = co2.Backward.rolling(window = "30D").mean()



#Display the result after somme action

co2.head(10)

#Upsampling data by "D"

co2_up = co2.asfreq("D", method = "bfill")



#Display resutl after upsampling data

co2_up.head(10)

#Downsampling data by "Q"

co2_down = co2.resample("Q").mean()



#Display result after downsampling data

co2_down.head(10)
# Decomposition data 

descompos = sm.tsa.seasonal_decompose(co2.Backward)
df_obs = descompos.observed.to_frame()

df_tre = descompos.trend.to_frame()

df_sea = descompos.seasonal.to_frame()

df_res = descompos.resid.to_frame()
init_notebook_mode(connected=True)

# using graphic_object in plotly to creat trace for Observed

trace1 = go.Scatter(

    x = df_obs.index,

    y = df_obs.Backward,

    name = "Observed"

)



# using graphic_object in plotly to creat trace for Trend

trace2 = go.Scatter(

    x = df_tre.index,

    y = df_tre.Backward,

    name = "Trend"

)



# using graphic_object in plotly to creat trace for Seasonal

trace3 = go.Scatter(

    x = df_sea.index,

    y = df_sea.Backward,

    name = "Seasonal"

)



# using graphic_object in plotly to creat trace for Residiual

trace4 = go.Scatter(

    x = df_res.index,

    y = df_res.Backward,

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
# Autocorrelation
from statsmodels.graphics import tsaplots

fig = tsaplots.plot_acf(co2.Backward, lags = 24)
# Visualization
# Creat trace for graph

trace1 = go.Scatter(

    x = co2.index,

    y = co2.Backward,

)

#creat layout

layout = go.Layout(

    xaxis = dict(title= "Year"), 

    yaxis = dict(title= "Co2 in Atmosphere"),

    title = "CO2 in Atmosphere from 1958 to 2001"

)

data = [trace1]

fig = go.Figure(data=data, layout=layout)

offline.iplot(fig)
# Creat normalize timeseri at co2.Backward by co2.Backward/first_value * 100

first_value = co2["Backward"][0]

co2["normalize"] = co2["Backward"].div(first_value).mul(100)



# co2["normalize"]

trace1 = go.Scatter(

    x = co2.index,

    y = co2["normalize"]

)

data = [trace1]

layout =  go.Layout(

    title = "rate of increase co2 over the years".upper(),

    xaxis = dict(title = "Year"),

    yaxis = dict(title = "Rate of CO2")

)

fig = go.Figure(data = data, layout = layout)

offline.iplot(fig)
# Creat months columns for each time

a = co2["Backward"]

a = a.reset_index()

# list_month = [i for i in ]

a["datestamp"].asfreq("W", method = "bfill")

list_month =  [str(i)[5:7] for i in a.datestamp ]

a["month"] = list_month

#caculate mean of co2 of each months

a = a.groupby("month")[["Backward"]].mean()



#Start Graphing

trace1 = go.Scatter(

    x = a.index,

    y = a.Backward,

)

layout = go.Layout(

    title  = "Average CO2 over the months".upper(),

    xaxis = dict(title = "Month"),

    yaxis = dict(title = "Average CO2")

)

data = [trace1]

fig = go.Figure(data = data, layout = layout)

offline.iplot(fig)
first_vl = a["Backward"][0]

a["normalize"] = a["Backward"].div(first_vl).mul(100)



trace1 =  go.Scatter(

    y = a["normalize"],

    x = a.index

)

data  = [trace1]

layout = go.Layout(

    title = "Rate of Co2 over months".upper(),

    xaxis = dict(title  = "Month"),

    yaxis = dict(title  = "Rate of Co2")

)

fig = go.Figure(data = data, layout = layout)

offline.iplot(fig)