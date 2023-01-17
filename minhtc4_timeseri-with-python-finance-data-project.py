import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import plotly as py

from plotly.offline import iplot, plot, init_notebook_mode, download_plotlyjs

import cufflinks as cf

import plotly.graph_objs as go

init_notebook_mode(connected=True)

import plotly.offline as offline



import statsmodels.api as sm
google = pd.read_csv("../input/google.csv", index_col="Date", parse_dates= ["Date"])

google.head()
#• Fill empty data by using backward or forward method or values
# Using forward method to fill NaN value in co2 column

google["Froward"] = google["Close"].fillna( method = "ffill")



# Using backward method to fill NaN value in co2 column

google["Backward"] = google["Close"].fillna(method = "bfill")



google.head()
# Shift, roll window, upsampling or downsampling data in co2 column



#Shift data in co2 column

google["Shift"] = google.Backward.shift()



#Rolling window by number that is size of the moving window

google["Rolling_int"] = google.Backward.rolling(window = 5).mean()



#Rolling window by offset that is time period

google["Rolling_offset"] = google.Backward.rolling(window = "30D").mean()



#Display the result after somme action

google.head(10)
#Upsampling data by "D"

google_up = google.asfreq("3D", method = "bfill")



#Display resutl after upsampling data

google_up.head(10)
#Downsampling data by "Q"

google_down = google.resample("Q").mean()



#Display result after downsampling data

google_down.head(10)
descompos = sm.tsa.seasonal_decompose(google.Backward)



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

fig = tsaplots.plot_acf(google.Backward, lags = 24)
# Visualization