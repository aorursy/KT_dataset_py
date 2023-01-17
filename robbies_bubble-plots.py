#Add data and make bubbles



import pandas as pd

import numpy as np

import plotly.plotly as py

import plotly.graph_objs as go
# Import data from csv

df = pd.read_csv('../input/UNdata_Export_20170112_044235144.csv')

df.head()
trace1 = go.Scatter(

                    x=df['2002'], y=df['2007'], # Data

                    mode='lines', name='First 5' # Additional options

                   )

trace2 = go.Scatter(x=df['2007'], y=df['2012'], mode='lines', name='Next 5' )

trace3 = go.Scatter(x=df['2002'], y=df['2012'], mode='lines', name='All 10')



layout = go.Layout(title='Simple Plot from csv data',

                   plot_bgcolor='rgb(230, 230,230)')



fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)



# Plot data in the notebook

py.iplot(fig, filename='simple-plot-from-csv')