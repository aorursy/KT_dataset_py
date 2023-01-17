import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from IPython.core import display as ICD
init_notebook_mode(connected=True)
museum = pd.read_csv('../input/museum-visitors.csv')
museum = museum.sort_values(by='Month')

trace1 = go.Scatter(
                    x = museum.Month,
                    y = museum['America Tropical Interpretive Center'],
                    mode = "lines+markers",
                    name = "America Tropical Interpretive Center",
                    marker = dict(color = 'rgba(0, 0, 255, 0.8)'),)
trace2 = go.Scatter(
                    x = museum.Month,
                    y = museum['Avila Adobe'],
                    mode = "lines+markers",
                    name = "Avila Adobe",
                    marker = dict(color = 'rgba(0, 128, 0, 0.8)'),)
trace3 = go.Scatter(
                    x = museum.Month,
                    y = museum['Chinese American Museum'],
                    mode = "lines+markers",
                    name = "Chinese American Museum",
                    marker = dict(color = 'rgba(0, 128, 128, 0.8)'),)
trace4 = go.Scatter(
                    x = museum.Month,
                    y = museum['Firehouse Museum'],
                    mode = "lines+markers",
                    name = "Firehouse Museum",
                    marker = dict(color = 'rgba(255, 0, 0, 0.8)'),)
trace5 = go.Scatter(
                    x = museum.Month,
                    y = museum['IAMLA'],
                    mode = "lines+markers",
                    name = "IAMLA",
                    marker = dict(color = 'rgba(255, 0, 255, 0.8)'),)
trace6 = go.Scatter(
                    x = museum.Month,
                    y = museum['Pico House '],
                    mode = "lines+markers",
                    name = "Pico House",
                    marker = dict(color = 'rgba(255, 120, 0, 0.8)'),)
trace7 = go.Scatter(
                    x = museum.Month,
                    y = museum['Museum of Social Justice'],
                    mode = "lines+markers",
                    name = "Museum of Social Justice",
                    marker = dict(color = 'rgba(90, 90, 90, 0.8)'),)
trace8 = go.Scatter(
                    x = museum.Month,
                    y = museum['Biscailuz Gallery'],
                    mode = "lines+markers",
                    name = "Biscailuz Gallery",
                    marker = dict(color = 'rgba(25, 25, 25, 0.8)'),)

data = [trace1, trace2, trace3, trace4, trace5, trace6,trace7,trace8]
layout = dict(title = 'Monthly Count of Museum Visitors in Los Angeles',
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),yaxis= dict(title= 'Number of Museum Visitors',ticklen= 5,zeroline= False),legend=dict(orientation= "h",x=0, y= 1.13))
fig = dict(data = data, layout = layout)
iplot(fig)
