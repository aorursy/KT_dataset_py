import numpy as np 

import pandas as pd 

import os

from datetime import date, datetime, timedelta

from IPython.core import display as ICD

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

init_notebook_mode(connected=True)

import sys

import warnings



base_dir = '../input/top-tracks-of-2017/'

fileName = 'featuresdf.csv'

filePath = os.path.join(base_dir,fileName)

top2017 = pd.read_csv(filePath) 



base_dir = '../input/top-spotify-tracks-of-2018/'

fileName = 'top2018.csv'

filePath = os.path.join(base_dir,fileName)

top2018 = pd.read_csv(filePath) 
print('Top 100 Spotify Tracks in 2017: ')

pd.set_option('display.max_rows', 100)

top2017[['artists','name']]
print('Top 100 Spotify Tracks in 2018: ')

pd.set_option('display.max_rows', 100)

top2018[['artists','name']]
artistCounts = top2017['artists'].value_counts()

artistCountsDf = pd.DataFrame(artistCounts)

trace1 = go.Bar(

                x = artistCountsDf.index,

                y = artistCountsDf.artists,

                name = "Top Spotify Tracks 2017",

                marker = dict(color = 'rgba(0, 0, 255, 0.8)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = artistCountsDf.index)

data = [trace1]



layout = go.Layout(barmode = "group",title='Number of Tracks per Artist in Spotify Top 100 List (2017)', yaxis= dict(title= 'Number of Tracks per Artist'))

fig = go.Figure(data = data, layout = layout)

iplot(fig)
artistCounts = top2018['artists'].value_counts()

artistCountsDf = pd.DataFrame(artistCounts)

trace1 = go.Bar(

                x = artistCountsDf.index,

                y = artistCountsDf.artists,

                name = "Top Spotify Tracks 2018",

                marker = dict(color = 'rgba(0, 0, 255, 0.8)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = artistCountsDf.index)

data = [trace1]



layout = go.Layout(barmode = "group",title='Number of Tracks per Artist in Spotify Top 100 List (2018)', yaxis= dict(title= 'Number of Tracks per Artist'))

fig = go.Figure(data = data, layout = layout)

iplot(fig)
frames = [top2017, top2018]

combinedDf = pd.concat(frames)

artistCounts = combinedDf['artists'].value_counts()

artistCountsDf = pd.DataFrame(artistCounts)

trace1 = go.Bar(

                x = artistCountsDf.index,

                y = artistCountsDf.artists,

                name = "Top Spotify Tracks 2017 and 2018",

                marker = dict(color = 'rgba(0, 0, 255, 0.8)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = artistCountsDf.index)

data = [trace1]



layout = go.Layout(barmode = "group",title='Number of Tracks per Artist in Spotify Top 100 List (2017 + 2018)', yaxis= dict(title= 'Number of Tracks per Artist'))

fig = go.Figure(data = data, layout = layout)

iplot(fig)