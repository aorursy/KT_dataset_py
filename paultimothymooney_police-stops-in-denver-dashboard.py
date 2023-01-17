import numpy as np

import pandas as pd 

import os

import time

from datetime import date, datetime

from dateutil import parser

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

init_notebook_mode(connected=True)

police_pedestrian_stops_and_vehicle_stops = pd.read_csv('../input/police_pedestrian_stops_and_vehicle_stops.csv')
police_pedestrian_stops_and_vehicle_stops = pd.read_csv('../input/police_pedestrian_stops_and_vehicle_stops.csv',parse_dates=True)

police_pedestrian_stops_and_vehicle_stops2 = police_pedestrian_stops_and_vehicle_stops

police_pedestrian_stops_and_vehicle_stops2['oldDate'] = police_pedestrian_stops_and_vehicle_stops2['TIME_PHONEPICKUP']

police_pedestrian_stops_and_vehicle_stops2['newDate'] = pd.DatetimeIndex(police_pedestrian_stops_and_vehicle_stops2.oldDate).normalize()

dateCounts = police_pedestrian_stops_and_vehicle_stops2['newDate'].value_counts()

dateCountsDf = pd.DataFrame(dateCounts)

dateCountsDf['date2'] = dateCountsDf.index

dateCountsDf = dateCountsDf.sort_values(by='date2')

dateCountsDf = dateCountsDf.resample('W', on='date2').sum()

dateCountsDf = dateCountsDf.reset_index(level='date2')

dateCountsDf = dateCountsDf[(dateCountsDf['date2'].dt.year > 2014)] 



df = dateCountsDf

trace1 = go.Scatter(

                    x = df.date2,

                    y = df.newDate,

                    mode = "lines+markers",

                    name = "gettingStarted",

                    marker = dict(color = 'rgba(0, 0, 255, 0.8)'),)

data = [trace1]

layout = dict(title = 'Weekly Count of Police Stops in Denver',

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),yaxis= dict(title= '# of Police Stops',ticklen= 5,zeroline= False),legend=dict(orientation= "h",x=0, y= 1.13)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
counts = police_pedestrian_stops_and_vehicle_stops['NEIGHBORHOOD_NAME'].value_counts()

counts = pd.DataFrame(counts)[0:10]

trace1 = go.Bar(

                x = counts.index,

                y = counts.NEIGHBORHOOD_NAME,

                name = "Kaggle",

                marker = dict(color = 'blue',

                             line=dict(color='black',width=1.5)),

                text = counts.NEIGHBORHOOD_NAME)

data = [trace1]

layout = go.Layout(barmode = "group",title='Total Incidents by Location')

fig = go.Figure(data = data, layout = layout)

iplot(fig)
counts = police_pedestrian_stops_and_vehicle_stops['PROBLEM'].value_counts()

# print('PROBLEM -- # of Incidents')

# print(counts.head(10))

counts = pd.DataFrame(counts)[0:10]

trace1 = go.Bar(

                x = counts.index,

                y = counts.PROBLEM,

                name = "Kaggle",

                marker = dict(color = 'blue',

                             line=dict(color='black',width=1.5)),

                text = counts.PROBLEM)

data = [trace1]

layout = go.Layout(barmode = "group",title='Total Incidents by Type')

fig = go.Figure(data = data, layout = layout)

iplot(fig)
counts = police_pedestrian_stops_and_vehicle_stops['CALL_DISPOSITION'].value_counts()

counts = pd.DataFrame(counts)[0:10]

trace1 = go.Bar(

                x = counts.index,

                y = counts.CALL_DISPOSITION,

                name = "Kaggle",

                marker = dict(color = 'blue',

                             line=dict(color='black',width=1.5)),

                text = counts.CALL_DISPOSITION)

data = [trace1]

layout = go.Layout(barmode = "group",title='Total Incidents by Disposition')

fig = go.Figure(data = data, layout = layout)

iplot(fig)