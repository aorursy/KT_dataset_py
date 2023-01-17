import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

pd.set_option('display.max_rows', None)

import datetime

from plotly.subplots import make_subplots
import io

import requests

url="https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv"

s=requests.get(url).content

time_series_data = pd.read_csv(io.StringIO(s.decode('utf-8')))

time_series_data.head()
Data_Nepal = time_series_data[(time_series_data['Country/Region'] == 'Nepal') ].reset_index(drop=True)

Data_Nepal.head()
Data_Nepal['Active'] = Data_Nepal['Confirmed']-(Data_Nepal['Recovered'] + Data_Nepal['Deaths'])
Data_Nepal.tail()
fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_Nepal['Date'], y=Data_Nepal['Confirmed'],

                    mode='lines',

                    name='Confirmed cases'))



fig.add_trace(go.Scatter(x=Data_Nepal['Date'], y=Data_Nepal['Active'],

                    mode='lines',

                    name='Active cases',line=dict( dash='dot')))

fig.add_trace(go.Scatter(x=Data_Nepal['Date'], y=Data_Nepal['Deaths'],name='Deaths',

                                   marker_color='black',mode='lines',line=dict( dash='dot') ))

fig.add_trace(go.Scatter(x=Data_Nepal['Date'], y=Data_Nepal['Recovered'],

                    mode='lines',

                    name='Recovered cases',marker_color='green'))

fig.update_layout(

    title='Evolution of cases over time in Nepal',

    template='plotly_dark',



)



fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_Nepal.index, y=Data_Nepal['Confirmed'],

                    mode='markers',

                    name='Confirmed cases'))





fig.update_layout(

    title='Evolution of Confirmed cases over time in Nepal',

        template='plotly_dark'



)



fig.show()
fig = go.Figure()





fig.add_trace(go.Scatter(x=Data_Nepal.index, y=Data_Nepal['Active'],

                    mode='lines',marker_color='yellow',

                    name='Active cases',line=dict( dash='dot')))



fig.update_layout(

    title='Evolution of Acitive cases over time in Nepal',

        template='plotly_dark'



)



fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(x=Data_Nepal.index, y=Data_Nepal['Recovered'],

                    mode='lines',

                    name='Recovered cases',marker_color='green'))



fig.update_layout(

    title='Evolution of Recovered cases over time in Nepal',

        template='plotly_dark'



)



fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(x=Data_Nepal.index, y=Data_Nepal['Deaths'],name='Deaths',

                                   marker_color='red',mode='lines',line=dict( dash='dot') ))



fig.update_layout(

    title='Evolution of Deaths over time in Nepal',

        template='plotly_dark'



)



fig.show()