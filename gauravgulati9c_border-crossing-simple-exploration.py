import os

import warnings

%matplotlib inline

import numpy as np

import pandas as pd

import seaborn as sns

import plotly.express as px

import matplotlib.pyplot as plt

import chart_studio.plotly as py

warnings.filterwarnings("ignore")

import plotly.graph_objects as go
df = pd.read_csv("../input/border-crossing-entry-data/Border_Crossing_Entry_Data.csv")
df.info()
print("Shape of DataFrame", df.shape)

print(" DataFrame Columns", df.columns)
print("Different Crossing Ports along with there counts")

print(df['Port Name'].value_counts())
def getLat(word):

    x = word.split(' ')

    lat = x[1][1:]

    return lat



def getLong(word):

    x = word.split(' ')

    long = x[2][:-1]

    return long



df['Latitude'] = df['Location'].apply(getLat)

df['Longitude'] = df['Location'].apply(getLong)
BoundaryDimensions = (df.Longitude.min(),   df.Longitude.max(),df.Latitude.min(), df.Latitude.max())

print(BoundaryDimensions)
x = df['State'].value_counts().index

y = df['State'].value_counts().values

fig = go.Figure(data=[go.Bar(

            x=x, y=y,

            text=y,

            textposition='auto',

            marker={'color': 'royalblue'}

        )])



fig.update_layout(

    title={

    'text': "State Wise Distribution",

    'y':0.9,

    'x':0.5,

    'xanchor': 'center',

    'yanchor': 'top'},

    xaxis_title="States",

    yaxis_title="Instances",

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="#7f7f7f"

    )

)
x = df['Border'].value_counts().index

y = df['Border'].value_counts().values



fig = go.Figure(data=[go.Bar(

            x=x, y=y,

            text=y,

            textposition='auto',

            marker={'color': 'royalblue'},

            name='BORDER'

        )

])



fig.update_layout(

    title={

    'text': "Border Distribution",

    'y':0.9,

    'x':0.5,

    'xanchor': 'center',

    'yanchor': 'top'},

    xaxis_title="Borders",

    yaxis_title="Instances",

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="#7f7f7f"

    )

)
df['Measure'].value_counts()
x = df['Measure'].value_counts().index

y = df['Measure'].value_counts().values



fig = go.Figure(data=[go.Bar(

            x=x, y=y,

            text=y,

            textposition='auto',

            marker={'color': 'royalblue'},

        )

])



fig.update_layout(

    title={

    'text': "Measure Distribution",

    'y':0.9,

    'x':0.5,

    'xanchor': 'center',

    'yanchor': 'top'},

    xaxis_title="Measures of Cross",

    yaxis_title="Instances",

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="#7f7f7f"

    )

)
df['Date'] = pd.to_datetime(df['Date'])
df.dtypes
df['Year'] =  df.Date.dt.year

df['Month'] = df.Date.dt.month
df[['Date', 'Year', 'Month']].sample(5)
frame = { 'Index': df[df['Year'] < 2019].Year.value_counts().sort_index().index, 'Values': df[df['Year'] < 2019].Year.value_counts().sort_index() } 

result = pd.DataFrame(frame) 



fig = px.line(result, x="Index", y="Values", title="Year-Wise Crossings", labels={'Index': 'Year', 'Values': 'Number of Crossings'})

fig.show()