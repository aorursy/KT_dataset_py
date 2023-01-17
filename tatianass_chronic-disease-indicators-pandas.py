### Working with data ###

import numpy as np

import pandas as pd



### Visualization ###

# plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



### Utils ###

import datetime

import warnings



# File format #

import csv

import json

import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/disease_indicators.csv", sep=',')
df.shape
df.columns
df['GeoLocation'].head(2)
df_location = df[(df['GeoLocation'].notnull())]
df_location.shape
df_location['GeoLocation'].head(2)
df_location.describe().T
# Convert to float and transform errors in nan

df_location['DataValue'] = pd.to_numeric(df_location['DataValue'], errors='coerce')
df_location.shape
top_10_cities = (df_location.assign(rank=df_location.groupby(['Topic'])['DataValue']

                     .rank(method='dense', ascending=False))

                     .query('rank <= 10')

                     .sort_values(['Topic', 'rank']))
top_10_cities.rename(columns={'LocationDesc':'city'}, inplace=True)
to_n_ranks_cities = (

    top_10_cities

    .groupby('city')['Topic']

    .nunique()

    .to_frame('Number of times in top 10')

    .reset_index()

    .sort_values(['Number of times in top 10'], ascending=True) # Since the orientation is horizontal, the sort must be the inverse order of what I want

)
data = [go.Bar(

            y=to_n_ranks_cities['city'],

            x=to_n_ranks_cities['Number of times in top 10'],

            orientation = 'h',

            text=to_n_ranks_cities['Number of times in top 10']

    )]



layout = go.Layout(

    title='Frequency of Cities in top 10 ranking for diseases',

    titlefont=dict(size=20),

    width=1000,

    height=1400,

    yaxis=go.layout.YAxis(

        ticktext=to_n_ranks_cities['city'],

        tickmode='array',

        automargin=True

    )

)



fig = go.Figure(data=data, layout=layout)



iplot(fig)