import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode,iplot,plot,download_plotlyjs

init_notebook_mode(connected=True)

%matplotlib inline
## DATA RETRIEVAL ##
file = 'https://covid.ourworldindata.org/data/ecdc/total_cases.csv'

df = pd.read_csv(file,na_values='0')

count = df.iloc[-1].loc['Afghanistan':'Zimbabwe'].index

df2 = pd.DataFrame()

df2['countries'] = count

ser = df[count].iloc[-1]

data = ser.array

df2['affected'] = data
## VISUALISATION ##
data = dict(

            type = 'choropleth',

            locations = df2['countries'],

            locationmode = "country names",

            z = df2['affected'],

            colorscale = 'Earth',

            text = df2['countries'],

            colorbar = dict(title = 'total no. of cases')

            )
layout = dict(

                title = 'CoVID-19 global scenario',

                geo = dict(   showframe = True,

                              projection = dict(type = 'orthographic'),

                              showocean = True,

                              showlakes = True

                            )

                )
choromap = go.Figure([data],layout = layout)

iplot(choromap,validate=False)