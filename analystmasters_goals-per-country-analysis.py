import pandas as pd

import plotly.plotly as py

import plotly.graph_objs as go

import numpy as np

import os 

import sys

import plotly.plotly as py



cwd=os.getcwd()

cwd

os.chdir('/kaggle/input/')

all_list=os.listdir()

if len(all_list)<5:

    os.chdir('/kaggle/input/earn-your-6-figure-prize/')
FTHT6 = pd.read_csv('FT_HT6.csv')

ranks6 = pd.read_csv('ranks6.csv')

winrate6 = pd.read_csv('winrate6.csv')

country6 = pd.read_csv('country6.csv')

names6 = pd.read_csv('names6.csv')

fresults6 = pd.read_csv('fresults6.csv')
import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import seaborn as sns

import time

import warnings

warnings.filterwarnings('ignore')

   

data = [ dict(

        type = 'choropleth',

        locations = country6,

        z = (fresults6.iloc[0]+fresults6.iloc[1]),

        locationmode = 'country names',

        text = country6,

        marker = dict(

            line = dict(color = 'rgb(0,0,0)', width = 1)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = '# Average\nTemperature,\n°C')

            )

       ]



layout = dict(

    title = 'Average land temperature in countries',

    geo = dict(

        showframe = False,

        showocean = True,

        oceancolor = 'rgb(0,255,255)',

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 60,

                    lat = 10),

        ),

        lonaxis =  dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

            ),

        lataxis = dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )



fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='worldmap')