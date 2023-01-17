# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cluster import KMeans
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/world-happiness/2015.csv')
print(df.head())
map_data = pd.read_csv('../input/countries/countries.csv')
map_data.head()
scl = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(240, 210, 250)"]]


# scl = [[0.0, 'rgb(50,10,143)'],[0.2, 'rgb(117,107,177)'],[0.4, 'rgb(158,154,200)'],\
#             [0.6, 'rgb(188,189,220)'],[0.8, 'rgb(218,208,235)'],[1.0, 'rgb(250,240,255)']]
data = dict(type = 'choropleth', 
            colorscale = scl,
            autocolorscale = True,
            reversescale = True,
           locations = df['Country'],
           locationmode = 'country names',
           z = df['Happiness Rank'], 
           text = df['Country'],
           colorbar = {'title':'Happiness'})
layout = dict(title = 'Global Happiness', 
             geo = dict(showframe = False, 
                       projection = {'type': 'Orthographic'}))

choromap3 = go.Figure(data = [data], layout=layout)
iplot(choromap3)
df = df.merge(map_data, how='left', on = ['Country'])
df.head()
df['Happiness Score'].min(), df['Happiness Score'].max()
df['text']=df['Country'] + '<br>Happiness Score ' + (df['Happiness Score']).astype(str)
scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
    [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]

data = [ dict(
        type = 'scattergeo',
        locationmode = 'country names',
        lon = df['Longitude'],
        lat = df['Latitude'],
        text = df['text'],
        mode = 'markers',
        marker = dict(
            size = df['Happiness Score']*3,
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            symbol = 'circle',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale = scl,
            cmin = 0,
            color = df['Economy (GDP per Capita)'],
            cmax = df['Economy (GDP per Capita)'].max(),
            colorbar=dict(
                title="GDP per Capita"
            )
        ))]

layout = dict(
        title = 'Happiness Scores by GDP',
        geo = dict(
#             scope='usa',
            projection=dict( type='Mercator' ),
            showland = True,
            landcolor = "rgb(250, 250, 250)",
#             subunitcolor = "rgb(217, 217, 217)",
#             countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )

symbolmap = go.Figure(data = data, layout=layout)

iplot(symbolmap)
