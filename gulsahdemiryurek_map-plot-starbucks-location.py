# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualization library

import plotly.plotly as py # visualization library

from plotly.offline import init_notebook_mode, iplot # plotly offline mode

init_notebook_mode(connected=True) 

import plotly.graph_objs as go # plotly graphical object

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
star=pd.read_csv("../input/directory.csv")

star["color"]="rgb(15,78,250)"
data = [dict(

    type='scattergeo',

    lon = star['Longitude'],

    lat = star['Latitude'],

    hoverinfo = 'text',

    text = "Country: "+star["Country"] +"City: " + star.City + " Street Adress: "+star["Street Address"]+" Store Name: " + star['Store Name'],

    mode = 'markers',

    marker=dict(

        sizemode = 'area',

        sizeref = 1,

        size= 10 ,

        line = dict(width=1,color = "rosybrown"),

        color = star["color"],

        symbol = 'cross',

        opacity = 0.7),

)]

layout = dict(

    title = 'Starbucks Locations ',

    hovermode='closest',

    geo = dict(showframe=True, showland=True, showcoastlines=True, showcountries=True,

               countrywidth=1, projection=dict(type='miller'),

              landcolor = 'rgb(217, 217, 217)',

              subunitwidth=1,

              showlakes = True,

              lakecolor = 'firebrick',

              countrycolor="rgb(5, 5, 5)")

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)