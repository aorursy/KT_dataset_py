# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import ipywidgets as widgets

from ipywidgets import interact, interactive

sns.set()

init_notebook_mode(connected=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data15=pd.read_csv('../input/2015.csv')

data16=pd.read_csv('../input/2016.csv')

data17=pd.read_csv('../input/2017.csv')

data17=data17.rename(columns={'Happiness.Rank':'Happiness Rank','Happiness.Score':'Happiness Score'})

data17.head()

dataall=[data15,data16,data17]

fig=[]

scl = [[0.0, '#7d899c'],

       [0.2, '#637082'],

       [0.4, '#4d5765'],

       [0.6, '#373e48'],

       [0.8, '#21252b'],

       [1.0, '#0b0c0e']]

for d in dataall:

    data = [go.Choropleth(

        colorscale = scl,

        autocolorscale = False,

        locations = d['Country'],

        z = d['Happiness Rank'],

        locationmode = 'country names',

        marker = go.choropleth.Marker(

            line = go.choropleth.marker.Line(

                color = 'rgb(255,255,255)',

                width = 1

            )),

        colorbar = go.choropleth.ColorBar(

            title = 'Rank')

    )]

    layout = go.Layout(

        width=750,

        height=700,

        title = go.layout.Title(

            text = 'World Happiness Ranking'

        ),

        geo = go.layout.Geo(

            scope = 'world',

            projection = go.layout.geo.Projection(),

            showlakes = False),

    )

    f = go.Figure(data = data, layout = layout)

    fig.append(f)

count=2015

for i in fig:

    print(str(count)+' Ranking')

    count+=1

    iplot(i)

x=[2015,2016,2017]

y=[]

top=[]

worst=[]

avg=[]

for d in dataall:

    y.append(d[d['Country']=='India']['Happiness Score'].item())

    top.append(d[d['Happiness Score']==np.max(d['Happiness Score'])]['Happiness Score'].item())

    worst.append(d[d['Happiness Score']==np.min(d['Happiness Score'])]['Happiness Score'].item())

    avg.append(np.average(d['Happiness Score']))

print(x,y,top,worst,avg)

layout = go.Layout(

    xaxis=dict(

        title='Year',

        titlefont=dict(

            family='Arial, sans-serif',

            size=18,

            color='black'

        ),

        nticks=3,

        showticklabels=True,

        tickangle=45,

        tickfont=dict(

            family='Old Standard TT, serif',

            size=14,

            color='black'

        ),

        exponentformat='e',

        showexponent='all'

    ),

    yaxis=dict(

        title='Happiness score',

        titlefont=dict(

            family='Arial, sans-serif',

            size=18,

            color='black'

        ),

        showticklabels=True,

        tickangle=45,

        tickfont=dict(

            family='Old Standard TT, serif',

            size=14,

            color='black'

        ),

        exponentformat='e',

        showexponent='all'

    )

)

data=[go.Scatter(x=x,y=y,name='India'),go.Scatter(x=x,y=top,mode='markers',name='Best score',marker = dict(

        size = 5,

        color = 'lightgreen',

        line = dict(

            width = 2,

        )

    )),go.Scatter(x=x,y=worst,mode='markers',name='Worst Score',marker = dict(

        size = 5,

        color = 'red',

        line = dict(

            width = 2,

        )

    )),go.Scatter(x=x,y=avg,mode='markers+lines',name='Average Score',marker = dict(

        size = 5,

        color = 'purple',

        line = dict(

            width = 2,

        )

    ))]

fig=go.Figure(data=data,layout=layout)

iplot(fig)