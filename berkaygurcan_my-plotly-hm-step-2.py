
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# word cloud library
from wordcloud import WordCloud

# matplotlib
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Load data that we will use.
hapinessdata = pd.read_csv("../input/2015.csv")
hapinessdata.info()
hapinessdata.head(50)
top10hapiness=hapinessdata.iloc[:10,:]
top10hapiness
# data preparation
top10hapiness=hapinessdata.iloc[:10,:]

data = [
    {
        'y': top10hapiness["Economy (GDP per Capita)"],
        'x': top10hapiness["Happiness Rank"],
        'mode': 'markers',
        'marker': {
            'color': top10hapiness.Family,
            'size': top10hapiness["Happiness Score"],
            'showscale': True
        },
        "text" :  top10hapiness.Country    
    }
]

iplot(data)
# create trace 1 that is 3d scatter

trace1 = go.Scatter3d(
    x=top10hapiness["Happiness Rank"],
    y=top10hapiness["Economy (GDP per Capita)"],
    z=top10hapiness["Happiness Score"],
    mode='markers',
    marker=dict(
        size=10,
        color='rgb(255,0,0)',                # set color to an array/list of desired values      
    )
)

data = [trace1]
layout = go.Layout(
    
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    )
    
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
scatterdata=hapinessdata.loc[:,["Economy (GDP per Capita)","Health (Life Expectancy)","Trust (Government Corruption)"]]
scatterdata.head()
# import figure factory
import plotly.figure_factory as ff
# prepare data

scatterdata=hapinessdata.loc[:,["Economy (GDP per Capita)","Health (Life Expectancy)","Trust (Government Corruption)"]]
scatterdata["index"]=np.arange(1,len(scatterdata)+1)
# scatter matrix
fig = ff.create_scatterplotmatrix(scatterdata, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',
                                  height=700, width=700)
iplot(fig)
hapinessdata.columns
trace1 = go.Scatter(
    x=hapinessdata["Happiness Rank"],
    y=hapinessdata["Economy (GDP per Capita)"],
    name = "Economy (GDP per Capita)"
)

trace2 = go.Scatter(
    x=hapinessdata["Happiness Rank"],
    y=hapinessdata["Health (Life Expectancy)"],
    xaxis='x2',
    yaxis='y2',
    name = "Health (Life Expectancy)"
)
trace3 = go.Scatter(
    x=hapinessdata["Happiness Rank"],
    y=hapinessdata["Generosity"],
    xaxis='x3',
    yaxis='y3',
    name = "Generosity"
)
trace4 = go.Scatter(
    x=hapinessdata["Happiness Rank"],
    y=hapinessdata["Trust (Government Corruption)"],
    xaxis='x4',
    yaxis='y4',
    name = "Trust (Government Corruption)"
)
data = [trace1, trace2, trace3, trace4]

layout = go.Layout(
    xaxis=dict(
        domain=[0, 0.45]
    ),
    yaxis=dict(
        domain=[0, 0.45]
    ),
    xaxis2=dict(
        domain=[0.55, 1]
    ),
    xaxis3=dict(
        domain=[0, 0.45],
        anchor='y3'
    ),
    xaxis4=dict(
        domain=[0.55, 1],
        anchor='y4'
    ),
    yaxis2=dict(
        domain=[0, 0.45],
        anchor='x2'
    ),
    yaxis3=dict(
        domain=[0.55, 1]
    ),
    yaxis4=dict(
        domain=[0.55, 1],
        anchor='x4'
    ),
    title = 'Economy (GDP per Capita), Health (Life Expectancy), Generosity and tTrust (Government Corruption) VS Hapiness Rank of Countries'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
