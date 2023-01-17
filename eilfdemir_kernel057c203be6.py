# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# for plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# for worldcloud library
from wordcloud import WordCloud

# for matplotlib
import matplotlib.pyplot as plt

# for seaborn
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df1 = pd.read_csv("../input/countries of the world.csv")
df1.head(10)
df1.info()
# Population and Area (sq. mi.) of Countries of the World

import plotly.graph_objs as go

trace1 = go.Scatter(
                    x = df1["Population"],
                    y = df1["Area (sq. mi.)"],
                    mode = "lines",
                    name = "Area (sq. mi.)",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text= df1.Country)
trace2 = go.Scatter(
                    x = df1["Population"],
                    y = df1.Population,
                    mode = "lines+markers",
                    name = "population",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= df1.Country)
data = [trace1, trace2]
layout = dict(title = 'Population and Area (sq. mi.) of Countries of the World',
              xaxis= dict(title= 'population',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
# Population and GDP ($ per capita) of Countries of the World

import plotly.graph_objs as go

trace1 = go.Scatter(
                    x = df1["Population"],
                    y = df1["GDP ($ per capita)"],
                    mode = "lines",
                    name = "GDP ($ per capita)",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text= df1.Country)
trace2 = go.Scatter(
                    x = df1["Population"],
                    y = df1.Population,
                    mode = "lines+markers",
                    name = "Population",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= df1.Country)
data = [trace1, trace2]
layout = dict(title = 'Population and GDP ($ per capita) of Countries of the World',
              xaxis= dict(title= 'Population',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(df1.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
# Scatter Example for Population and Area (sq. mi.) of Countries of the World


import plotly.graph_objs as go

trace1 =go.Scatter(
                    x = df1["Population"],
                    y = df1["Area (sq. mi.)"],
                    mode = "markers",
                    name = "Area (sq. mi.)",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text= df1.Country)

trace2 =go.Scatter(
                    x = df1["Population"],
                    y = df1.Population,
                    mode = "markers",
                    name = "Population",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text= df1.Country)

data = [trace1, trace2]
layout = dict(title = 'Population and Area (sq. mi.) of Countries of the World',
              xaxis= dict(title= 'Population',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Area (sq. mi.)',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
# Scatter Example for Population and GDP ($ per capita) of Countries of the World


import plotly.graph_objs as go

trace1 =go.Scatter(
                    x = df1["Population"],
                    y = df1["GDP ($ per capita)"],
                    mode = "markers",
                    name = "GDP ($ per capita)",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text= df1.Country)

trace2 =go.Scatter(
                    x = df1["Population"],
                    y = df1.Population,
                    mode = "markers",
                    name = "Population",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text= df1.Country)

data = [trace1, trace2]
layout = dict(title = 'Population and GDP ($ per capita) of Countries of the World',
              xaxis= dict(title= 'Population',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'GDP ($ per capita)',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
df1.Region.value_counts().unique
import plotly.graph_objs as go

trace1 = go.Bar(
                x = df1.Country,
                y = df1.Region,
                name = "region",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df1.Country)

trace2 = go.Bar(
                x = df1.Country,
                y = df1.Industry,
                name = "industry",
                marker = dict(color = 'rgba(255, 255, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df1.Country)
data = [trace1, trace2]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)
import plotly.graph_objs as go

trace1 = go.Bar(
                x = df1.Country,
                y = df1.Birthrate,
                name = "birthrate",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df1.Country)

trace2 = go.Bar(
                x = df1.Country,
                y = df1.Deathrate,
                name = "deathrate",
                marker = dict(color = 'rgba(255, 255, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df1.Country)
data = [trace1, trace2]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)
import plotly.graph_objs as go
x = df1.Country
trace1 = {
  'x': x,
  'y': df1.Agriculture,
  'name': 'agriculture',
  'type': 'bar'
};
trace2 = {
  'x': x,
  'y': df1.Service,
  'name': 'service',
  'type': 'bar'
};
data = [trace1, trace2];
layout = {
  'barmode': 'relative',
  'title': 'Agriculture and Service of Countries'
};
fig = go.Figure(data = data, layout = layout)
iplot(fig)
import plotly.graph_objs as go
x = df1.Country
trace1 = {
  'x': x,
  'y': df1.Population,
  'name': 'population',
  'type': 'bar'
};
trace2 = {
  'x': x,
  'y': df1.Region,
  'name': 'region',
  'type': 'bar'
};
data = [trace1, trace2];
layout = {
  'barmode': 'relative',
  'title': 'Population and Region of Countries'
};
fig = go.Figure(data = data, layout = layout)
iplot(fig)
data = df1.head()
pie1 = data["Population"]
labels = data.Country

fig = {
  "data": [
    {
      "values": pie1,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "Population of Countries",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Population of Countries",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Population",
                "x": 0.32,
                "y": 1
            },
        ]
    }
}
iplot(fig)
data = df1.head(10)
pie1 = data["GDP ($ per capita)"]
pie2 = data["Area (sq. mi.)"]
labels = data.Country

fig = {
  "data": [
    {
      "values": pie1,
      "values": pie2,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "GDP ($ per capita) and Area (sq. mi.) of Countries of the World",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"GDP ($ per capita) and Area (sq. mi.) of Countries of the World",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "GDP ($ per capita) and Area (sq. mi.)",
                "x": 0.32,
                "y": 1
            },
        ]
    }
}
iplot(fig)
data1 = df1.Country
data2 = df1.Region
trace1 = go.Histogram(
    x=data1,
    opacity=0.75,
    name = "Country",
    marker=dict(color='rgba(171, 50, 96, 0.6)'))
trace2 = go.Histogram(
    y=data2,
    opacity=0.75,
    name = "Region",
    marker=dict(color='rgba(12, 50, 196, 0.6)'))

data = [trace1, trace2]
layout = go.Layout(barmode='overlay',
                   title=' Countries and Regions ',
                   xaxis=dict(title='Countries'),
                   yaxis=dict( title='Regions'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

data = df1.head(100)
data1 = [
    {
        'y': data.Country,
        'x': data["Country"],
        'mode': 'markers',
        'marker': {
            'color': data["GDP ($ per capita)"],
            'size': data["Population"],
            'showscale': True
        },
        "text" :  data.Country    
    }
]
iplot(data1)
data=df1.Region

plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(data))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()
data1=df1.Region
data2=df1.Region
data=pd.concat([data1,data2])

plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='purple',
                          width=512,
                          height=384
                         ).generate(" ".join(data))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()
trace1 = go.Box(
    x=df1["GDP ($ per capita)"],
    name = 'GDP ($ per capita) of Countries',
    marker = dict(
        color = 'rgb(50, 500, 300)',
    )
)
trace2 = go.Box(
    x=df1.Population,
    name = 'Population of Countries',
    marker = dict(
        color = 'rgb(50, 128, 128)',
    )
)
data = [trace1, trace2]
iplot(data)
trace1 = go.Scatter(
    x=df1["Country"],
    y=df1["Service"],
    name = "services",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)
# second line plot
trace2 = go.Scatter(
    x=df1["Country"],
    y=df1["Industry"],
    xaxis='x2',
    yaxis='y2',
    name = "industry",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)
data = [trace1, trace2]
layout = go.Layout(
    xaxis2=dict(
        domain=[0.6, 0.95],
        anchor='y2',        
    ),
    yaxis2=dict(
        domain=[0.6, 0.95],
        anchor='x2',
    ),
    title = 'Service and Industry of Countries'

)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
trace1 = go.Scatter3d(
    x=df1["Country"],
    y=df1.Agriculture,
    z=df1.Industry,
    mode='markers',
    marker=dict(
        size=10,
        color='rgb(16, 112, 2)',                      
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
trace1 = go.Scatter(
    x=df1["Country"],
    y=df1.Population,
    name = "population"
)
trace2 = go.Scatter(
    x=df1["Country"],
    y=df1["Agriculture"],
    xaxis='x2',
    yaxis='y2',
    name = "agriculture"
)
trace3 = go.Scatter(
    x=df1["Country"],
    y=df1.Region,
    xaxis='x3',
    yaxis='y3',
    name = "region"
)
trace4 = go.Scatter(
    x=df1["Country"],
    y=df1["Service"],
    xaxis='x4',
    yaxis='y4',
    name = "service"
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
    title = 'Population, Agriculture, Region and Service of Countries'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)