# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

#worldcloud library
from wordcloud import WordCloud

#matplotlib
import matplotlib.pyplot as plt

#seaborn
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df2015=pd.read_csv("../input/2015.csv")
df2016=pd.read_csv("../input/2016.csv")
df2017=pd.read_csv("../input/2017.csv")
df2015.head()
df2015.info()
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df2015.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
# Happiness Score and Freedom vs Happiness Rank in 2015

# import graph objects as "go"
import plotly.graph_objs as go

# Creating trace1
trace1 = go.Scatter(
                    x = df2015["Happiness Rank"],
                    y = df2015["Happiness Score"],
                    mode = "lines",
                    name = "happiness score",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text= df2015.Country)
# Creating trace2
trace2 = go.Scatter(
                    x = df2015["Happiness Rank"],
                    y = df2015.Freedom,
                    mode = "lines+markers",
                    name = "freedom",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= df2015.Country)
data = [trace1, trace2]
layout = dict(title = 'Happiness Score and Freedom vs Happiness Rank in 2015',
              xaxis= dict(title= 'Happiness Rank',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
# Happiness Score and Economy (GDP per Capita) vs Happiness Rank in 2015

# import graph objects as "go"
import plotly.graph_objs as go

# Creating trace1
trace1 = go.Scatter(
                    x = df2015["Happiness Rank"],
                    y = df2015["Happiness Score"],
                    mode = "lines",
                    name = "happiness score",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text= df2015.Country)
# Creating trace2
trace2 = go.Scatter(
                    x = df2015["Happiness Rank"],
                    y = df2015["Economy (GDP per Capita)"],
                    mode = "lines+markers",
                    name = "economy (GDP per capita)",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= df2015.Country)
data = [trace1, trace2]
layout = dict(title = 'Happiness Score and Economy (GDP per Capita) vs Happiness Rank in 2015',
              xaxis= dict(title= 'Happiness Rank',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)

# Happiness Score and Trust (Government Corruption) vs Happiness Rank in 2015

# import graph objects as "go"
import plotly.graph_objs as go

# Creating trace1
trace1 = go.Scatter(
                    x = df2015["Happiness Rank"],
                    y = df2015["Happiness Score"],
                    mode = "lines",
                    name = "happiness score",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text= df2015.Country)
# Creating trace2
trace2 = go.Scatter(
                    x = df2015["Happiness Rank"],
                    y = df2015["Trust (Government Corruption)"],
                    mode = "lines+markers",
                    name = "trust (government corruption)",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= df2015.Country)
data = [trace1, trace2]
layout = dict(title = 'Happiness Score and Trust (Government Corruption) vs Happiness Rank in 2015',
              xaxis= dict(title= 'Happiness Rank',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)

#Scatter Example: Happiness Score vs Happiness Rank of Countries with 2015, 2016 and 2017 years

# import graph objects as "go"
import plotly.graph_objs as go
# creating trace1
trace1 =go.Scatter(
                    x = df2015["Happiness Rank"],
                    y = df2015["Happiness Score"],
                    mode = "markers",
                    name = "2015",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text= df2015.Country)
# creating trace2
trace2 =go.Scatter(
                    x = df2016["Happiness Rank"],
                    y = df2016["Happiness Score"],
                    mode = "markers",
                    name = "2016",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text= df2016.Country)
# creating trace3
trace3 =go.Scatter(
                    x = df2017["Happiness.Rank"],
                    y = df2017["Happiness.Score"],
                    mode = "markers",
                    name = "2017",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text= df2017.Country)
data = [trace1, trace2, trace3]
layout = dict(title = 'Happiness Score vs Happiness Rank of Countries with 2015, 2016 and 2017 years',
              xaxis= dict(title= 'Happiness Rank',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Happiness Score',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)

#Scatter Example: Freedom vs Happiness Rank of Countries with 2015, 2016 and 2017 years

# import graph objects as "go"
import plotly.graph_objs as go
# creating trace1
trace1 =go.Scatter(
                    x = df2015["Happiness Rank"],
                    y = df2015["Freedom"],
                    mode = "markers",
                    name = "2015",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text= df2015.Country)
# creating trace2
trace2 =go.Scatter(
                    x = df2016["Happiness Rank"],
                    y = df2016["Freedom"],
                    mode = "markers",
                    name = "2016",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text= df2016.Country)
# creating trace3
trace3 =go.Scatter(
                    x = df2017["Happiness.Rank"],
                    y = df2017["Freedom"],
                    mode = "markers",
                    name = "2017",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text= df2017.Country)
data = [trace1, trace2, trace3]
layout = dict(title = 'Freedom vs Happiness Rank of Countries with 2015, 2016 and 2017 years',
              xaxis= dict(title= 'Happiness Rank',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Freedom',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)

#Scatter Example: Family vs Happiness Rank of Countries with 2015, 2016 and 2017 years

# import graph objects as "go"
import plotly.graph_objs as go
# creating trace1
trace1 =go.Scatter(
                    x = df2015["Happiness Rank"],
                    y = df2015["Family"],
                    mode = "markers",
                    name = "2015",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text= df2015.Country)
# creating trace2
trace2 =go.Scatter(
                    x = df2016["Happiness Rank"],
                    y = df2016["Family"],
                    mode = "markers",
                    name = "2016",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text= df2016.Country)
# creating trace3
trace3 =go.Scatter(
                    x = df2017["Happiness.Rank"],
                    y = df2017["Family"],
                    mode = "markers",
                    name = "2017",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text= df2017.Country)
data = [trace1, trace2, trace3]
layout = dict(title = 'Family vs Happiness Rank of Countries with 2015, 2016 and 2017 years',
              xaxis= dict(title= 'Happiness Rank',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Family',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
df2015.Region.value_counts().unique
# Compare Happiness Score average vs Health (Life Expectancy) average of each region in 2015

#preparing data
list_of_region=list(df2015.Region.unique())
region_avg=[]
health_avg=[]
for i in list_of_region:
    x=df2015[df2015.Region==i]
    avg=sum(x["Happiness Score"])/len(x)
    health_av=sum(x["Health (Life Expectancy)"])/len(x)
    region_avg.append(avg)
    health_avg.append(health_av)
data=pd.DataFrame({"Region":list_of_region,"Happiness Average":region_avg,"Health (Life Expectancy) Average":health_avg})
new_index=(data["Happiness Average"].sort_values(ascending=False)).index.values
sorted_data=data.reindex(new_index)

# create trace1 
trace1 = go.Bar(
                x = sorted_data.Region,
                y = sorted_data["Happiness Average"],
                name = "happiness",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = sorted_data.Region)
# create trace2 
trace2 = go.Bar(
                x = sorted_data.Region,
                y = sorted_data["Health (Life Expectancy) Average"],
                name = "health (life expectancy) average",
                marker = dict(color = 'rgba(255, 255, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = sorted_data.Region)
data = [trace1, trace2]
layout = go.Layout(barmode = "group",title= 'Happiness and Health (Life Expectancy) Average of Regions in 2015')
fig = go.Figure(data = data, layout = layout)
iplot(fig)
# Compare Happiness Score average vs Health (Life Expectancy) average of top 4 region in 2015

data1=sorted_data.iloc[:4,:]
# create trace1 
trace1 = go.Bar(
                x = data1.Region,
                y = data1["Happiness Average"],
                name = "happiness",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = data1.Region)
# create trace2 
trace2 = go.Bar(
                x = data1.Region,
                y = data1["Health (Life Expectancy) Average"],
                name = "health (life expectancy) average",
                marker = dict(color = 'rgba(255, 255, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = data1.Region)
data = [trace1, trace2]
layout = go.Layout(barmode = "group",title= 'Happiness and Health (Life Expectancy) Average of top 4 Regions in 2015')
fig = go.Figure(data = data, layout = layout)
iplot(fig)

#another bar example

x=data1.Region

trace1 = {
  'x': x,
  'y': data1["Happiness Average"],
  'name': 'happiness',
  'type': 'bar'
};
trace2 = {
  'x': x,
  'y': data1["Health (Life Expectancy) Average"],
  'name': 'health (life expectancy) average',
  'type': 'bar'
};
data = [trace1, trace2];
layout = {
  'xaxis': {'title': 'Top 4 Regions'},
  'barmode': 'relative',
  'title': 'Happiness and Health (Life Expectancy) Average of top 4 Regions in 2015'
};
fig = go.Figure(data = data, layout = layout)
iplot(fig)
df2015.head()
#Economy of happiest 7 countires in 2015
# data preparation
data2 = df2015.head(7)
pie1 = data2["Economy (GDP per Capita)"]
labels = data2.Country
# figure
fig = {
  "data": [
    {
      "values": pie1,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "Economy Rate",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Economy Rate of the 7 Happiest Countries in 2015",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Economy (GDP per Capita)",
                "x": 0.32,
                "y": 1
            },
        ]
    }
}
iplot(fig)
df2016.head()
#happiness rank of top 50 countries vs generosity with Lower Confidence Interval (size) and Upper Confidence Interval (color) in 2016

data3=df2016.head(50)
data = [
    {
        'y': data3.Generosity,
        'x': data3["Happiness Score"],
        'mode': 'markers',
        'marker': {
            'color': data3["Upper Confidence Interval"],
            'size': data3["Lower Confidence Interval"],
            'showscale': True
        },
        "text" :  data3.Country    
    }
]
iplot(data)
# Lets look at the generosity of the countries in 2015 and 2016

# prepare data
x2015 = df2015.Generosity
x2016 = df2016.Generosity

trace1 = go.Histogram(
    x=x2015,
    opacity=0.75,
    name = "2015",
    marker=dict(color='rgba(171, 50, 96, 0.6)'))
trace2 = go.Histogram(
    x=x2016,
    opacity=0.75,
    name = "2016",
    marker=dict(color='rgba(12, 50, 196, 0.6)'))

data = [trace1, trace2]
layout = go.Layout(barmode='overlay', #overlay=iç ice gecir.
                   title=' generosity of the countries in 2015 and 2016',
                   xaxis=dict(title='generosity'),
                   yaxis=dict( title='count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
# Lets look at the freedom of the countries in 2015 and 2016

# prepare data
x2015 = df2015.Freedom
x2016 = df2016.Freedom

trace1 = go.Histogram(
    x=x2015,
    opacity=0.75,
    name = "2015",
    marker=dict(color='rgba(100, 200, 80, 0.6)'))
trace2 = go.Histogram(
    x=x2016,
    opacity=0.75,
    name = "2016",
    marker=dict(color='rgba(50, 14, 196, 0.6)'))

data = [trace1, trace2]
layout = go.Layout(barmode='overlay', #overlay=iç ice gecir.
                   title=' Freedom of the countries in 2015 and 2016',
                   xaxis=dict(title='freedom'),
                   yaxis=dict( title='count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
# Lets look at the freedom of the countries in 2015,2016 and 2017

# prepare data
x2015 = df2015.Freedom
x2016 = df2016.Freedom
x2017 = df2017.Freedom

trace1 = go.Histogram(
    x=x2015,
    opacity=0.75,
    name = "2015",
    marker=dict(color='rgba(100, 200, 80, 0.6)'))
trace2 = go.Histogram(
    x=x2016,
    opacity=0.75,
    name = "2016",
    marker=dict(color='rgba(50, 14, 196, 0.6)'))

trace3 = go.Histogram(
    x=x2017,
    opacity=0.75,
    name = "2017",
    marker=dict(color='rgba(171, 50, 96, 0.6)'))

data = [trace1, trace2, trace3]
layout = go.Layout(barmode='overlay', #overlay=iç ice gecir.
                   title=' Freedom of the countries in 2015, 2016 and 2017',
                   xaxis=dict(title='freedom'),
                   yaxis=dict( title='count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
#word cloud of the regions in 2015
regions=df2015.Region

plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(regions))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()
#word cloud of the regions in 2015 and 2016
df1=df2015.Region
df2=df2016.Region
df=pd.concat([df1,df2])

plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='yellow',
                          width=512,
                          height=384
                         ).generate(" ".join(df))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()
trace0 = go.Box(
    y=df2015["Happiness Score"],
    name = 'happiness score of countries in 2015',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace1 = go.Box(
    y=df2015.Family,
    name = 'family in 2015',
    marker = dict(
        color = 'rgb(12, 128, 128)',
    )
)
data = [trace0, trace1]
iplot(data)
# import figure factory
import plotly.figure_factory as ff
# prepare data
data2015=df2015.loc[:,["Happiness Score","Family","Economy (GDP per Capita)"]]

data2015["index"] = np.arange(1,len(data2015)+1)
# scatter matrix
fig = ff.create_scatterplotmatrix(data2015, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',
                                  height=700, width=700)
iplot(fig)
# first line plot
trace1 = go.Scatter(
    x=df2015["Happiness Rank"],
    y=df2015["Economy (GDP per Capita)"],
    name = "economy",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)
# second line plot
trace2 = go.Scatter(
    x=df2015["Happiness Rank"],
    y=df2015["Trust (Government Corruption)"],
    xaxis='x2',
    yaxis='y2',
    name = "trust",
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
    title = 'Trust (Government Corruption) and Economy (GDP per Capita) vs Happiness Rank of Countries in 2015'

)

fig = go.Figure(data=data, layout=layout)
iplot(fig)

# create trace 1 that is 3d scatter
trace1 = go.Scatter3d(
    x=df2015["Happiness Rank"],
    y=df2015.Family,
    z=df2015.Freedom,
    mode='markers',
    marker=dict(
        size=10,
        color='rgb(16, 112, 2)',                # set color to an array/list of desired values      
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
    x=df2015["Happiness Rank"],
    y=df2015.Freedom,
    name = "freedom"
)
trace2 = go.Scatter(
    x=df2015["Happiness Rank"],
    y=df2015["Trust (Government Corruption)"],
    xaxis='x2',
    yaxis='y2',
    name = "trust"
)
trace3 = go.Scatter(
    x=df2015["Happiness Rank"],
    y=df2015.Family,
    xaxis='x3',
    yaxis='y3',
    name = "family"
)
trace4 = go.Scatter(
    x=df2015["Happiness Rank"],
    y=df2015["Health (Life Expectancy)"],
    xaxis='x4',
    yaxis='y4',
    name = "health"
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
    title = 'Freedom, trust, family and health vs Happiness Rank of Countries in 2015'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
