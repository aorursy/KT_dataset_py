# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

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
gamesdata = pd.read_csv("../input/vgsales.csv")
gamesdata.info()
gamesdata.columns
gamesdata.head()
gamesdata.tail()
#preparing data frame

df = gamesdata.iloc[:100,:]



#import graph objects as "go"

import plotly.graph_objs as go



#creating trace1

trace1 = go.Scatter(

                    x = df.Rank,

                    y = df.NA_Sales,

                    mode = "lines",

                    name = "NA Sales",

                    marker = dict(color="rgba(166,11,2,0.8)"),

                    text = df.Name)

#creating trace2

trace2 = go.Scatter(

                    x = df.Rank,

                    y = df.EU_Sales,

                    mode = "lines+markers",

                    name = "EU Sales",

                    marker = dict(color = "rgba(80,12,160,0.5)"),

                    text = df.Name)

data = [trace1,trace2]

layout = dict(title = "Global Sales of Top 100 Games",

                xaxis = dict(title="Rank",ticklen= 5, zeroline=False)

             )

fig = dict(data = data, layout = layout)

py.offline.iplot(fig)
gamesdata["Platform"].unique()
platformlist = list(gamesdata["Platform"].unique())

platformearns = []

for i in platformlist:

    x = gamesdata[gamesdata["Platform"] == i]

    sums = sum(x.Global_Sales)

    platformearns.append(sums)

    

data = pd.DataFrame({"platformlist": platformlist, "platformearns": platformearns})

new_index = (data["platformearns"].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)



#create trace1

trace1 = go.Bar(

                x = sorted_data.platformlist,

                y = sorted_data.platformearns,

                name = "Global Sales of Platforms",

                marker = dict(color = "rgba(10,255,50,0.7)",line = dict(color="rgb(0,0,0)",width=1)))

data = [trace1]

layout = dict(title = "Global Sales of Gaming Platforms")

fig = go.Figure(data = data, layout = layout)

py.offline.iplot(fig)
# data preparetion

df2016 = gamesdata[gamesdata.Year == 2016].iloc[:10,:]

pie = df2016.Global_Sales

labels = df2016.Name

publisher = df2016.Publisher

# figure



fig = {

    "data": [

        {

            "values": pie,

            "labels": labels,

            "domain":{"x": [0,.5]},

            "name": "Sale Rate",

            "hoverinfo": "label+percent+name",

            "hole": .3,

            "type":"pie"

        },],

    "layout": {

        "title":"Top 10 Market Leaders In 2016",

        "annotations": [

            { "font": { "size": 15},

              "showarrow": True,

              "text": "Games",

                "x": 0.20,

                "y": 1

    },]}



}

py.offline.iplot(fig)
#preparing data

publisherlist = list(gamesdata["Publisher"].unique())

publisherearns = []

for i in publisherlist:

    y = gamesdata[gamesdata["Publisher"] == i]

    sums1 = sum(y.Global_Sales)

    publisherearns.append(sums1)

    

data1 = pd.DataFrame({"Publisher": publisherlist,"publisherearns":publisherearns})

new_index = (data1["publisherearns"].sort_values(ascending=False).index.values)

sorted_data = data1.reindex(new_index)



df = sorted_data.iloc[:20]

df["Rank"] = "1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"

df["NA_Sales"] = gamesdata.NA_Sales

#visualization

data = [

    {

        "x":df.Rank,

        "y":df.publisherearns,

        "mode": "markers",

        "marker":{

            "color": df.NA_Sales,

            "size": gamesdata.Publisher.value_counts()/10,

            "showscale": True

        },

        "text": df.Publisher

    }

]

py.offline.iplot(data)
x2016 = gamesdata.Genre[gamesdata.Year == 2016]

x2006 = gamesdata.Genre[gamesdata.Year == 2010]



trace1 = go.Histogram(

                        x = x2016,

                        opacity = 0.75,

                        name = "2016",

                        marker = dict(color="rgba(162,50,70,0.9)"))

trace2 = go.Histogram(

                        x = x2006,

                        opacity = 0.75,

                        name = "2010",

                        marker = dict(color="rgba(24,68,200,0.6)"))



data = [trace1,trace2]

layout = go.Layout(barmode = "overlay",

                    title = "Number of Genres in 2016 and 2010 ",

                  xaxis = dict(title="Genre"),

                  yaxis = dict(title = "Count"),

                  )

fig = go.Figure(data=data,layout=layout)

py.offline.iplot(fig)
trace1 = go.Scatter3d(

    x=gamesdata.Rank.iloc[:100],

    y=gamesdata.JP_Sales,

    z=gamesdata.EU_Sales,

    mode="markers",

    marker=dict(

        size=10,

        color="rgb(216,34,78)",#set color to an array/list of desired values

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

fig = go.Figure(data=data,layout=layout)

py.offline.iplot(fig)