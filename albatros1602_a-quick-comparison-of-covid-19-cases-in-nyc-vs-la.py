# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go

import matplotlib.pyplot as plt

from wordcloud import WordCloud

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
covidData = pd.read_csv("../input/us-counties-covid-19-dataset/us-counties.csv")
covidData.info()
covidData.head(50)
#Since this is a very big data, I would like to focus on NYC and LA only.
df_NYC = covidData[covidData['county'] == 'New York City']
df_LA = covidData[covidData['county'] == 'Los Angeles']
#I add a timeline column to the data which will show the cases irrelevant to the dates. 
#In this instance the first case detected in a city is d1, which corresponds to 2020-03-01 for NYC and 2020-01-26 for LA.
df_NYC['timeline'] = ""
df_LA['timeline'] = ""
#I fill the timeline column
i = 1
for each in df_NYC.index:
    df_NYC.loc[each, 'timeline'] = ('d' + str(i))
    i += 1

i = 1
for each in df_LA.index:
    df_LA.loc[each, 'timeline'] = ('d' + str(i))
    i += 1
df_NYC.head()
df_LA.head()
#Let's make a line chart
import plotly.graph_objs as go
trace1 = go.Scatter(
                    x = df_NYC.timeline,
                    y = df_NYC.cases,
                    mode = "lines",
                    name = "cases in NYC",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text = (df_NYC.county))

trace2 = go.Scatter(
                    x = df_NYC.timeline,
                    y = df_NYC.deaths,
                    mode = "lines",
                    name = "deaths in NYC",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text = (df_NYC.county))
trace3 = go.Scatter(
                    x = df_LA.timeline,
                    y = df_LA.cases,
                    mode = "lines",
                    name = "cases in LA",
                    marker = dict(color = 'rgba(150, 170, 15, 0.8)'),
                    text = (df_LA.county))

trace4 = go.Scatter(
                    x = df_LA.timeline,
                    y = df_LA.deaths,
                    mode = "lines",
                    name = "deaths in LA",
                    marker = dict(color = 'rgba(200, 35, 198, 0.8)'),
                    text = (df_LA.county))
data = [trace1, trace2, trace3, trace4]
layout = dict(title = 'Covid-19 Cases & Deaths in New York City VS Los Angeles by Timeline',
             xaxis = dict(title = 'Timeline', ticklen = 5, zeroline = False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
#Lets make a scatter plot
import plotly.graph_objs as go

trace1 =go.Scatter(
                    x = df_NYC.timeline,
                    y = df_NYC.cases,
                    mode = "markers",
                    name = "NYC",
                    marker = dict(color = 'rgba(200, 35, 198, 0.8)'),
                    text = "NYC")

trace2 =go.Scatter(
                    x = df_LA.timeline,
                    y = df_LA.cases,
                    mode = "markers",
                    name = "LA",
                    marker = dict(color = 'rgba(150, 170, 15, 0.8)'),
                    text = "LA")

data = [trace1, trace2]
layout = dict(title= 'Covid-19 Cases in NYC VS LA',
             xaxis = dict(title = 'Timeline', ticklen = 5, zeroline = False),
             yaxis = dict(title = 'Cases', ticklen = 5, zeroline = True)
             )

fig = dict(data = data, layout = layout)
iplot(fig)
#Lets make a bar chart
#Barmode = group
import plotly.graph_objs as go

trace1 =go.Bar(
                    x = df_NYC.timeline,
                    y = df_NYC.cases,
                    name = "cases",
                    marker = dict(color = 'rgba(200, 35, 198, 0.8)',
                                 line = dict(color = 'rgb(0,0,0)',width=1.5)),
                    text = df_NYC.county)

trace2 =go.Bar(
                    x = df_NYC.timeline,
                    y = df_NYC.deaths,
                    name = "deaths",
                    marker = dict(color = 'rgba(150, 170, 15, 0.8)',
                                 line = dict(color = 'rgb(0,0,0)',width=1.5)),
                    text = df_NYC.county)

data = [trace1, trace2]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)
#Lets make a bar chart
#Barmode = stack
import plotly.graph_objs as go

trace1 =go.Bar(
                    x = df_NYC.timeline,
                    y = df_NYC.cases,
                    name = "cases",
                    marker = dict(color = 'rgba(200, 35, 198, 0.8)',
                                 line = dict(color = 'rgb(0,0,0)',width=1.5)),
                    text = df_NYC.county)

trace2 =go.Bar(
                    x = df_NYC.timeline,
                    y = df_NYC.deaths,
                    name = "deaths",
                    marker = dict(color = 'rgba(150, 170, 15, 0.8)',
                                 line = dict(color = 'rgb(0,0,0)',width=1.5)),
                    text = df_NYC.county)

data = [trace1, trace2]
layout = go.Layout(barmode = "stack")
fig = go.Figure(data = data, layout = layout)
iplot(fig)
#Pie Chart
df_NYC_d14 = df_NYC[df_NYC['timeline'] == 'd14']
df_LA_d14 = df_LA[df_LA['timeline'] == 'd14']
df_pie = df_NYC_d14.append(df_LA_d14)
pie1 = df_pie.cases
pie1_list = [each for each in df_pie.cases]
labels = df_pie.county
fig = {
    "data": [
        {
            "values": pie1_list,
            "labels": labels,
            "domain": {"x": [0, .5]},
            "name": "Number of Cases Rates",
            "hoverinfo": "label+percent+name",
            "hole": .3,
            "type": "pie"
        },],
    "layout": {
        "title": "Number of Cases in NYC VS LA on Day 14",
        "annotations": [
            {"font": {"size":20},
             "showarrow": False,
             "text": "Number of Cases on Day 14",
             "x": 0.20,
             "y": 1
            },
        ]
    }
}
iplot(fig)
#Word Clooud-1
#This word cloud shows the number of days on which a covid-19 case is detected in a county, not the number of cases.
df_WC_county = covidData.county
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                        background_color = 'white',
                        width = 512,
                        height = 384
                        ).generate(" ".join(df_WC_county))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')
plt.show
#Word Clooud-2
#This word cloud shows the number of days on which a covid-19 case is detected in a state, not the number of cases.
state = covidData.state
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                        background_color = 'white',
                        width = 512,
                        height = 384
                        ).generate(" ".join(state))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')
plt.show
#Box Plot

trace0 = go.Box(
    y=df_NYC.cases,
    name = 'covid-19 cases in NYC',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace1 = go.Box(
    y=df_LA.cases,
    name = 'covid-19 cases in LA',
    marker = dict(
        color = 'rgb(12, 128, 128)',
    )
)
data = [trace0, trace1]
iplot(data)
df_LA.head()
#Scatter Matrix Plots
import plotly.figure_factory as ff
df_SMP = df_LA.loc[:, ["cases", "deaths", "fips"]]
df_SMP["index"] = np.arange(1,len(df_SMP)+1)
fig = ff.create_scatterplotmatrix(df_SMP, diag = 'box', index = 'index', colormap = 'Portland',
                                 colormap_type = 'cat',
                                 height = 700, width = 700)
iplot(fig)
#3D Scatter Plot with Colorscaling
trace1 = go.Scatter3d(
    x = covidData.state,
    y = covidData.cases,
    z = covidData.deaths,
    mode = 'markers',
    marker = dict(
        size = 10,
        color='rgb(255,0,0)',
    )
)

data = [trace1]
layout = go.Layout(
    margin = dict(
        l = 0,
        r = 0,
        b = 0,
        t = 0
    )
)
fig = go.Figure(data = data, layout = layout)
iplot(fig)
#Multiple Subplots
trace1 = go.Scatter(
    x = df_NYC.timeline,
    y = df_NYC.cases,
    name = "NYC cases"
)
trace2 = go.Scatter(
    x = df_LA.timeline,
    y = df_LA.cases,
    xaxis='x3',
    yaxis='y3',
    name = "LA cases"
)
trace3 = go.Scatter(
    x = df_NYC.timeline,
    y = df_NYC.deaths,
    xaxis='x2',
    yaxis='y2',
    name = "NYC deaths"
)
trace4 = go.Scatter(
    x = df_LA.timeline,
    y = df_LA.deaths,
    xaxis='x4',
    yaxis='y4',
    name = "LA deaths"
)
data = [trace1, trace2, trace3, trace4]
layout = go.Layout(
    xaxis = dict(
        domain = [0, 0.45]
    ),
    yaxis = dict(
        domain = [0, 0.45]
    ),
    xaxis2 = dict(
        domain = [0.55, 1]
    ),
    xaxis3 = dict(
        domain = [0, 0.45],
        anchor = 'y3'
    ),
    xaxis4 = dict(
        domain = [0.55, 1],
        anchor = 'y4'
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
    title = 'Covid-19 cases and deaths VS timeline in NYC and LA'
)
fig = go.Figure (data=data, layout=layout)
iplot(fig)