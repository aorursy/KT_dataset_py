# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objs as go # plotly for plots
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/NBA_player_of_the_week.csv')
data.head()
data.tail()
data.info()
data.Date = pd.to_datetime(data.Date)
data.info()
data.Position = data.Position.str.replace('-','')
# Histogram of position
hist_data = [go.Histogram(x =data.Position)]
layout = go.Layout(
    title='Histogram of Positon',
    xaxis=dict(
        title='Position'
    ),
    yaxis=dict(
        title='Count'
    ),
)
fig = go.Figure(data=hist_data, layout=layout)
iplot(fig)
#scatter graph of age and season in league
trace = go.Scatter(x = data.Age,
                   y = data['Seasons in league'],
                   mode = 'markers',
                   text = data.Team)
layout = go.Layout(
    title='Age vs Season in league',
    xaxis=dict(
        title='Age'
    ),
    yaxis=dict(
        title='Seasons in league'
    ),
)
scatter_data = [trace]
fig = go.Figure(data=scatter_data, layout=layout)
iplot(fig)
#pie chart of the conferences.
fig = {
  "data": [
    {
      "values": data.Conference.value_counts().tolist(),
      "labels": data.Conference.value_counts().keys().tolist(),
      "domain": {"x": [0, .5]},
      "hoverinfo":"label+percent+value",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Percentage of Conferences"
            },
    }
iplot(fig)
# bubble plot of age of player of the week by time. Size of the bubble is weight of the player
# color of the bubble is season
bubble_data = [
    {
        'y': data.Age,
        'x': data.Date,
        'mode': 'markers',
        'marker': {
            'color': data['Season short'],
            'size': data.Weight/10,
            'showscale': True
        },
        "text" :  data.Player    
    }
]
iplot(bubble_data)
#box plot of weight by position

x_data = data.Position.unique()
y_data = []
text_data = []
for each in x_data:
    y_data.append(data[data.Position == each].Weight)
    text_data.append(data[data.Position == each].Player)
    
x_data = np.append(x_data,'total')

y_data.append(data.Weight)
text_data.append(data.Player)
#y_data = y_data.tolist()

traces = []

for xd, yd, td in zip(x_data, y_data, text_data):
        traces.append(go.Box(
            y=yd,
            name=xd,
            boxpoints='all',
            jitter=0.5,
            text = td,
            whiskerwidth=0.2,
            marker=dict(
                size=2,
            ),
            line=dict(width=1),
        ))

layout = go.Layout(
    title='Weight of player of the week by position',
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
        dtick=5,
        gridcolor='rgb(255, 255, 255)',
        gridwidth=1,
        zerolinecolor='rgb(255, 255, 255)',
        zerolinewidth=2,
    ),
    margin=dict(
        l=40,
        r=30,
        b=80,
        t=100,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
    showlegend=False
)

fig = go.Figure(data=traces, layout=layout)
iplot(fig)
