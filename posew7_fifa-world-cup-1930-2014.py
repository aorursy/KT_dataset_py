# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
# import figure factory
import plotly.figure_factory as ff

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
matches  = pd.read_csv("../input/WorldCupMatches.csv")
players  = pd.read_csv("../input/WorldCupPlayers.csv")
cups     = pd.read_csv("../input/WorldCups.csv")
matches.info()
players.info()
cups.info()
matches.head()
players.head()
cups.head()
cups.corr()
plt.subplots(figsize=(17,9))
sns.barplot(x=cups.Winner.value_counts().index, y=cups.Winner.value_counts().values)
plt.title("")
plt.show()
a = cups.Winner
b = cups["Runners-Up"]
c = cups.Third
d = cups.Fourth
first_four = pd.concat([a,b,c,d],axis=0)
plt.subplots(figsize=(17,9))
sns.barplot(x=first_four.value_counts().values,y=first_four.value_counts().index)
plt.show()
trace1 = go.Scatter(
    x=cups.Year,
    y=cups.GoalsScored,
    name = "Goals Scored",
    mode = "lines",
    marker = dict(color="rgba(0,0,255,0.7)"),
)
trace2 = go.Scatter(
    x=cups.Year,
    y=cups.QualifiedTeams,
    name="Qualified Teams",
    mode="lines",
    marker=dict(color="rgba(0,255,0,0.7)")
)
trace3 = go.Scatter(
    x=cups.Year,
    y=cups.MatchesPlayed,
    name="Matches Played",
    mode="lines",
    marker=dict(color="rgba(255,0,0,0.7)")
)
data = [trace1,trace2,trace3]
layout = dict(xaxis=dict(title="Year"))
fig = go.Figure(data=data,layout=layout)
iplot(fig)
trace = go.Scatter3d(
    x = cups.GoalsScored,
    y = cups.QualifiedTeams,
    z = cups.MatchesPlayed,
    marker = dict(color="rgba(0,0,255,0.7)", size=10)
)
data = [trace]
layout = go.Layout(margin=dict(l=0,r=0,t=0,b=0))
fig = go.Figure(data=data, layout=layout)
iplot(fig)
trace1 = go.Bar(
    x = matches.Year,
    y = matches["Home Team Goals"],
    name = "Home Team Goals"
)
trace2 = go.Bar(
    x = matches.Year,
    y = matches["Away Team Goals"],
    name = "Away Team Goals"
)
data=[trace1, trace2]
layout = go.Layout(barmode="group")
fig = go.Figure(data=data, layout=layout)
iplot(fig)
trace1 = go.Box(
    y = matches["Home Team Goals"],
    name = "Home Team Goals"
)
trace2 = go.Box(
    y = matches["Away Team Goals"],
    name = "Away Team Goals"
)
data = [trace1, trace2]
layout=go.Layout()
fig = go.Figure(data=data,layout=layout)
iplot(fig)






















