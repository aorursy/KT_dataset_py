# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
df_players = pd.read_csv('../input/Players.csv')
df_stats = pd.read_csv('../input/Seasons_Stats.csv')
df_playerdata = pd.read_csv('../input/player_data.csv')
df_inter = df_stats.merge(df_players,how = 'left', on = 'Player')
df_inter.head()
df_inter['Age']= df_inter['Year']- df_inter['born']
df_inter.head()
df_inter.columns.unique()
from IPython.core.display import display, HTML, Javascript
from string import Template
import json
import IPython.display
PlayerPoints = df_inter[['Player','PTS']].groupby('Player').sum().sort_values('PTS', ascending = False).head(20)
trace1 = go.Bar(
    x = PlayerPoints.index.tolist(),
    y = PlayerPoints["PTS"].tolist(),
    name='Career Points',
    marker=dict(
        color='rgba(55, 128, 191, 0.7)',
        line=dict(
            color='rgba(55, 128, 191, 1.0)',
            width=2,
        )
    )
)

layout = go.Layout(
    barmode='stack',
    title = 'NBA Highest Scorers',
    titlefont=dict(size=25),
    width=850,
    height=500,
    paper_bgcolor='rgb(244, 238, 225)',
    plot_bgcolor='rgb(244, 238, 225)',
    yaxis = dict(
        title= 'Career Points',
        anchor = 'x',
        rangemode='tozero'
    ),
    xaxis = dict(title= 'Player Names'),
    yaxis2=dict(
        title='Value [M€]',
        titlefont=dict(
            color='rgb(148, 103, 189)'
        ),
        tickfont=dict(
            color='rgb(148, 103, 189)'
        ),
        overlaying='y',
        side='right',
        anchor = 'x',
        rangemode = 'tozero',
        dtick = 20
    ),
    legend=dict(x=0.05, y=0.05)
)

fig = go.Figure(data= [trace1], layout=layout)
py.iplot(fig)
df_2017 = df_inter[df_inter['Year'] == 2017]
df_2017.head()
data = df_2017[['Age',  '3PAr',
        'BPM', 'FG', 'FGA',
       '2P', 
       'AST', 'STL', 'BLK', 'PTS', 'height',
       'weight']]

data.plot(kind='density',layout = (3,4), subplots=True, sharex=False)
plt.show()
names = ['Age',  'PTS',  'height', 'weight', 'FG','AST%', 'STL%',
       'BLK%']

data2 = df_2017[names]

correlations = data2.corr()
# plot correlation matrix

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,8,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()