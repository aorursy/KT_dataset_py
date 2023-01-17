# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Year_wise_batting_stats_all_ODI_players(30-05-19).csv")

data = data.fillna(0)

data.head()
data.dtypes
data['HS'] = data['HS'].map(lambda x: int(str(x).rstrip('*')))
data.dtypes
op = data.loc[data['Year'] == 'Overall']

op = op.drop(columns = "Year")
op["Outs"] = op["Inns"]-op["NO"]
op["O/I"] = op["Outs"]/op["Inns"]
op.head()
hard_hitters = op.loc[(op["Inns"]>30) & (op["S/R"]>80)].sort_values(by='S/R', ascending=False)

hard_hitters.head(10)
trace1 =go.Scatter(

                    x = hard_hitters["Inns"],

                    y = hard_hitters["S/R"],

                    mode = "markers",

                    name = "Name",

                    marker = dict(color=(

            (hard_hitters["S/R"] > 110)            

        ).astype('int'),

        colorscale=[[0, 'yellow'], [1, 'red']]),

                    text= hard_hitters["player_name"])



data = [trace1]

layout = dict(title = 'The Hard Hitters of ODI Cricket',

              xaxis= dict(title= 'Number of Matches',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Strike Rate',ticklen= 5,zeroline= False),

              hovermode= 'closest',

             )

fig = dict(data = data, layout = layout)

iplot(fig)
consistent = op.loc[op["Inns"]>30].sort_values(by='Avg', ascending=False)

consistent.head(10)
trace1 =go.Scatter(

                    x = consistent["Runs"],

                    y = consistent["Avg"],

                    mode = "markers",

                    name = "Name",

                    marker = dict(color=(

            ((consistent["Avg"]-(consistent["Runs"]/1000)*1.641-37.5)>0)

        ).astype('int'),

        colorscale=[[0, 'yellow'], [1, 'red']]),

                    text= consistent["player_name"])



data = [trace1]

layout = dict(title = 'The Consistent Runmakers of ODI Cricket',

              xaxis= dict(title= 'Runs Scored',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Average',ticklen= 5,zeroline= False),

              hovermode= 'closest',

             )

fig = dict(data = data, layout = layout)

iplot(fig)
h100 = op.sort_values(by='100s',ascending=False).head(30)

h100.head(10)
h50 = op.sort_values(by='50s',ascending=False).head(30)

h50.head(10)
trace2 = go.Bar(x = h100["player_name"],

                y = h100["100s"],

                name = "ODI Centuries",

                marker = dict(color = 'red',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = h100.country)

trace1 = go.Bar(

                x = h100["player_name"],

                y = h100["50s"],

                name = "ODI 50s",

                marker = dict(color = 'blue',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = h50.country)

data = [trace1, trace2]

layout = go.Layout(barmode = "group",hovermode= 'closest', title = "Top 30 ODI Century scorers", yaxis= dict(title= 'Num 50 : Num 100',ticklen= 5,zeroline= False))

fig = go.Figure(data = data, layout = layout)

iplot(fig)
trace2 = go.Bar(x = h50["player_name"],

                y = h50["100s"],

                name = "ODI Centuries",

                marker = dict(color = 'red',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = h100.country)

trace1 = go.Bar(

                x = h50["player_name"],

                y = h50["50s"],

                name = "ODI 50s",

                marker = dict(color = 'blue',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = h50.country)

data = [trace1, trace2]

layout = go.Layout(barmode = "group",hovermode= 'closest', title = "Top 30 ODI Half-Century scorers", yaxis= dict(title= 'Num 50 : Num 100',ticklen= 5,zeroline= False))

fig = go.Figure(data = data, layout = layout)

iplot(fig)
ducks = op.loc[(op["0s"]>10)].sort_values(by='0s',ascending=False)

ducks.head(10)
trace1 =go.Scatter(

                    x = ducks["Inns"],

                    y = ducks["0s"],

                    mode = "markers",

                    name = "Name",

                    marker = dict(color=(

            (((ducks["0s"]/ducks["Inns"])-10/100)>0)

        ).astype('int'),

        colorscale=[[0, 'red'], [1, 'yellow']]),

                    text= ducks["player_name"])



data = [trace1]

layout = dict(title = 'Getting Out at Nought',

              xaxis= dict(title= 'Number of Innings Played',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Ducks',ticklen= 5,zeroline= False),

              hovermode= 'closest',

             )

fig = dict(data = data, layout = layout)

iplot(fig)
stay_play = op.loc[op["Inns"]>100].sort_values(by='O/I',ascending=True)

stay_play.head(10)
trace1 =go.Scatter(

                    x = stay_play["Runs"],

                    y = stay_play["O/I"],

                    mode = "markers",

                    name = "Name",

                    marker = dict(color=(

            ((stay_play["O/I"]<0.75)&(stay_play["Runs"]>5000))

        ).astype('int'),

        colorscale=[[0, 'yellow'], [1, 'red']]),

                    text= stay_play["player_name"])



data = [trace1]

layout = dict(title = 'Batsmen who try and stay till the end',

              xaxis= dict(title= 'Runs',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Outs/Innings',ticklen= 5,zeroline= False),

              hovermode= 'closest',

             )

fig = dict(data = data, layout = layout)

iplot(fig)
runs = op.loc[op["Runs"]>1000]
trace1 =go.Scatter(

                    x = runs["Runs"],

                    y = runs["Inns"],

                    mode = "markers",

                    name = "Name",

                    marker = dict(color=(

            (((runs["Inns"]*1000)/(runs["Runs"]))<27)

        ).astype('int'),

        colorscale=[[0, 'yellow'], [1, 'red']]),

                    text= runs["player_name"])



data = [trace1]

layout = dict(title = 'The Great Run Scorers of ODI Cricket',

              xaxis= dict(title= 'Runs',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Innings',ticklen= 5,zeroline= False),

              hovermode= 'closest',

             )

fig = dict(data = data, layout = layout)

iplot(fig)