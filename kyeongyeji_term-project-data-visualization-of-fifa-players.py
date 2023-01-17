# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rd

import seaborn as sns



from plotly import tools

import plotly.figure_factory as ff

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

import plotly.figure_factory as ff

init_notebook_mode(connected = True)

import plotly.graph_objs as go



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/data.csv")

new_data = data.copy()

data.head(10) # To show TOP 10 Players in data set
data.columns # To show what columns are in data set
age = data.Age.value_counts(sort = False).reset_index()

age.sort_values('index', inplace = True)



x = age.iloc[:,0]

y = age.iloc[:,1]



x = x.tolist()

y = y.tolist()
# Histogram for Age of Players



trace1 = go.Bar(x = x, y = y, name = 'Age', opacity = 0.7, marker = dict(color = 'rgb(6, 80, 100)'))

data_a = [trace1]



layout = go.Layout(barmode = "group", title = "Age of Players")

fig = go.Figure(data = data_a, layout = layout)

iplot(fig)
# To show each player's stat

# When you want to see the other player's stat, you can set the player's name to the name value of x.



x = data[data["Name"] == "L. Messi"]



data_b = [go.Scatterpolar(

      r = [x['Crossing'].values[0],x['Finishing'].values[0],x['Dribbling'].values[0],x['ShortPassing'].values[0],x['LongPassing'].values[0],x['BallControl'].values[0]],

      theta = ['Crossing', 'Finishing', 'Dribbling', 'ShortPassing', 'LongPassing', 'BallControl'],

      fill = 'toself',

      name = x["Name"].values[0]

    )]



layout = go.Layout( polar = dict(radialaxis = dict(visible = True)), showlegend = True, title = "Stat of Player")



fig = go.Figure(data = data_b, layout = layout)

iplot(fig, filename = "Player stats")
# Comparison of Crossing and Shortpassing



new_index = (new_data["Overall"].sort_values(ascending = False)).index.values

sortedData = new_data.reindex(new_index)

best_players = sortedData.head(100)



trace1 = go.Scatter(

                    x = best_players.Name,

                    y = best_players.ShortPassing,

                    mode = "lines",

                    name = "ShortPassing",

                    marker = dict(color = "rgb(6, 80, 100)"),

                    text = best_players.Nationality)

trace2 = go.Scatter(

                    x = best_players.Name,

                    y = best_players.Crossing,

                    mode = "lines",

                    name = "Crossing",

                    marker = dict(color = "rgb(109, 99, 109)"),

                    text = best_players.Nationality)

data = [trace1, trace2]

layout = dict(title = "Comparison between Crossing and ShortPassing" , xaxis = dict(ticklen = 5,zeroline = False) )



fig = dict(data = data, layout = layout)

iplot(fig)
top_100 = new_data.head(100)

nat = top_100.Nationality.value_counts(sort = False).reset_index()

nat.sort_values('index', inplace = True)



x = nat.iloc[:,0]

y = nat.iloc[:,1]



x = x.tolist()

y = y.tolist()
# Histogram for Nationality with TOP 100 players



trace1 = go.Bar(x = x, y = y, name = 'Nationality', opacity = 0.7, marker = dict(color = 'rgb(100, 80, 100)'))

data_a = [trace1]



layout = go.Layout(barmode = "group", title = "Nationality of TOP 100 Players")

fig = go.Figure(data = data_a, layout = layout)

iplot(fig)