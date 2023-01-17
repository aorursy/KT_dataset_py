# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import plotly.plotly as py

from plotly.offline import init_notebook_mode,iplot 

init_notebook_mode(connected=True)

import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from wordcloud import WordCloud



import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
timesData=pd.read_csv("../input/timesData.csv")
timesData.info()
timesData.head(10)
df=timesData.iloc[:100,:]

import plotly.graph_objs as go



trace1=go.Scatter(

                    x=df.world_rank,

                    y=df.citations,

                    mode="lines",

                    name="citations",

                    marker=dict(color='rgba(6,112,2,0.8)'),

                    text=df.university_name

                )



trace2=go.Scatter(

                    x = df.world_rank,

                    y = df.teaching,

                    mode = "lines+markers",

                    name = "teaching",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                    text= df.university_name

)

data = [trace1, trace2]

layout = dict(title = 'Citation and Teaching vs World Rank of Top 100 Universities',

              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
df2014 = timesData[timesData.year == 2014].iloc[:100,:]

df2015 = timesData[timesData.year == 2015].iloc[:100,:]

df2016 = timesData[timesData.year == 2016].iloc[:100,:]

# import graph objects as "go"

import plotly.graph_objs as go

# creating trace1

trace1 =go.Scatter(

                    x = df2014.world_rank,

                    y = df2014.citations,

                    mode = "markers",

                    name = "2014",

                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),

                    text= df2014.university_name)

# creating trace2

trace2 =go.Scatter(

                    x = df2015.world_rank,

                    y = df2015.citations,

                    mode = "markers",

                    name = "2015",

                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),

                    text= df2015.university_name)

# creating trace3

trace3 =go.Scatter(

                    x = df2016.world_rank,

                    y = df2016.citations,

                    mode = "markers",

                    name = "2016",

                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),

                    text= df2016.university_name)

data = [trace1, trace2, trace3]

layout = dict(title = 'Citation vs world rank of top 100 universities with 2014, 2015 and 2016 years',

              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Citation',ticklen= 5,zeroline= False)

             )

fig = go.Figure(data = data, layout = layout)

iplot(fig)
df2014 = timesData[timesData.year == 2014].iloc[:3,:]

# import graph objects as "go"

import plotly.graph_objs as go



x = df2014.university_name



trace1 = {

  'x': x,

  'y': df2014.citations,

  'name': 'citation',

  'type': 'bar'

};

trace2 = {

  'x': x,

  'y': df2014.teaching,

  'name': 'teaching',

  'type': 'bar'

};

data = [trace1, trace2];

layout = {

  'xaxis': {'title': 'Top 3 universities'},

  'barmode': 'relative',

  'title': 'citations and teaching of top 3 universities in 2014'

};

fig = go.Figure(data = data, layout = layout)

iplot(fig)