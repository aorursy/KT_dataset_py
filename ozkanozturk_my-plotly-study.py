# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import chart_studio.plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)



from wordcloud import WordCloud



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# read data

timesData = pd.read_csv("/kaggle/input/world-university-rankings/timesData.csv")
timesData.info()
timesData.head()
# line Charts :  Citation and teaching vs Wolrd Rank of 100 universities



df = timesData.iloc[:100, :]    # first 100 rows of data



trace1 = go.Scatter(

x = df.world_rank,

y = df.citations,

mode = "lines",

name = "citations",

marker = dict(color = "rgba(16,112,2,0.8)"),

text = df.university_name)



trace2 = go.Scatter(

x=df.world_rank,

y=df.teaching,

mode="lines+markers",

name="teaching",

marker=dict(color="rgba(80,26,80,0.8)"),

text = df.university_name)



data = [trace1, trace2]

layout=dict(title = "Citation and teaching vs Wolrd Rank of 100 universities",

           xaxis= dict(title="World Rank", ticklen=5, zeroline=False ))



fig = dict (data = data, layout = layout)

iplot(fig)

# scatter plot:  Citation vs Wolrd Rank of 100 universities on 2014,2015,2016



df2014 = timesData[timesData.year == 2014].iloc[:100, :]

df2015 = timesData[timesData.year == 2015].iloc[:100, :]

df2016 = timesData[timesData.year == 2016].iloc[:100, :]



trace1 = go.Scatter(

x = df2014.world_rank,

y = df2014.citations,

mode = "markers",

name = "2014",

marker = dict(color = "rgba(255,128,255,0.8)"),

text = df2014.university_name)



trace2 = go.Scatter(

x = df2015.world_rank,

y = df2015.citations,

mode = "markers",

name = "2015",

marker = dict(color = "rgba(255,128,2,0.8)"),

text = df2015.university_name)



trace3 = go.Scatter(

x = df2016.world_rank,

y = df2016.citations,

mode = "markers",

name = "2016",

marker = dict(color = "rgba(0,255,200,0.8)"),

text = df2016.university_name)



data = [trace1, trace2, trace3]

layout=dict(title = "Citation vs Wolrd Rank of 100 universities on 2014,2015,2016",

           xaxis= dict(title="World Rank", ticklen=5, zeroline=False ),

           yaxis= dict(title="Citations", ticklen=5, zeroline=False ))



fig = dict (data = data, layout = layout)

iplot(fig)



timesData.head()
# barchart1: Citations and Teaching of top 3 universities in 2014



df2014 = timesData[timesData.year== 2014].iloc[:3,:]



trace1 = go.Bar(

x = df2014.university_name,

y = df2014.citations,

name = "citations",

marker = dict(color = "rgba(255,174,255,0.8)",

             line=dict(color = "rgba(0,0,0)", width = 1.5)),

text = df2014.country)





trace2 = go.Bar(

x = df2014.university_name,

y = df2014.teaching,

name = "teaching",

marker = dict(color = "rgba(255,255,128,0.8)",

             line=dict(color = "rgba(0,0,0)", width = 1.5)),

text = df2014.country)



data = [trace1, trace2]

layout=go.Layout(barmode="group")

fig = go.Figure(data=data, layout=layout)

iplot(fig)

# Students rate of top 7 universities in 2016



df2016 = timesData[timesData.year== 2016].iloc[0:7, :]

pie1 = [float(each.replace(",",".")) for each in df2016.num_students]

labels = df2016.university_name



# plot

fig = {

    "data": [

        {

            "values":pie1,

            "labels":labels,

            "domain":{"x":[0,.5]},

            "name":"Student rates",

            "hoverinfo":"label+percent+name",

            "hole":.3,

            "type":"pie"

        },],

    "layout":{

        "title":"Student rates of top 7 universities in 2016",

        "annotations":[

            {"font":{"size":20},

            "showarrow":False,

            "text":"Student rates",

            "x":0.2,

             "y":1

            },

        ]

    }

}



iplot(fig)