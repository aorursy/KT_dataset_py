
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns


# plotly
import plotly.plotly as py
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
# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/gun-violence-data_01-2013_03-2018.csv')
data.info()
x=data.date
x.str.partition('-')
y=x.str.split('-')
year=[row[0] for row in y]
data['year'] = year
data1=data.iloc[:278,:]
# import graph objects as "go"
import plotly.graph_objs as go
df=data1
# Creating trace1
trace1 = go.Scatter(
                    x = df.date,
                    y = df.n_killed,
                    mode = "lines",
                    name = "n_killed",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text= df.state)
# Creating trace2
trace2 = go.Scatter(
                    x = df.date,
                    y = df.n_injured,
                    mode = "lines+markers",
                    name = "n_injured",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= df.city_or_county)
data = [trace1, trace2]
layout = dict(title = 'Killed and İnjuired People in 2013',
              xaxis= dict(title= 'Date',ticklen= 4,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
# prepare data frames

# import graph objects as "go"
import plotly.graph_objs as go
# creating trace1
trace1 =go.Scatter(
                    x = df.date,
                    y = df.n_killed,
                    mode = "markers",
                    name = "n_killed",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text= df.state)
# creating trace2
trace2 =go.Scatter(
                    x = df.date,
                    y = df.n_injured,
                    mode = "markers",
                    name = "n_injured",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text= df.city_or_county)


data = [trace1, trace2]
layout = dict(title = 'Killed and İnjuired People in 2013',
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Number Of People',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
# import graph objects as "go"
import plotly.graph_objs as go
# create trace1 
trace1 = go.Bar(
                x = df.date,
                y = df.n_killed,
                name = "n_killed",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df.state)
# create trace2 
trace2 = go.Bar(
                x = df.date,
                y = df.n_injured,
                name = "n_injured",
                marker = dict(color = 'rgba(255, 255, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df.city_or_county)
data = [trace1, trace2]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)
# data prepararion
x2011 = data1.state
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(x2011))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()
# import figure factory
import plotly.figure_factory as ff
# prepare data
dataframe = data1
data2015 = dataframe.loc[:,["n_killed","n_injured", "date"]]
data2015["index"] = np.arange(1,len(data2015)+1)
# scatter matrix
fig = ff.create_scatterplotmatrix(data2015, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',
                                  height=700, width=700)
iplot(fig)
# create trace 1 that is 3d scatter
trace1 = go.Scatter3d(
    x=data1.date,
    y=data1.n_killed,
    z=data1.n_injured,
    mode='markers',
    marker=dict(
        size=10,
        color='rgb(255,0,0)',                # set color to an array/list of desired values      
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
    x=data1.date,
    y=data1.n_killed,
    name = "killed"
)
trace2 = go.Scatter(
    x=data1.date,
    y=data1.n_injured,
    xaxis='x2',
    yaxis='y2',
    name = "injuired"
)
trace3 = go.Scatter(
    x=data1.date,
    y=data1.city_or_county,
    xaxis='x3',
    yaxis='y3',
    name = "city or county"
)
trace4 = go.Scatter(
    x=data1.date,
    y=data1.state,
    xaxis='x4',
    yaxis='y4',
    name = "state"
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
    title = 'Killed And Injuired People in 2013 Acording to date And Location'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)