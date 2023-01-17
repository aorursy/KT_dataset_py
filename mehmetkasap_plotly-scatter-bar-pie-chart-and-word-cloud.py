# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# plotly library
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# word cloud library
from wordcloud import WordCloud

# matplotlib library
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
timesData = pd.read_csv('../input/timesData.csv')
timesData.columns
# data frames
df2014 = timesData [timesData.year == 2014].iloc[:100,:]
df2015 = timesData [timesData.year == 2015].iloc[:100,:]
df2016 = timesData [timesData.year == 2016].iloc[:100,:]
df2014.head()
# trace 1
trace1 = go.Scatter (x = df2014.world_rank,
                     y = df2014.citations,
                     mode = 'markers',
                     name = '2014',
                     marker = dict(color = 'rgba(255,128,255,0.8)'),
                     text = df2014.university_name) 
# trace 2
trace2 = go.Scatter (x = df2015.world_rank,
                     y = df2015.citations,
                     mode = 'markers',
                     name = '2015',
                     marker = dict(color = 'rgba(255,128,2,0.8)'),
                     text = df2015.university_name) 
# trace 3
trace3 = go.Scatter (x = df2016.world_rank,
                     y = df2016.citations,
                     mode = 'markers',
                     name = '2016',
                     marker = dict(color = 'rgba(0,255,200,0.8)'),
                     text = df2016.university_name) 
data = [trace1, trace2, trace3]
layout = dict(title = 'citation vs world rank of top 100 universities with 2014 2015 2016',
              xaxis= dict(title = 'world rank', ticklen=5, zeroline=False),
              yaxis= dict(title = 'citation', ticklen=5, zeroline = False))
fig = dict(data = data, layout = layout)
iplot(fig)
plt.savefig('plotly-scatter.png')
# data frame for bar plot
df2014 = timesData [timesData.year == 2014].iloc[:3,:]
df2014.columns
trace1 = go.Bar(x = df2014.university_name,
                y = df2014.citations,
                name = 'citations',
                marker = dict(color = 'rgba(255,174,255,0.5)', 
                              line = dict(color='rgb(0,0,0)',width=1.5)),
                text = df2014.country)
trace2 = go.Bar(x = df2014.university_name,
                y = df2014.teaching,
                name = 'teaching',
                marker = dict(color = 'rgba(255,255,128,0.5)', 
                              line = dict(color='rgb(0,0,0)',width=1.5)),
                text = df2014.country)
data = [trace1, trace2]
layout = go.Layout (barmode = 'group')
fig = go.Figure (data=data, layout=layout)
iplot(fig)
plt.savefig('plotly-bar.png')
df2014 = timesData [timesData.year == 2014].iloc[:3,:]
x = df2014.university_name

trace1 = {'x': x,
          'y': df2014.citations,
          'name': 'citations',
          'type': 'bar'};
trace2 = {'x':x,
          'y': df2014.teaching,
          'name': 'teaching',
          'type': 'bar'};
data = [trace1, trace2]

layout = {'xaxis': {'title': 'top 3 universities'},
          'barmode': 'relative',
          'title': 'citation and teaching of top 3 universities in 2014'};
fig = go.Figure(data=data, layout=layout)
iplot(fig)
plt.savefig('plotly-bar-type2.png')
          
df2016 = timesData [timesData.year == 2016].iloc[:7,:]
df2016.info()
pie1 = df2016.num_students
# below we will change strings that contain ',' to string thats contain '.' and 
# then convert them into float.
pie1_list = [float(each.replace(',', '.')) for each in df2016.num_students]
labels = df2016.university_name
# figure
fig = {
    'data': [
      {
          'values': pie1_list,
          'labels': labels,
          'domain': {'x': [0, 0.5]},
          'name': 'number of students rate',
          'hoverinfo': 'label+percent+name',
          'hole': 0.3,
          'type': 'pie'
      },],
    'layout': {
        'title': 'universities number of students rates',
        'annotations': [
            {'font': {'size':20},
             'showarrow': False,
             'text': 'number of students',
              'x': 0.20,
              'y': 1 },
        ]
    }
}
iplot(fig)
df2016 = timesData [timesData.year == 2016].iloc[:20,:]
num_student_size = [float(each.replace(',','.')) for each in df2016.num_students]

# here df2016.international is an object=> convert into float
df2016.international = df2016.international.astype(float, inplace=True)
international_color = df2016.international
df2016.info()
data = [
    {       
        'y': df2016.teaching,
        'x': df2016.world_rank,
        'mode': 'markers', 
        'marker': {
            'color': international_color,
            'size': num_student_size,
            'showscale': True
        },
        'text': df2016.university_name
    } 
]
iplot(data)
plt.savefig('plotly-bubble.png')
timesData.columns
# data to be used
x2011 = timesData.student_staff_ratio[timesData.year == 2011]
x2012 = timesData.student_staff_ratio[timesData.year == 2012]

# trace 1 and 2
trace1 = go.Histogram(
    x=x2011,
    opacity = 0.75,
    name = '2011',
    marker = dict(color = 'rgba(171, 50, 96, 0.6)'))

trace2 = go.Histogram(
    x=x2012,
    opacity = 0.75,
    name = '2012',
    marker = dict(color = 'rgba(12, 50, 196, 0.6)'))

data = [trace1, trace2]

layout = go.Layout(barmode='overlay',
                   title=' students-staff ratio in 2011 and 2012',
                   xaxis=dict(title='students-staff ratio'),
                   yaxis=dict( title='Count'))
fig = go.Figure(data=data, layout=layout)
iplot(fig)
plt.savefig('plotly-histogram.png')

timesData.columns
x2011 = timesData.country[timesData.year == 2011]

plt.subplots(figsize = (8,8))

wordcloud = WordCloud (
                    background_color = 'white',
                    width = 512,
                    height = 384
                        ).generate(' '.join(x2011))
plt.imshow(wordcloud) # image show
plt.axis('off') # to off the axis of x and y
plt.savefig('Plotly-World_Cloud.png')
plt.show()