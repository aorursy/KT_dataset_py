# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, plot
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from wordcloud import WordCloud 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
timesData = pd.read_csv("../input/world-university-rankings/timesData.csv")

timesData.info()
timesData.head(10)
df=timesData.iloc[:100,:]

trace1=go.Scatter(x=df.world_rank,y=df.citations,mode="lines",name="citation",marker=dict(color='rgba(188, 112, 2, 0.8)'), text=df.university_name)

trace2=go.Scatter(x=df.world_rank,y=df.teaching,mode="lines",name="teaching",marker=dict(color='rgba(80, 26, 80, 0.8)'), text=df.university_name)

data = [trace1, trace2]


layout = dict(title = 'Citation and Teaching vs World Rank of Top 100 Universities',
              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= True)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
df2014=timesData[timesData.year==2014].iloc[:100,:]
df2015=timesData[timesData.year==2015].iloc[:100,:]
df2016=timesData[timesData.year==2016].iloc[:100,:]
import plotly.graph_objs as go

trace1=go.Scatter(x=df2014.world_rank,y=df.citations,mode='markers',marker=dict(color='rgba(255,18,25,08)'),text=df2014.university_name)
trace2=go.Scatter(x=df2015.world_rank,y=df.citations,mode='markers',marker=dict(color='rgba(200,150,30,05)'),text=df2015.university_name)
trace3=go.Scatter(x=df2016.world_rank,y=df.citations,mode='markers',marker=dict(color='rgba(150,170,36,05)'),text=df2015.university_name)

data=[trace1,trace2,trace3]
layout=dict(title='citation vs world rank 100 university',xaxis=dict(title='world rank',ticklen=5,zeroline=False),yaxis=dict(title= 'Citation',ticklen= 5,zeroline= False))

          
fig = dict(data = data, layout = layout)
iplot(fig)

df2014=timesData[timesData.year==2014].iloc[:3,:]

import plotly.graph_objs as go


              
        # create trace1 
trace1 = go.Bar(
                x = df2014.university_name,
                y = df2014.citations,
                name = "citations",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df2014.country)
              

trace2 = go.Bar(
                x = df2014.university_name,
                y = df2014.teaching,
                name = "teaching",
                marker = dict(color = 'rgba(255, 255, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df2014.country)    
              
data=[trace1,trace2]
              
              
layout=go.Layout(barmode='group')
              
              
fig=go.Figure(data=data,layout=layout)
iplot(fig)
df2014=timesData[timesData.year==2014].iloc[:3,:]
import plotly.graph_objs as go

x=df2014.university_name

trace1 = {
        'x':x,
       'y':df2014.citations,
       'name':'citation',
       'type':'bar'}


trace2={'x':x,
       'y':df2014.teaching,
       'name':'teaching',
       'type':'bar'}

data=[trace1, trace2]

layout={'xaxis':{'title':'Top three university'},
       'barmode':'relative',
       'title':'citation and teaching of top 3 üniveristi'}

fig=go.Figure(data=data, layout=layout)

iplot(fig)
# data preparation
df2016 = timesData[timesData.year == 2016].iloc[:7,:]
pie1 = df2016.num_students
pie1_list = [float(each.replace(',', '.')) for each in df2016.num_students]  # str(2,4) => str(2.4) = > float(2.4) = 2.4
labels = df2016.university_name
# figure
fig = {
  "data": [
    {
      "values": pie1_list,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "Number Of Students Rates",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Universities Number of Students rates",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Number of Students",
                "x": 0.20,
                "y": 1
            },
        ]
    }
}
iplot(fig)
# data preparation
df2016 = timesData[timesData.year == 2016].iloc[:7,:]

pie1 = df2016.num_students
pie1_list = [float(each.replace(',', '.')) for each in df2016.num_students]  # str(2,4) => str(2.4) = > float(2.4) = 2.4
labels = df2016.university_name

fig = go.Figure(data=[go.Pie(labels=labels, values=pie1, hole=.3)])
fig.show()
df2015=timesData[timesData.year==2016].iloc[:20,:]
df2015

df2015=timesData[timesData.year==2016].iloc[:20,:]
num_students_size = [float(each.replace(',','.')) for each in  df2015.num_students]
internationa_color = [float(each) for each in df2015.international]

data=[
    {'y':df2015.teaching,
     'x':df2015.world_rank,
     'mode': 'markers',
     'marker':{
         'color':internationa_color,
         'size':num_students_size,
         'showscale':True
         
     },
        'text':df2015.university_name
    }
]
iplot(data)
staff2011= timesData.student_staff_ratio[timesData.year==2011]
staff2012= timesData.student_staff_ratio[timesData.year==2012]

import plotly.graph_objs as go

trace1=go.Histogram(
    x=staff2011,
    opacity=0.75,
    name='2011',
    marker=dict(color='rgba(150,255,10,0.8)'))

trace2=go.Histogram(
    x=staff2012,
    opacity=0.75,
    name='2012',
    marker=dict(color='rgba(0,200,200,0.6)'))

data=[trace1,trace2]


layout=go.Layout(barmode='overlay',
                 title='student_staff_ratio',
                 xaxis=dict(title='student_staff_ratio'),              
                 yaxis=dict(title='Count'))
                 
fig=go.Figure(data=data,layout=layout)                
iplot(fig)












# data prepararion
x2011 = timesData.country[timesData.year == 2011]
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='red',
                          width=512,
                          height=384
                         ).generate(" ".join(x2011))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()
# data preparation
x2015 = timesData[timesData.year == 2015]

trace0 = go.Box(
    y=x2015.total_score,
    name = 'total score of universities in 2015',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace1 = go.Box(
    y=x2015.research,
    name = 'research of universities in 2015',
    marker = dict(
        color = 'rgb(12, 128, 128)',
    )
)
data = [trace0, trace1]
iplot(data)
import numpy as np
import plotly.figure_factory as ff
dataframe=timesData[timesData.year==2011]

data2015=dataframe.loc[:,['research','international','total_score']]

data2015['index']=np.arange(1,len(data2015)+1)

fig=ff.create_scatterplotmatrix(data2015,diag='box',index='index',colormap='Portland',colormap_type='cat', height=800,width=800)

iplot(fig)

dataframe=timesData[timesData.year==2011]


trace1=go.Scatter(x=dataframe.world_rank,
                  y=dataframe.teaching,
                  name='teaching',
                  marker=dict(color='rgba(12,50,255,08)'),)

trace2=go.Scatter(x=dataframe.world_rank,
                 y=dataframe.income,
                 name='teaching',
                 xaxis='x2',
                 yaxis='y2',
                 marker=dict(color='rgba(50,50,50,08)'))


data=[trace1,trace2]

layout=go.Layout(xaxis2=dict(domain=[0.6,0.95],anchor='y2') ,yaxis2=dict(domain=[0.7,0.95],anchor='x2'), title='Income and Teaching vs World Rank of Universities')
fig=go.Figure(data=data,layout=layout)
iplot(fig)