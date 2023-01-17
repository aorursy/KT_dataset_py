# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from wordcloud import WordCloud
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
timesData=pd.read_csv("../input/timesData.csv")
timesData.head()

timesData.info()

df=pd.DataFrame(timesData)
df=timesData.iloc[:100,:]
import plotly.graph_objs as go

trace1=go.Scatter(
    x=df.world_rank,
    y=df.total_score,
    name = "citations",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
    text= df.university_name)
    
trace2=go.Scatter(
    x = df.world_rank,
    y = df.teaching,
    mode = "lines+markers",
    name = "teaching",
    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
    text= df.university_name
)

data=[trace1,trace2]
layout = dict(title = 'Citation and Teaching vs World Rank of Top 100 Universities',
              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
    
    
    
df14 = timesData[timesData.year == 2014].iloc[:100,:]
df15 = timesData[timesData.year == 2015].iloc[:100,:]
df16 = timesData[timesData.year == 2016].iloc[:100,:]

trace1 =go.Scatter(
                    x = df14.world_rank,
                    y = df14.total_score ,
                    mode = "markers",
                    name = "2014",
                    marker = dict(color = 'rgba(0, 102, 0,0.8)'),
                    text= df14.university_name)
trace2 =go.Scatter(
                    x = df15.world_rank,
                    y = df15.total_score ,
                    mode = "markers",
                    name = "2015",
                    marker = dict(color = 'rgba(230, 92, 0,0.8)'),
                    text= df15.university_name)

trace3 =go.Scatter(
                    x = df16.world_rank,
                    y = df16.total_score ,
                    mode = "markers",
                    name = "2016",
                    marker = dict(color = 'rgba(51, 51, 153,0.8)'),
                    text= df16.university_name)
data = [trace1, trace2, trace3]
layout = dict(title = 'Citation vs world rank of top 100 universities with 2014, 2015 and 2016 years',
              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Citation',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)

df14 = timesData[timesData.year == 2014].iloc[:100,:]
df15 = timesData[timesData.year == 2015].iloc[:100,:]
df16 = timesData[timesData.year == 2016].iloc[:100,:]

trace1 =go.Scatter(
                    x = df14.world_rank,
                    y = df14.citations,
                    mode = "markers",
                    name = "2014",
                    marker = dict(color = 'rgba(0, 102, 0,0.8)'),
                    text= df14.university_name)
trace2 =go.Scatter(
                    x = df15.world_rank,
                    y = df15.citations,
                    mode = "markers",
                    name = "2015",
                    marker = dict(color = 'rgba(230, 92, 0,0.8)'),
                    text= df15.university_name)

trace3 =go.Scatter(
                    x = df16.world_rank,
                    y = df16.citations,
                    mode = "markers",
                    name = "2016",
                    marker = dict(color = 'rgba(51, 51, 153,0.8)'),
                    text= df16.university_name)
data = [trace1, trace2, trace3]
layout = dict(title = 'Citation vs world rank of top 100 universities with 2014, 2015 and 2016 years',
              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Citation',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
df16 = timesData[timesData.year == 2016].iloc[:7,:]
pie1=df16.num_students
pie_list=[float(each.replace(',','.')) for each in df16.num_students]

labels = df16.university_name
fig = {
  "data": [
    {
      "values": pie_list,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "Number Of Students Rates",
      "hoverinfo":"label+percent+name",
      "hole": .4,
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



df16 = timesData[timesData.year == 2016].iloc[:20,:]
num_students_size  = [float(each.replace(',', '.')) for each in df16.num_students]
international_color = [float(each) for each in df16.international]
data = [
    {
        'y': df16.teaching,
        'x': df16.world_rank,
        'mode': 'markers',
        'marker': {
            'color': international_color,
            'size': num_students_size,
            'showscale': True
        },
        "text" :  df16.university_name    
    }
]
iplot(data)
x2011 = timesData.student_staff_ratio[timesData.year == 2011]
x2012 = timesData.student_staff_ratio[timesData.year == 2012]

trace1 = go.Histogram(
    x=x2011,
    opacity=0.75,
    name = "2011",
    marker=dict(color='rgba(171, 50, 96, 0.6)'))
trace2 = go.Histogram(
    x=x2012,
    opacity=0.75,
    name = "2012",
    marker=dict(color='rgba(12, 50, 196, 0.6)'))

data = [trace1, trace2]
layout = go.Layout(barmode='overlay',
                   title=' students-staff ratio in 2011 and 2012',
                   xaxis=dict(title='students-staff ratio'),
                   yaxis=dict( title='Count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
x2011 = timesData.country[timesData.year == 2011]
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
import plotly.figure_factory as ff

dataframe = timesData[timesData.year == 2015]
data2015 = dataframe.loc[:,["research","international", "total_score"]]
data2015["index"] = np.arange(1,len(data2015)+1)
# scatter matrix
fig = ff.create_scatterplotmatrix(data2015, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',
                                  height=700, width=700)
iplot(fig)
trace1 = go.Scatter(
    x=dataframe.world_rank,
    y=dataframe.research,
    name = "research"
)
trace2 = go.Scatter(
    x=dataframe.world_rank,
    y=dataframe.citations,
    xaxis='x2',
    yaxis='y2',
    name = "citations"
)
trace3 = go.Scatter(
    x=dataframe.world_rank,
    y=dataframe.income,
    xaxis='x3',
    yaxis='y3',
    name = "income"
)
trace4 = go.Scatter(
    x=dataframe.world_rank,
    y=dataframe.total_score,
    xaxis='x4',
    yaxis='y4',
    name = "total_score"
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
    title = 'Research, citation, income and total score VS World Rank of Universities'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)