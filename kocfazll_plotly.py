# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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

# Any results you write to the current directory are saved as output.a
timesData=pd.read_csv("../input/timesData.csv")
timesData.info()
timesData.head(10)

df=timesData.iloc[0:100,:]
import plotly.graph_objs as go
trace1=go.Scatter(
                    x=df.world_rank,
                    y=df.total_score,
                    mode="lines",
                    name="world rank",
                    marker={"color":"rgba(178,34,34,0.8)"},
                    text=df.university_name)
trace2=go.Scatter(
                    x=df.world_rank,
                    y=df.research,
                    mode= "lines+markers",
                    name= "research",
                    marker= dict(color="rgba(0,191,255,0.8)"),
                    text= df.university_name )

data = [trace1, trace2]
layout = dict(title = 'Citation and Teaching vs World Rank of Top 100 Universities',
              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
df2014=timesData[timesData.year==2014].iloc[:100,:]
df2015=timesData[timesData.year==2015].iloc[:100,:]
df2016=timesData[timesData.year==2016].iloc[:100,:]

import plotly.graph_objs as go

trace1=go.Scatter(
                    x=df2014.world_rank,
                    y=df2014.research,
                    mode="markers",name="2014"
                    ,marker=dict(color="rgba(139,125,107,0.8)"),
                    text= df2014.university_name)


trace2=go.Scatter(x=df2015.world_rank,
                  y=df2015.research,
                  name="2015",
                  marker=dict(color="rgba(73.54,96,1)"),
                  mode="markers",
                  text=df2015.university_name)

trace3=go.Scatter(x=df2016.world_rank,
                  y=df2016.research,
                  name="2016",
                  marker=dict(color="rgba(1.23,10,1)"),
                  mode="markers",
                  text=df2016.university_name)

data=[trace1,trace2,trace3]

x=dict(title="Scatter Plotly",
            xaxis=dict(title="World Rank",ticklen=5,zeroline=False),
            yaxis=dict(title="Total Score",ticklen=5,zeroline=False))

fig=dict(data=data,layout=x)
iplot(fig)


df2014_3=df2014[df2014.year==2014].iloc[9:13,:]
df2014_3
import plotly.graph_objs as go

trace1=go.Bar(x=df2014_3.university_name,
              y=df2014_3.research,
              name="Research",
              marker = dict(color="rgba(32,178,170,0.8)" , line=dict(color='rgba(0,0,0)',width=1.5)),
              text=df2014_3.country)


trace2=go.Bar(x=df2014_3.university_name,
              y=df2014_3.teaching,
              name="Teaching",
              marker=dict(color='rgba(165,42,42,0.9)',line=dict(color='rgba(0,0,0)',width=1.5)),
             text=df2014_3.country)


data=[trace1,trace2]
layout=go.Layout(barmode="group")
fig=go.Figure(data=data,layout=layout)
iplot(fig)
df2016_3=df2016[df2016.year==2016].iloc[13:16,:]

import plotly.graph_objs as go

x = df2016_3.university_name

trace1 = {
  'x': x,
  'y': df2014.citations,
  'name': 'citation',
  'type': 'bar'
}
trace2 = {
  'x': x,
  'y': df2014.teaching,
  'name': 'teaching',
  'type': 'bar'
}
data = [trace1, trace2];
layout = {
  'xaxis': {'title': 'Top 3 universities'},
  'barmode': 'relative',
  'title': 'citations and teaching of top 3 universities in 2014'
}
fig = dict(data = data, layout = layout)
iplot(fig)
df2016 = timesData[timesData.year == 2016].iloc[:7,:]
pie1 = df2016.num_students
df2016.num_students= [float(each.replace(',', '.')) for each in df2016.num_students]
labels = df2016.university_name

fig={"data": [
    {
      "values": df2016.num_students,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "Number Of Students Rates",
      "hoverinfo":"label+percent",
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
df2016 = timesData[timesData.year == 2016].iloc[:20,:]
df2016
# data preparation
df2016 = timesData[timesData.year == 2016].iloc[:20,:]
num_students_size  = [float(each.replace(',', '.')) for each in df2016.num_students]
international_color = [float(each) for each in df2016.international]
data = [
    {
        'y': df2016.teaching, 
        'x': df2016.world_rank,
        'mode': 'markers',
        'marker': {
            'color': international_color,
            'size': num_students_size,
            'showscale': True
        },
        "text" :  df2016.university_name    
    }
]
iplot(data)
# prepare data
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
# data prepararion
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
# data preparation
x2015 = timesData[timesData.year == 2015]

trace0 = go.Box(
    y=x2015.research    ,
    name = 'research of universities in 2015',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace1 = go.Box(
    y=x2015.citations                 ,
    name = 'citations of universities in 2015',
    marker = dict(
        color = 'rgb(12, 128, 128)',
    )
)
data = [trace0, trace1]
iplot(data)
# import figure factory
import plotly.figure_factory as ff
# prepare data
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
    y=dataframe.teaching,
    name = "teaching",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
     mode = "markers"
)
# second line plot
trace2 = go.Scatter(
    x=dataframe.world_rank,
    y=dataframe.income,
    xaxis='x2',
    yaxis='y2',
    name = "income",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)
data = [trace1, trace2]
layout = go.Layout(
    xaxis2=dict(
        domain=[0.6, 0.95],
        anchor='y2',        
    ),
    yaxis2=dict(
        domain=[0.6, 0.95],
        anchor='x2',
    ),
    title = 'Income and Teaching vs World Rank of Universities'

)

fig = dict(data=data, layout=layout)
iplot(fig)
# create trace 1 that is 3d scatter
trace1 = go.Scatter3d(
    x=dataframe.world_rank,
    y=dataframe.research,
    z=dataframe.citations,
    mode='markers',
    marker=dict(
        size=10,# bunada bir şey diyip 5 boyut yapılabilir 
        color=dataframe.student_staff_ratio
        #color='rgb(0,128,128)',                # set color to an array/list of desired values      
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
    yaxis2=dict(
        domain=[0, 0.45],
        anchor='x2'
    ),
    xaxis3=dict(
        domain=[0, 0.45],
        anchor='y3'
    ),
    xaxis4=dict(
        domain=[0.55, 1],
        anchor='y4'
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






























