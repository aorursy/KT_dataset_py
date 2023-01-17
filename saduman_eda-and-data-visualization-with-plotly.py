import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

#plotly library
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

#matplotlib library
import matplotlib.pyplot as plt

#word cloud library
from wordcloud import WordCloud

#from a unix time to a date
from time import strftime
from datetime import datetime

import warnings            
warnings.filterwarnings("ignore") 

import os
print(os.listdir("../input"))
#Load data from csv file
dataframe=pd.read_csv('../input/ted_main.csv')
#Let's get general information about our data
dataframe.info()
#rare visualization tool
#import missing library
import missingno as msno
msno.matrix(dataframe)
plt.show()
dataframe.head()
trace1 = go.Scatter(
                    x = dataframe.index,
                    y = dataframe.comments,
                    mode = "lines",
                    name = "comments",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= dataframe.main_speaker)
data2 = [trace1]
layout = dict(title = 'Comment numbers for Ted Talks',
              xaxis= dict(title= 'index',ticklen= 5,zeroline= False)
             )
fig = dict(data = data2, layout = layout)
iplot(fig)
df=dataframe[dataframe['main_speaker']=='Richard Dawkins']
df
dataframe['main_speaker'].nunique()
dfGlobal=dataframe[dataframe['event']=='TEDGlobal 2005']
df2002=dataframe[dataframe['event']=='TED2002']
dfRoyal=dataframe[dataframe['event']=='Royal Institution']

trace1 =go.Scatter(
                    x = dfGlobal.index,
                    y = dfGlobal.views,
                    mode = "markers",
                    name = "TEDGlobal 2005",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text= dfGlobal.main_speaker)
# creating trace2
trace2 =go.Scatter(
                    x = df2002.index,
                    y = df2002.views,
                    mode = "markers",
                    name = "TED2002",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text= df2002.main_speaker)
# creating trace3
trace3 =go.Scatter(
                    x = dfRoyal.index,
                    y = dfRoyal.views,
                    mode = "markers",
                    name = "Royal Institution",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text= dfRoyal.main_speaker)
data3 = [trace1, trace2, trace3]
layout = dict(title = 'Number of views received at TEDGlobal 2005, TED2002, Royal Institution',
              xaxis= dict(title= 'index',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Views',ticklen= 5,zeroline= False)
             )
fig = dict(data = data3, layout = layout)
iplot(fig)
#sort by highest number of comments
data_sorted=dataframe.sort_values(by='comments',ascending=False)
#convert unix timestamp
data_sorted['published_date']=[datetime.fromtimestamp(int(item)).strftime('%Y') for item in data_sorted.published_date]
#get 6 speakers with the highest number of comments
data_comments=data_sorted.iloc[:6,:]
#duration convert  to minute
import datetime
data_duration=[]
data_duration=[str(datetime.timedelta(seconds=i))+" minute " for i in data_comments.duration]
date=[]
for item in data_comments.published_date:
    date.append(item + 'Year')

#visualization
#create trace1
trace1 = go.Bar(
                x = date,
                y = data_comments.comments,
                name = "comments",
                marker = dict(color = 'rgba(255, 58, 255, 0.4)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = data_comments.main_speaker)
# create trace2 
trace2 = go.Bar(
                x = date,
                y = data_comments.duration,
                name = "duration",
                marker = dict(color = 'rgba(15, 15, 250, 0.4)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = (data_duration + data_comments.main_speaker))
data4 = [trace1, trace2]
layout = go.Layout(barmode = "group")
fig = go.Figure(data=data4, layout=layout)
iplot(fig)
#get 3 speakers with the highest number of comments
data_comments=data_sorted.iloc[:3,:]
#visualization
trace1 = {
  'x': data_comments.main_speaker,
  'y': data_comments.comments,
  'name': 'comments',
  'type': 'bar',
  'marker':dict(
        color='rgb(58,200,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
   'opacity':0.6,
};
trace2 = {
  'x': data_comments.main_speaker,
  'y': data_comments.duration,
  'name': 'duration',
  'type': 'bar',
  'text':data_duration,
  'marker':dict(
        color='rgb(158,202,225)',
        line=dict(color='rgb(8,48,107)',
                    width=1.5)),
  'opacity':0.6,
};
data5 = [trace1, trace2];
layout = {
  'xaxis': {'title': 'Top 3 speakers'},
  'barmode': 'relative',
  'title': 'Number of comments and speech duration of the 3 most commented'
};
fig = go.Figure(data = data5, layout = layout)
iplot(fig)
#from a unix time to a date
from time import strftime
from datetime import datetime

dataframe['published_date']=[datetime.fromtimestamp(int(item)).strftime('%Y') for item in dataframe.published_date]
data_2006=dataframe[dataframe.published_date=='2006'].iloc[:,:]
labels=data_2006.event
# figure
fig = {
  "data": [
    {
      "values": data_2006.views,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "Number Of Views Rates",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"The number of watched talks events published in 2006",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Number of Views",
                "x": 0.30,
                "y": 1.10
            },
        ]
    }
}
iplot(fig)
data_sorted=dataframe.sort_values(by='views',ascending=True)
df=data_sorted.iloc[:20,:]
df.index=range(0,len(df))
#visualization
data = [
    {
        'y': df.views,
        'x': df.index,
        'mode': 'markers',
        'marker': {
            'color': df.duration,
            'size': df.comments,
            'showscale': True
        },
        "text" :  df.main_speaker    
    }
]
iplot(data)

data_2014=dataframe.comments[dataframe.event=='TED2014']
data_2015=dataframe.comments[dataframe.event=='TED2015']
    
trace2 = go.Histogram(
    x=data_2014,
    opacity=0.75,
    name = "2014",
    marker=dict(color='rgba(12, 50, 196, 0.6)'))
trace3 = go.Histogram(
    x=data_2015,
    opacity=0.75,
    name = "2015",
    marker=dict(color='rgba(125, 2, 100, 0.6)'))
data = [trace2, trace3]
layout = go.Layout(barmode='overlay',
                   title=' Comments in 2014 and 2015',
                   xaxis=dict(title='number of comments'),
                   yaxis=dict( title='Count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
data_2017=dataframe.tags[dataframe.published_date=='2017']
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(data_2017))
plt.imshow(wordcloud)
plt.axis('off')


plt.show()


data_2012=dataframe[dataframe.event=='TED2012']
#visualization
trace0 = go.Box(
    y=data_2012.comments,
    name = 'number of comments in TED2012',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace1 = go.Box(
    y=data_2012.duration,
    name = 'number of duration in TED2012',
    marker = dict(
        color = 'rgb(12, 128, 128)',
    )
)
data = [trace0, trace1]
iplot(data)
data_2012[data_2012.duration==181]
# import figure factory
import plotly.figure_factory as ff
df_occupation=dataframe[dataframe.event=='TED2012']
data_occupation = df_occupation.loc[:,["comments", "views"]]
data_occupation['index'] = np.arange(1,len(data_occupation)+1)
# scatter matrix
fig = ff.create_scatterplotmatrix(data_occupation, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',
                                  height=700, width=700)
iplot(fig)
#duration convert  to minute
import datetime
data_duration2=[]
data_duration2=[str(datetime.timedelta(seconds=i))+" minute " for i in dataframe.duration]
df_100=dataframe.iloc[:100,:]
#visualization
# first line plot
trace1 = go.Scatter(
    x=df_100.index,
    y=df_100.views,
    name = "views",
    marker = dict(color = 'rgba(200, 75, 45, 0.8)'),
)
# second line plot
trace2 = go.Scatter(
    x=df_100.index,
    y=df_100.duration,
    xaxis='x2',
    yaxis='y2',
    name = "duration",
    text=data_duration2,
    marker = dict(color = 'rgba(85, 20, 200, 0.8)'),
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
    title = 'Views and Comments'
)
fig = go.Figure(data=data, layout=layout)
plt.savefig('graph.png')
iplot(fig)
plt.show()

data_sorted2=dataframe.sort_values(by='views',ascending=False)
df_150=data_sorted2.iloc[:150,:]
df_150['views_rank']=np.arange(1,len(df_150)+1)

x, y, z = np.random.multivariate_normal(np.array([0,0,0]), np.eye(3), 400).transpose()
# create trace 1 that is 3d scatter
trace1 = go.Scatter3d(
    x=df_150.views_rank,
    y=df_150.comments,
    z=df_150.duration,
    mode='markers',
    marker=dict(
        size=12,
        color=z,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    ),
    text=data_duration2,)
data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0))
fig = go.Figure(data=data, layout=layout)
iplot(fig)
trace1 = go.Scatter(
    x=df_150.views_rank,
    y=df_150.comments,
    xaxis='x3',
    yaxis='y3',
    name = "comments"
)
trace2 = go.Scatter(
    x=df_150.views_rank,
    y=df_150.duration,
    xaxis='x4',
    yaxis='y4',
    name = "duration",
    text=data_duration2,
)
data = [trace1, trace2]
layout = go.Layout(
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
    title = 'Number of Comments and Number of Duration VS Number of Views Rank '
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)