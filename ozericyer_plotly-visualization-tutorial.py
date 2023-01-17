# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/world-university-rankings/timesData.csv")
df=data.copy()
df.tail()
df.isnull().sum()
df.info() # we have 2603 observations
df.describe()
df["year"].unique()
# import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# prepare data frame.First 100 worldrank for values of citations
df = df.iloc[:100,:]

# import graph objects as "go"
import plotly.graph_objs as go

# Creating trace1
trace1 = go.Scatter(
                    x = df.world_rank,
                    y = df.citations,
                    mode = "lines",
                    name = "citations",                               
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),   #mode = type of plot like marker, line or line + markers
                    text= df.university_name)                         #color = color of lines. It takes RGB (red, green, blue) and opacity (alpha)
# Creating trace2                                                     #text = The hover text (hover is curser)
trace2 = go.Scatter(
                    x = df.world_rank,
                    y = df.teaching,
                    mode = "lines+markers",
                    name = "teaching",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= df.university_name)
data = [trace1, trace2]   #data = is a list that we add traces into it
layout = dict(title = 'Citation and Teaching vs World Rank of Top 100 Universities',  #layout = it is dictionary.
              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)    #title = label of x axis,ticklen = length of x axis ticks, zeroline = showing zero line or not
             )
fig = dict(data = data, layout = layout) #fig = it includes data and layout
iplot(fig)   #iplot() = plots the figure(fig) that is created by data and layout
# prepare data frames
df=data.copy()

df2014 = df[df.year == 2014].iloc[:100,:]
df2015 = df[df.year == 2015].iloc[:100,:]
df2016 = df[df.year == 2016].iloc[:100,:]
# import graph objects as "go"
import plotly.graph_objs as go
# creating trace1
trace1 =go.Scatter(
                    x = df2014.world_rank,
                    y = df2014.citations,
                    mode = "markers",   #mode = type of plot like marker, line or line + markers
                    name = "2014",  #name = name of the plots
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'), #marker = marker is used with dictionary,color = color of lines. It takes RGB (red, green, blue) and opacity (alpha) 
                    text= df2014.university_name)#text = The hover text (hover is curser)
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
data = [trace1, trace2, trace3]  #data = is a list that we add traces into it
layout = dict(title = 'Citation vs world rank of top 100 universities with 2014, 2015 and 2016 years', #layout = it is dictionary,title = title of layout,
              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False), #ticklen = length of x axis ticks,zeroline = showing zero line or not
              yaxis= dict(title= 'Citation',ticklen= 5,zeroline= False) #y axis = it is dictionary and same with x axis
             )
fig = dict(data = data, layout = layout)#fig = it includes data and layout
iplot(fig) #
data=pd.read_csv("/kaggle/input/world-university-rankings/timesData.csv")
df=data.copy()

# prepare data frames
df2014 = df[df.year == 2014].iloc[:3,:]
df2014
# prepare data frames
df2014 = df[df.year == 2014].iloc[:3,:]
# import graph objects as "go"
import plotly.graph_objs as go
# create trace1 
trace1 = go.Bar(
                x = df2014.university_name,
                y = df2014.citations,
                name = "citations",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df2014.country)  #text = The hover text (hover is curser)
# create trace2 
trace2 = go.Bar(
                x = df2014.university_name,
                y = df2014.teaching,
                name = "teaching",
                marker = dict(color = 'rgba(255, 255, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df2014.country)
data = [trace1, trace2]
layout = go.Layout(barmode = "group")  #barmode = bar mode of bars like grouped
fig = go.Figure(data = data, layout = layout)
iplot(fig)
data=pd.read_csv("/kaggle/input/world-university-rankings/timesData.csv")
df=data.copy()

df2014 = df[df.year == 2014].iloc[:3,:]
# import graph objects as "go"
import plotly.graph_objs as go

x = df2014.university_name

trace1 = {
  'x': x,
  'y': df2014.citations,
  'name': 'citation',
  'type': 'bar'       #type = type of plot like bar plot
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
  'barmode': 'relative', #barmode = bar mode of bars like grouped( previous example) or relative
  'title': 'citations and teaching of top 3 universities in 2014'
};
fig = go.Figure(data = data, layout = layout)
iplot(fig)

import plotly.graph_objs as go
from plotly import tools #Tools: used for subplots
import matplotlib.pyplot as plt
# prepare data frames
df2016 = df[df.year == 2016].iloc[:7,:]

y_saving = [each for each in df2016.research]
y_net_worth  = [float(each) for each in df2016.income]
x_saving = [each for each in df2016.university_name]
x_net_worth  = [each for each in df2016.university_name]
trace0 = go.Bar(            #bar: bar plot
                x=y_saving,
                y=x_saving,
                marker=dict(color='rgba(171, 50, 96, 0.6)',line=dict(color='rgba(171, 50, 96, 1.0)',width=1)),#color of bar,bar line color and with
                name='research',  #name: name of bar
                orientation='h', #orientation: orientation like horizontal
)
trace1 = go.Scatter(
                x=y_net_worth,
                y=x_net_worth,
                mode='lines+markers', #mode: scatter type line line + markers or only markers
                line=dict(color='rgb(63, 72, 204)'),#line: properties of line,color: color of line
                name='income',  #name: name of scatter plot
)
layout = dict(      #layout: axis, legend, margin, paper and plot properties *
                title='Citations and income',
                yaxis=dict(showticklabels=True,domain=[0, 0.85]),
                yaxis2=dict(showline=True,showticklabels=False,linecolor='rgba(102, 102, 102, 0.8)',linewidth=2,domain=[0, 0.85]),
                xaxis=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0, 0.42]),
                xaxis2=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0.47, 1],side='top',dtick=25),
                legend=dict(x=0.029,y=1.038,font=dict(size=10) ),
                margin=dict(l=200, r=20,t=70,b=70),
                paper_bgcolor='rgb(248, 248, 255)',
                plot_bgcolor='rgb(248, 248, 255)',
)
annotations = []
y_s = np.round(y_saving, decimals=2)
y_nw = np.rint(y_net_worth)
# Adding labels
for ydn, yd, xd in zip(y_nw, y_s, x_saving):
    # labeling the scatter savings
    annotations.append(dict(xref='x2', yref='y2', y=xd, x=ydn - 4,text='{:,}'.format(ydn),font=dict(family='Arial', size=12,color='rgb(63, 72, 204)'),showarrow=False))
    # labeling the bar net worth
    annotations.append(dict(xref='x1', yref='y1', y=xd, x=yd + 3,text=str(yd),font=dict(family='Arial', size=12,color='rgb(171, 50, 96)'),showarrow=False))

layout['annotations'] = annotations

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,
                          shared_yaxes=False, vertical_spacing=0.001)

fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)

fig['layout'].update(layout)
iplot(fig)
# data preparation
df2016 = df[df.year == 2016].iloc[:7,:]
pie1 = df2016.num_students
pie1_list = [float(each.replace(',', '.')) for each in df2016.num_students]  # str(2,4) => str(2.4) = > float(2.4) = 2.4
labels = df2016.university_name
# create figure
fig = {
  "data": [    #data: plot type
    {
      "values": pie1_list, #values: values of plot
      "labels": labels, #labels: labels of plot
      "domain": {"x": [0, .5]},
      "name": "Number Of Students Rates",
      "hoverinfo":"label+percent+name", #hoverinfo: information in hover
      "hole": .3, #hole: hole width
      "type": "pie" #type: plot type like pie****
    },],
  "layout": {   #layout: layout of plot
        "title":"Universities Number of Students rates",#title: title of layout
        "annotations": [                            #annotations: font, showarrow, text, x, y
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
df2016.info()
# data preparation
df2016 = df[df.year == 2016].iloc[:20,:]
num_students_size  = [float(each.replace(',', '.')) for each in df2016.num_students]
international_color = [float(each) for each in df2016.international]
data = [
    {
        'y': df2016.teaching,
        'x': df2016.world_rank,
        'mode': 'markers',  #mode = markers(scatter)
        'marker': {    #marker = marker properties;color = third dimension of plot. Internaltional score,size = fourth dimension of plot. Number of students
            'color': international_color,
            'size': num_students_size,
            'showscale': True
        },
        "text" :  df2016.university_name #text: university names   
    }
]
iplot(data)
# prepare data
x2011 = df.student_staff_ratio[df.year == 2011]
x2012 = df.student_staff_ratio[df.year == 2012]

trace1 = go.Histogram(  #trace1 = first histogram
    x=x2011,
    opacity=0.75,  #opacity = opacity of histogram
    name = "2011", #name = name of legend
    marker=dict(color='rgba(171, 50, 96, 0.6)')) #marker = color of histogram
trace2 = go.Histogram(
    x=x2012,
    opacity=0.75,
    name = "2012",
    marker=dict(color='rgba(12, 50, 196, 0.6)'))

data = [trace1, trace2]
layout = go.Layout(barmode='overlay',  #barmode = mode of histogram like overlay. Also you can change it with stack
                   title=' students-staff ratio in 2011 and 2012',
                   xaxis=dict(title='students-staff ratio'),
                   yaxis=dict( title='Count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
# word cloud library
from wordcloud import WordCloud
# data prepararion
x2011 = df.country[df.year == 2011]
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(         #WordCloud = word cloud library that I import at the beginning of kernel
                          background_color='white', #background_color = color of back ground
                          width=512,
                          height=384
                         ).generate(" ".join(x2011)) #generate = generates the country name list(x2011) a word cloud
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()
# data preparation
x2015 = df[df.year == 2015]

trace0 = go.Box(
    y=x2015.total_score,
    name = 'total score of universities in 2015',
    marker = dict(                          #marker = color
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
# import figure factory
import plotly.figure_factory as ff
# prepare data
dataframe =df[df.year == 2015]
data2015 = dataframe.loc[:,["research","international", "total_score"]]#data2015 = prepared data. It includes research, international and total scores with index from 1 to 401
data2015["index"] = np.arange(1,len(data2015)+1)
# scatter matrix
fig = ff.create_scatterplotmatrix(data2015, diag='box', index='index',colormap='Portland',  #colormap = color map of scatter plot
                                  colormap_type='cat',    #colormap_type = color type of scatter plot
                                  height=700, width=700)
iplot(fig)
# first line plot
trace1 = go.Scatter(
    x=df.world_rank,
    y=df.teaching,
    name = "teaching",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)
# second line plot
trace2 = go.Scatter(
    x=df.world_rank,
    y=df.income,
    xaxis='x2',
    yaxis='y2',
    name = "income",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)
dat = [trace1, trace2]
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

fig = go.Figure(data=dat, layout=layout)
iplot(fig)
# create trace 1 that is 3d scatter
trace1 = go.Scatter3d(    #go.Scatter3d: create 3d scatter plot
    x=df.world_rank,    #x,y,z: axis of plots
    y=df.research,
    z=df.citations,
    mode='markers',     #mode: market that is scatter
    marker=dict(
        size=10,        #size: marker size, color: axis of colorscale, colorscale: actually it is 4th dimension
        color='rgb(255,0,0)',    # set color to an array/list of desired values      
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
