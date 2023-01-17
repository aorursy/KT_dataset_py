# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# matplotlib
import matplotlib.pyplot as plt

# Plotly
# import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go

#word cloud library
from wordcloud import WordCloud

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load data that we will use.
timesData = pd.read_csv("/kaggle/input/world-university-rankings/timesData.csv")
# information about timesData
timesData.info()
timesData.head(10)
# prepare data frame
df = timesData.iloc[:100,:]

# Creating trace1
trace1 = go.Scatter(x=df.world_rank, 
                    y=df.citations, 
                    mode="lines+markers", #lines or lines+marker
                    name="citations", 
                    marker = dict(color="rgba(16,112,2,0.8)"), # color of lines
                    text = df.university_name #when you put a curser of mouse to along line that you get info about which university
                   )
# Creating trace2
trace2 = go.Scatter(x=df.world_rank, 
                    y=df.teaching, 
                    mode="lines+markers",
                    name="teaching",
                    marker = dict(color = "rgba(80,26,80,0.8)"), # color of lines+marker
                    text = df.university_name #when you put a curser of mouse to along line that you get info about which university
                   )
data= [trace1,trace2]
layout = dict(title="Citation and Teaching vs World Rank of Top 100 Universities",
              xaxis = dict(title = "World Rank", ticklen=5, zeroline=False)
             )
fig = dict(data=data, layout=layout)
iplot(fig)
# prepare data frame
df2014= timesData[timesData.year == 2014].iloc[:100,:]
df2015= timesData[timesData.year == 2015].iloc[:100,:]
df2016= timesData[timesData.year == 2016].iloc[:100,:]

# Creating trace1
trace1 = go.Scatter(x=df2014.world_rank,
                    y=df2014.citations,
                    mode= "markers",
                    name= "2014",
                    marker = dict(color="rgba(255,128,255,0.9)"),
                    text = df2014.university_name
                   )
# Creating trace2
trace2 = go.Scatter(x=df2015.world_rank,
                    y=df2015.citations,
                    mode = "markers",
                    name = "2015",
                    marker = dict(color="rgba(255,128,2,0.8)"),
                    text = df2015.university_name
                   )
# Creating trace3
trace3 = go.Scatter(x=df2016.world_rank,
                    y=df2016.citations,
                    mode = "markers",
                    name = "2016",
                    marker = dict(color="rgba(0,255,200,0.9)"),
                    text = df2016.university_name
                   )
data1=[trace1,trace2,trace3]
layout = dict(title="Citation vs world rank of top 100 universities with 2014, 2015 and 2016 years",
              xaxis = dict(title="World Rank", ticklen=5, zeroline=False),
              yaxis = dict(title="Citation",   ticklen=5, zeroline=False)
             )
fig = dict(data=data1, layout=layout)
iplot(fig)
# filter data frame
df2014 = timesData[timesData.year == 2014].iloc[:3,:]
df2014
# prepare data frame
df2014 = timesData[timesData.year ==2014].iloc[:3,:]
# Style 1
# create trace1
trace1 = go.Bar(x=df2014.university_name,
                y=df2014.citations,
                name="citations",
                marker=dict(color="rgba(255,174,255,0.65)",
                            line=dict(color="rgb(0,0,0)", width=1.5)),
                text=df2014.country
)
# create trace2
trace2 = go.Bar(x=df2014.university_name,
                y=df2014.teaching,
                name="teaching",
                marker=dict(color="rgba(255,255,128,0.75)",
                            line=dict(color="rgb(0,0,0)", width=1.5)),
                text=df2014.country
)
data2=[trace1,trace2]
layout=go.Layout(barmode="group")
fig = go.Figure(data=data2, layout=layout)
iplot(fig)
# prepare data frame
df2014 = timesData[timesData.year == 2014].iloc[:3,:]

x = df2014.university_name

trace1 = {"x": x,
          "y": df2014.citations,
          "name": "citation",
          "type": "bar"
         }
trace2 = {"x": x,
          "y": df2014.teaching,
          "name": "teaching",
          "type": "bar"
         }
data3 = [trace1,trace2]
layout = {
    "xaxis": {"title": "Top 3 universities"},
    "barmode": "relative",# this code change but all thins are same 
    "title": "citations and teaching of top 3 universities in 2014"
}
fig = go.Figure(data=data3, layout=layout)
iplot(fig)
# import graph objects as go and import tools
from plotly import tools

# prepare data frames
df2016 = timesData[timesData.year == 2016].iloc[:7,:]

y_saving = [each for each in df2016.research]
y_net_worth = [float(each) for each in df2016.income]
x_saving = [each for each in df2016.university_name]
x_net_worth = [each for each in df2016.university_name]

# trace 0
trace0 = go.Bar( x=y_saving,
                 y=x_saving,
                 marker=dict(color="rgba(171,50,96,0.6)", line=dict(color="rgba(171,68,96,1.0)", width=1)),
                 name="research",
                 orientation="h"
               )
trace1 = go.Scatter(x=y_net_worth,
                    y=x_net_worth,
                    mode="lines+markers",
                    line=dict(color="rgba(63,72,204)"),
                    name="income",
                   )
layout = dict(
              title = "Citations and Income",
              yaxis=dict(showticklabels=True, domain=[0, 0.85]),
              yaxis2=dict(showline=True, showticklabels=False,linecolor="rgba(102, 102, 102,0.8)", linewidth=2, domain=[0, 0.85]),
              xaxis=dict(zeroline=False, showline=False, showticklabels=True, showgrid=True,domain=[0, 0.42]),
              xaxis2=dict(zeroline=False, showline=False, showticklabels=True, showgrid=True, domain=[0.47, 1], side="top", dtick=25),
              legend=dict(x=0.029, y=1.038, font=dict(size=10)),
              margin=dict(l=200, r=20, t=70, b=70),
              paper_bgcolor="rgb(248,248,255)",
              plot_bgcolor="rgb(248,248,255)",
)
annotations = []
y_s = np.round(y_saving, decimals=2)
y_nw = np.rint(y_net_worth)
# Adding Labels
for ydn, yd, xd in zip(y_nw, y_s, x_saving):
    # labeling the scatter savings
    annotations.append(dict(xref="x2", yref="y2", y=xd, x=ydn-4, text="{:,}".format(ydn), font=dict(family="Arial", size=12,color="rgb(63,72,204)"), showarrow=False))
    # labeling the bar net worth
    annotations.append(dict(xref="x1", yref="y1", y=xd, x=yd+3, text=str(yd), font=dict(family="Arial", size=12, color="rgb(171,50,96)"), showarrow=False))

layout["annotations"]=annotations
                       
# creating two subplots
fig = tools.make_subplots(rows=1, cols=2, 
                          specs=[[{},{}]], shared_xaxes=True,
                          shared_yaxes=False, vertical_spacing=0.001)
fig.append_trace(trace0, 1,1)
fig.append_trace(trace1, 1,2)
                       
fig["layout"].update(layout)
iplot(fig)
# data preparetion
df2016 = timesData[timesData.year == 2016].iloc[:7,:]
pie1 = df2016.num_students
pie1_list = [float(each.replace(",", ".")) for each in df2016.num_students ]
labels = df2016.university_name
# figure 
fig = { 
        # first one
        "data": [ {
         "values": pie1_list,
         "labels": labels,
         "domain": {"x": [0, 0.45]},
         "name": "Number of Students Rates",
         "hoverinfo": "label+percent+name",
         "hole": 0.3,
         "type": "pie"
} ,],
       # second one
        "layout": {
          "title": "Universities Number of Students rates",
        
          "annotations": [
              { 
                 "font": { "size": 15},
                 "showarrow": False,
                 "text": "Number of Students",
                 "x": 0,
                 "y": 0.85
              },
                         ]
    }   
}
iplot(fig)
df2016.info()
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
# prepare data frame
x2011 = timesData.student_staff_ratio[timesData.year == 2011]
x2012 = timesData.student_staff_ratio[timesData.year == 2012]

trace1 = go.Histogram(
                      x=x2011,
                      opacity=0.75,
                      name = "2011",
                      marker=dict(color="rgba(171,50,96, 0.6)")
)
trace2 = go.Histogram(x=x2012,
                      opacity=0.75,
                      name="2012",
                      marker=dict(color="rgba(12,56,196, 0.7)")
                     )
data=[trace1, trace2]
layout = go.Layout(
                   barmode="overlay",
                   title="Students-staff ratio in 2011 and 2012",
                   xaxis=dict(title="Students staff ratio"),
                   yaxis=dict(title="Count")
)
fig = go.Figure(data=data, layout=layout)

iplot(fig)
# data preparation
x2011 = timesData.country[timesData.year == 2011]
plt.subplots(figsize=(8,8))
wordcloud = WordCloud( 
                       background_color="white",
                       width=512,
                       height=384,
                     ).generate(" ".join(x2011))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("graph.png")

plt.show()
# data preparation
x2015 = timesData[timesData.year == 2015]

trace1 = go.Box( y=x2015.total_score,
                 name="Total score of universities in 2015",
                 marker = dict(color="rgb(12,12,140)"),
               )
trace2 = go.Box( y=x2015.research,
                 name="Research of universities in 2015",
                 marker=dict(color="rgb(12,120,128)")
               )
data=[trace1,trace2]
iplot(data)
# import figure factory
import plotly.figure_factory as ff
# prepare data
dataframe = timesData[timesData.year == 2015]
data2015 = dataframe.loc[:,["research", "international", "total_score"]]
data2015["index"] = np.arange(1,len(data2015)+1) # start from "1" to stop:402
# scatter matrix
fig = ff.create_scatterplotmatrix(data2015, diag="box", index="index", colormap="Portland",
                                  colormap_type="cat",
                                  height=700, width=700)
iplot(fig)
# first line plot
trace1 = go.Scatter(
                    x=dataframe.world_rank,
                    y=dataframe.teaching,
                    name="Teaching",
                    marker=dict(color="rgba(16,112,2, 0.8)")
                   )
# second line plot
trace2 = go.Scatter(
                    x=dataframe.world_rank,
                    y=dataframe.income,
                    xaxis="x2",
                    yaxis="y2",
                    name="income",
                    marker=dict(color="rgba(160, 112, 20, 0.8)")
                   )
data = [trace1, trace2]
layout = go.Layout(
         xaxis2=dict(
                     domain=[0.6, 0.95],anchor="y2"
                    ),
         yaxis2=dict(domain=[0.6, 0.95],
                    anchor="x2"),
         title="Income and Teaching vs World Rank of Universities"
)
fig=go.Figure(data=data, layout = layout)
iplot(fig)
# create trace1 that is 3d scatter
trace1 = go.Scatter3d( 
                      x=dataframe.world_rank,
                      y=dataframe.research,
                      z=dataframe.citations,
                      mode="markers",
                      marker=dict(size=10, color="rgb(255,0,0)")
                     )
#data=[trace1]
layout = go.Layout(margin=dict(l=0, r=0, t=0, b=0))

fig = go.Figure(data=trace1, layout=layout)
iplot(fig)


# create trace1
trace1 = go.Scatter(x=dataframe.world_rank,
                    y=dataframe.research,
                    name="research"
                   )
trace2 = go.Scatter(x=dataframe.world_rank,
                    y=dataframe.citations,
                    xaxis="x2",
                    yaxis="y2",
                    name="citations"
                   )
trace3 = go.Scatter(x=dataframe.world_rank,
                    y=dataframe.income,
                    xaxis="x3",
                    yaxis="y3",
                    name="income"
                   )
trace4 = go.Scatter(x=dataframe.world_rank,
                    y=dataframe.total_score,
                    xaxis="x4",
                    yaxis="y4",
                    name="total score"
                   )
data=[trace1, trace2, trace3, trace4]
                          # range(x): from 0 to 0.45 
layout = go.Layout(xaxis=dict(domain=[0, 0.45]), # trace1
                   yaxis=dict(domain=[0, 0.45]), # trace1
                          # range(y): from 0 to 0.45 
                   xaxis2=dict(domain=[0.55, 1], anchor="y2"), # trace2                   
                   yaxis2=dict(domain=[0, 0.45], anchor="x2"), # trace2
                   
                   xaxis3=dict(domain=[0, 0.45], anchor="y3"), # trace3
                   yaxis3=dict(domain=[0.55, 1], anchor="x3"), # trace3
                   
                   xaxis4=dict(domain=[0.55, 1], anchor="y4"), # trace4
                   yaxis4=dict(domain=[0.55, 1], anchor="x4"), # trace4
                   title="Research, citation, income and total score vs World Rank of Universities"
                  )
fig = go.Figure(data=data, layout=layout)

iplot(fig)
