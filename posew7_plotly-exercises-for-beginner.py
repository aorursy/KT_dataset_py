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
# import figure factory
import plotly.figure_factory as ff

# word cloud library
from wordcloud import WordCloud

# matplotlib
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataTimes = pd.read_csv("../input/timesData.csv")
dataTimes.info()
dataTimes.head()
    # Line Charts
df = dataTimes.iloc[:100,:]

trace1 = go.Scatter(
                    x=df.world_rank,
                    y=df.citations,
                    mode="lines",
                    name="citations",
                    marker=dict(color="rgba(16,112,2,0.7)"),
                    text=df.university_name)

trace2 = go.Scatter(
                    x=df.world_rank,
                    y=df.teaching,
                    mode="lines+markers",
                    name="teaching",
                    marker=dict(color="rgba(80,6,80,0.7)"),
                    text=df.university_name)

data = [trace1, trace2]
layout = dict(title="Citation and Teaching vs World Rank of Top 100 Universities",
             xaxis=dict(title="World Rank", zeroline=False, ticklen=5))
fig = dict(data=data, layout=layout)
iplot(fig)
    # Scatter Charts
df2014 = dataTimes[dataTimes.year == 2014].iloc[:100,:]
df2015 = dataTimes[dataTimes.year == 2015].iloc[:100,:]
df2016 = dataTimes[dataTimes.year == 2016].iloc[:100,:]

trace1 = go.Scatter(
    x = df2014.world_rank,
    y = df2014.citations,
    mode = "markers",
    text = df2014.university_name,
    name = "2014",
    marker = dict(color=("rgba(255,0,0,0.8)"))
)

trace2 = go.Scatter(
    x = df2015.world_rank,
    y = df2015.citations,
    name = "2015",
    text = df2015.university_name,
    mode = "markers",
    marker = dict(color=("rgba(0,255,0,0.8)"))
)

trace3 = go.Scatter(
    x = df2016.world_rank,
    y = df2016.citations,
    name = "2016",
    mode = "markers",
    marker = dict(color=("rgba(0,0,255,0.8)")),
    text = df2016.university_name
)

data = [trace1,trace2,trace3]
layout = dict(title="Citation vs world rank of top 100 universities with 2014, 2015 and 2016 years",
             xaxis=dict(title = "World Rank", zeroline=False, ticklen=5),
             yaxis=dict(title = "Citation", zeroline=False,ticklen=5))
fig = dict(data=data, layout=layout)
iplot(fig)
    # Bar Charts
df2014 = dataTimes[dataTimes.year == 2014].iloc[:3,:]

trace1 = go.Bar(
    x = df2014.university_name,
    y =df2014.citations,
    name = "citations",
    marker = dict(color=("rgba(0,0,255,.7)"), line=dict(color="rgb(0,0,0)", width=1.5)),
    text = df2014.country
)

trace2 = go.Bar(
    x=df2014.university_name,
    y=df2014.teaching,
    name = "teaching",
    marker = dict(color=("rgba(255,255,0,.7)"), line=dict(color=("rgb(0,0,0)"), width=1.5)),
    text=df2014.country
)

data = [trace1,trace2]
layout = go.Layout(barmode="group")
fig = go.Figure(data=data,layout=layout)
iplot(fig)
    # Bar Charts
df2014 = df2014[df2014.year == 2014].iloc[:3,:]

trace1 = {
    "x" : df2014.university_name,
    "y" : df2014.citations,
    "type" : "bar",
    "name" : "citations"
}

trace2 ={
    "x" : df2014.university_name,
    "y" : df2014.teaching,
    "name" : "teaching",
    "type" : "bar"
}

data = [trace1,trace2]
layout = {
    "barmode" : "relative",
    "title" : "citations and teaching of top 3 universities in 2014",
    "xaxis" : {"title" : "Top 3 Universites"}
}
fig = go.Figure(data=data,layout=layout)
iplot(fig)
    # Pie Charts
df2016 = dataTimes[dataTimes.year == 2016].iloc[:7,:]
pie_list = [float(i.replace(",",".")) for i in df2016.num_students]
labels = df2016.university_name

fig = {
  "data": [
    {
      "values": pie_list,
      "labels": labels,
      "domain": {"x": [0, .7]},
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
    # Bubble Charts
df2016 = dataTimes[dataTimes.year == 2016].iloc[:20,:]
num_students_size = [float(i.replace(",",".")) for i in df2016.num_students]
international_color = [float(i) for i in df2016.international]
data = [
    {
        "x" : df2016.world_rank,
        "y" : df2016.teaching,
        "marker" : {
            "color" : international_color,
            "size" : num_students_size,
            "showscale" : True
        },
        "text" : df2016.university_name,
        "mode" : "markers"
    }
]
iplot(data)
    # Histogram
x2011 = dataTimes.student_staff_ratio[dataTimes.year == 2011]
x2012 = dataTimes.student_staff_ratio[dataTimes.year == 2012]

trace1 = go.Histogram(
    x=x2011,
    name="2011",
    opacity=0.75,
    marker=dict(color=("rgba(0,0,255,0.7)"))
)

trace2 = go.Histogram(
    x=x2012,
    name="2012",
    opacity=0.75,
    marker=dict(color=("rgba(255,255,0,0.7)"))
)

data = [trace1,trace2]
layout = go.Layout(barmode="overlay",
                  title="Students Staff Ratio in 2011 and 2012",
                  xaxis=dict(title="Students Staff Ratio"),
                  yaxis=dict(title="Count"))
fig = go.Figure(data=data, layout=layout)
iplot(fig)
    # World Cloud
x2011 = dataTimes.country[dataTimes.year==2011]
plt.subplots(figsize=(9,9))
worldcloud = WordCloud(
    background_color="white",
    width=512,
    height=384
).generate(" ".join(x2011))

plt.imshow(worldcloud)
plt.axis("off")
plt.show()
    # Box Plot
x2015 = dataTimes[dataTimes.year==2015]

trace1 = go.Box(
    y=x2015.total_score,
    name="total score of universities in 2015",
    marker = dict(color=("rgba(0,0,255,0.7)"))
)
trace2 = go.Box(
    y=x2015.research,
    name="research of universities in 2015",
    marker = dict(color=("rgba(255,255,0,0.7)"))
)

data = [trace1,trace2]
iplot(data)
    # Scatter Plot Matrix
df = dataTimes[dataTimes.year==2015]
data2015 = df.loc[:,["research","international","total_score"]]
data2015["index"] = np.arange(1,len(data2015)+1)

fig = ff.create_scatterplotmatrix(data2015, diag="box", index="index", colormap="Portland", colormap_type="cat", height=700,width=700)
iplot(fig)
    # Inset Plot
trace1 = go.Scatter(
    x = df.world_rank,
    y = df.teaching,
    name = "teaching",
    marker = dict(color=("rgba(0,0,255,0.7)"))
)
trace2 = go.Scatter(
    x = df.world_rank,
    y = df.income,
    name = "income",
    marker = dict(color=("rgba(255,255,0,0.7)")),
    xaxis="x2",
    yaxis="y2"
)
data = [trace1,trace2]
layout = go.Layout(xaxis2=dict(domain=[0.6,0.95],anchor="x2"),yaxis2=dict(domain=[0.6,0.95],anchor="y2"),
                  title = "Income and Teaching vs World Rank of Universities")
fig = go.Figure(data=data, layout=layout)
iplot(fig)
    # 3D Scatter Plot with Colorscaling
trace = go.Scatter3d(
    x=df.world_rank,
    y=df.citations,
    z=df.research,
    marker=dict(color=("rgba(0,0,255,0.7)"), size=10)
)
data =[trace]
layout = go.Layout(margin=dict(l=0,r=0,t=0,b=0))
fig = go.Figure(data=data,layout=layout)
iplot(fig)
    # Multiple Subplots
trace1 = go.Scatter(
    x=df.world_rank,
    y=df.research,
    name="research"
)
trace2 = go.Scatter(
    x=df.world_rank,
    y=df.citations,
    name="citations",
    xaxis="x2",
    yaxis="y2"
)
trace3 = go.Scatter(
    x=df.world_rank,
    y=df.income,
    name="income",
    xaxis="x3",
    yaxis="y3"
)
trace4 = go.Scatter(
    x=df.world_rank,
    y=df.total_score,
    name="total score",
    xaxis="x4",
    yaxis="y4"
)
layout = go.Layout(xaxis=dict(domain=[0,0.45]),yaxis=dict(domain=[0,0.45]),
                  xaxis2=dict(domain=[0.55,1],anchor="x2"),yaxis2=dict(domain=[0,0.45],anchor="y2"),
                  xaxis3=dict(domain=[0,0.45],anchor="x3"),yaxis3=dict(domain=[0.55,1],anchor="y3"),
                  xaxis4=dict(domain=[0.55,1],anchor="x4"),yaxis4=dict(domain=[0.55,1],anchor="y4"),
    title = 'Research, citation, income and total score VS World Rank of Universities')
data = [trace1,trace2,trace3,trace4]
fig = go.Figure(data=data, layout=layout)
iplot(fig)
    # Multiple Subplots
trace1 = go.Scatter(
    x=df.world_rank,
    y=df.research,
    name="research"
)
trace2 = go.Scatter(
    x=df.world_rank,
    y=df.citations,
    name="citations",
    xaxis="x2",
    yaxis="y2"
)
trace3 = go.Scatter(
    x=df.world_rank,
    y=df.income,
    name="income",
    xaxis="x3",
    yaxis="y3"
)
trace4 = go.Scatter(
    x=df.world_rank,
    y=df.total_score,
    name="total score",
    xaxis="x4",
    yaxis="y4"
)
layout = go.Layout(xaxis=dict(domain=[0,0.45]),yaxis=dict(domain=[0,0.45]),
                  xaxis2=dict(domain=[0.55,1],anchor="x2"),yaxis2=dict(domain=[0,0.45],anchor="y2"),
                  xaxis3=dict(domain=[0,0.45],anchor="x3"),yaxis3=dict(domain=[0.55,1],anchor="y3"),
                  xaxis4=dict(domain=[0.55,1],anchor="x4"),yaxis4=dict(domain=[0.55,1],anchor="y4"))
data = [trace1,trace2,trace3,trace4]
fig = go.Figure(data=data, layout=layout)
iplot(fig)

























