# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import chart_studio.plotly as ply

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

from wordcloud import WordCloud

import plotly.graph_objs as go



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data1 = pd.read_csv("/kaggle/input/world-university-rankings/timesData.csv")

data1.head()

data1.info()
df = data1.iloc[:50,:].copy()

df.head()
# line chart

trace1 = go.Scatter(x=df.world_rank,

                    y=df.citations,

                    mode="lines",

                    name="citations",

                    marker=dict(color="rgba(16,112,2,0.8)"),

                    text=df.university_name)



trace2 = go.Scatter(x=df.world_rank,

                    y=df.teaching,

                    mode="lines+markers",

                    name="teaching",

                    marker=dict(color="rgba(80,26,80,0.8)"),

                    text=df.university_name)



plot_data = [trace1,trace2]

layout=dict(title="citation and teaching vs world rank of top 50 universities",xaxis=dict(title="world rank",ticklen=5,zeroline=False))

fig = dict(data=plot_data,layout=layout)

iplot(fig)
# scatter chart



df2014 = data1[data1.year==2014].iloc[:50,:]

df2015 = data1[data1.year==2015].iloc[:50,:]

df2016 = data1[data1.year==2016].iloc[:50,:]



trace1=go.Scatter(x=df2014.world_rank,

                  y=df2014.citations,

                  mode="markers",

                  name="2014",

                  marker=dict(color="rgba(120,220,20,0.8)"),

                  text=df2014.university_name)



trace2=go.Scatter(x=df2015.world_rank,

                  y=df2015.citations,

                  mode="markers",

                  name="2015",

                  marker=dict(color="rgba(20,220,220,0.8)"),

                  text=df2015.university_name)



trace3=go.Scatter(x=df2016.world_rank,

                  y=df2016.citations,

                  mode="markers",

                  name="2014",

                  marker=dict(color="rgba(220,20,220,0.8)"),

                  text=df2016.university_name)



plot_data = [trace1,trace2,trace3]

layout=dict(title="citations vs world rank of top 50 universities (2014, 2015, 2016)",

            xaxis=dict(title="world rank",ticklen=5, zeroline=False),

            yaxis=dict(title="citation",ticklen=5, zeroline=False))

fig = dict(data=plot_data,layout=layout)

iplot(fig)
# bar plot



df2014 = data1[data1.year==2014].iloc[:3,:]

trace1 = go.Bar(x=df2014.university_name,

                y=df2014.citations,

                name="citations",

                marker=dict(color="rgba(255,174,255,0.5)",

                            line=dict(color="rgb(0,0,0)",width=1.5)),

                text=df2014.country)

trace2 = go.Bar(x=df2014.university_name,

                y=df2014.teaching,

                name="teaching",

                marker=dict(color="rgba(255,255,128,0.5)",

                            line=dict(color="rgb(0,0,0)",width=1.5)),

                text=df2014.country)

plot_data = [trace1,trace2]

layout = go.Layout(barmode="group")

fig = go.Figure(data=plot_data,layout=layout)

iplot(fig)
x = df2014.university_name

trace1 = {"x": x,

          "y": df2014.citations,

          "name": "citations",

          "type": "bar"}



trace2 = {"x": x,

          "y": df2014.teaching,

          "name": "teaching",

          "type": "bar"}



plot_data = [trace1, trace2]



layout = {"xaxis": {"title": "top 3 universities"},

          "barmode": "relative",

          "title": "my title"}



fig = go.Figure(data=plot_data,layout=layout)

iplot(fig)
# pie chart



df2016 = data1[data1.year==2016].iloc[:7,:]

pie1 = df2016.num_students

pie1_list = [float(i.replace(",",".")) for i in pie1.values]

print(pie1_list)

labels = df2016.university_name



fig = {"data": [{"values": pie1_list,

                 "labels": labels,

                 "domain": {"x": [0,.5]},

                 "name": "number of students rates",

                 "hoverinfo": "label+percent+name",

                 "hole": .3,

                 "type": "pie"}],

       "layout": {"title": "my title",

                  "annotations": [{"font": {"size": 20},

                                   "showarrow": False,

                                   "text": "number of students",

                                   "x": 0.20,

                                   "y": 1}]}}

iplot(fig)
# bubble chart



df2016 = data1[data1.year==2016].iloc[:20,:]

num_students_size = [float(i.replace(",",".")) for i in df2016.num_students]

international_score = df2016.international.astype(float)



plot_data = [{"x": df2016.teaching,

              "y": df2016.world_rank,

              "mode": "markers",

              "marker": {"color": international_score,

                         "size": num_students_size,

                         "showscale": True},

              "text": df2016.university_name}]



iplot(plot_data)
# histogram



x2011 = data1.student_staff_ratio[data1.year==2011]

x2012 = data1.student_staff_ratio[data1.year==2012]



trace1 = go.Histogram(x=x2011,

                      opacity=0.75,

                      name="2011",

                      marker=dict(color="rgba(171,50,96,0.6)"))



trace2 = go.Histogram(x=x2012,

                      opacity=0.75,

                      name="2012",

                      marker=dict(color="rgba(12,50,196,0.6)"))



plot_data=[trace1,trace2]

layout=go.Layout(barmode="overlay",

                 title="my title",

                 xaxis=dict(title="students staff ratio"),

                 yaxis=dict(title="counts"))



fig = go.Figure(data=plot_data,layout=layout)

iplot(fig)
# word cloud



x2011 = data1.country[data1.year==2011]

plt.subplots(figsize=(8,8))



wordcloud = WordCloud(background_color="lightblue",

                      width=512,

                      height=384).generate(" ".join(x2011))



plt.imshow(wordcloud)

plt.axis("off")

plt.savefig("graph.png")

plt.show()

# box plot



x2015 = data1[data1.year==2015]



trace0 = go.Box(y=x2015.total_score,

                name="total score of universities in 2015",

                marker=dict(color="rgb(120,12,140)"))



trace1 = go.Box(y=x2015.research,

                name="research of universities in 2015",

                marker=dict(color="rgb(12,102,140)"))



plot_data=[trace0,trace1]

iplot(plot_data)
# scatter plot matrix



import plotly.figure_factory as ff 



df = data1[data1.year==2015]

data2015 = df.loc[:,["research", "international", "total_score"]]

data2015["index"] = np.arange(1,len(data2015)+1)



fig = ff.create_scatterplotmatrix(data2015, diag="box", index="index", colormap="Portland",

                                  colormap_type="cat", height=700, width=700)



iplot(fig)
# inset plot



trace1=go.Scatter(x=data1.world_rank,

                  y=data1.teaching,

                  name="teaching",

                  marker=dict(color="rgba(12,120,250,0.5)"))



trace2=go.Scatter(x=data1.world_rank,

                  y=data1.income,

                  xaxis="x2",

                  yaxis="y2",

                  name="income",

                  marker=dict(color="rgba(212,120,50,0.5)"))



plot_data=[trace1,trace2]

layout=go.Layout(xaxis2=dict(domain=[0.6,0.95],

                             anchor="y2"),

                 yaxis2=dict(domain=[0.6,0.95],

                             anchor="x2"),

                 title="my title")



fig = go.Figure(data=plot_data,layout=layout)

iplot(fig)
# 3D scatter



trace1 = go.Scatter3d(x=data1.world_rank,

                      y=data1.research,

                      z=data1.citations,

                      mode="markers",

                      marker=dict(size=2,

                                  color=5))



plot_data = [trace1]

layout = go.Layout(margin=dict(l=0,

                               r=0,

                               b=0,

                               t=0))



fig = go.Figure(data=plot_data,layout=layout)

iplot(fig)
# multiple subplots



trace1 = go.Scatter(x=data1.world_rank,

                    y=data1.research,

                    mode="markers",

                    name="research")



trace2 = go.Scatter(x=data1.world_rank,

                    y=data1.citations,

                    xaxis="x2",

                    yaxis="y2",

                    name="citations")



trace3 = go.Scatter(x=data1.world_rank,

                    y=data1.income,

                    xaxis="x3",

                    yaxis="y3",

                    name="income")



trace4 = go.Scatter(x=data1.world_rank,

                    y=data1.total_score,

                    xaxis="x4",

                    yaxis="y4",

                    name="total score")



plot_data = [trace1,trace2,trace3,trace4]

layout = go.Layout(xaxis=dict(domain=[0,0.45]),

                   yaxis=dict(domain=[0,0.45]),

                   xaxis2=dict(domain=[0.55,1],anchor="y2"),

                   xaxis3=dict(domain=[0.55,1],anchor="y3"),

                   xaxis4=dict(domain=[0,0.45],anchor="y4"),

                   yaxis2=dict(domain=[0.55,1],anchor="x2"),

                   yaxis3=dict(domain=[0,0.45],anchor="x3"),

                   yaxis4=dict(domain=[0.55,1],anchor="x4"),

                   title="my title")



fig = go.Figure(data=plot_data,layout=layout)

iplot(fig)