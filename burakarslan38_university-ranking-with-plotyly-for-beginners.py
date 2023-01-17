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



# Any results you write to the current directory are saved as output.
timesdata = pd.read_csv("../input/timesData.csv")
timesdata.info() # some informations about data
timesdata.head(3) # first 3 universties
#go is plotly

df=timesdata.iloc[:100]



trace1 = go.Scatter(

    x=df.world_rank,

    y=df.citations,

    mode = "lines+markers",

    name="Citiations",

    marker = dict(color="rgba(250,0,0,0.6)"),

    text = df.university_name

)



trace2 = go.Scatter(

x=df.world_rank,

y=df.teaching,

mode="lines+markers",

name="Teaching",

marker=dict(color="rgba(0,0,250,0.6)"),

text= df.university_name

)



data=[trace1,trace2]

layout=dict(title="Citation and Teaching",

           xaxis=dict(title="World Rank",ticklen=3,zeroline= False)

           )

fig = dict(data=data,layout=layout)

iplot(fig)
df2011 = timesdata[timesdata.year==2011].iloc[:100]

df2013 = timesdata[timesdata.year==2013].iloc[:100]

df2015 = timesdata[timesdata.year==2015].iloc[:100]



trace2011 = go.Scatter(

x=df2011.world_rank,

y=df2011.citations,

mode="markers",

name="in 2011",

marker=dict(color="rgba(0,0,0,0.6)"),

text= df.university_name)



trace2013 = go.Scatter(

x=df2013.world_rank,

y=df2013.citations,

mode="markers",

name="in 2013",

marker=dict(color="rgba(35,56,190,0.6)"),

text= df.university_name)



trace2015 = go.Scatter(

x=df2015.world_rank,

y=df2015.citations,

mode="markers",

name="in 2015",

marker=dict(color="rgba(255,2,69,0.6)"),

text= df.university_name)



data=[trace2011,trace2013,trace2015]



layout=dict(title="Citations in 2011,2013 and 2015",

           xaxis=dict(title= 'World Rank',ticklen= 5))

fig = dict(data=data,layout=layout)

iplot(fig)
# prepare data frames

df2014 = timesdata[timesdata.year == 2014].iloc[:3]

# import graph objects as "go"

import plotly.graph_objs as go

# create trace1 

trace1 = go.Bar(

                x = df2014.university_name,

                y = df2014.citations,

                name = "citations",

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df2014.country)

# create trace2 

trace2 = go.Bar(

                x = df2014.university_name,

                y = df2014.teaching,

                name = "teaching",

                marker = dict(color = 'rgba(255, 255, 128, 0.5)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df2014.country)

data = [trace1, trace2]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
df2016 = timesdata[timesdata.year == 2016].iloc[:7,:]

pie1 = df2016.num_students

pie1_list = [float(each.replace(",",".")) for each in df2016.num_students]

labels = df2016.university_name



fig = {

  "data": [

    {"values": pie1_list,

      "labels": labels,

      "domain": {"x": [0, .5]},

      "name": "Number Of Students Rates",

      "hoverinfo":"label+percent+name",

      "hole": .2,

      "type": "pie"},],

  "layout": {

        "title":"Universities Number of Students rates",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

              "text": "Number of Students",

                "x": 0.50,

                "y": 1},]

  }

}

iplot(fig)

timesdata.head(5

              )
d2011=timesdata.student_staff_ratio[timesdata.year==2011]

d2012=timesdata.student_staff_ratio[timesdata.year==2012]

trace1 = go.Histogram(

    x=d2011,

    opacity=0.75,

    name = "2011",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))



trace1 = go.Histogram(

    x=d2011,

    opacity=0.55,

    name = "2011",

    

    marker=dict(color='rgba(120, 150, 226, 250)'))

trace2 = go.Histogram(

    x=d2012,

    opacity=0.75,

    name = "2012",

    marker=dict(color='rgba(171, 0, 96, 0.6)'))

    

data=[trace1,trace2]

layout = go.Layout ( barmode="overlay",

                   title = "Students and Staff Ratio in 2011-2012",

                   )

fig = go.Figure(data=data,layout=layout)

iplot(fig)
x2011 = timesdata.country[timesdata.year==2015]

plt.subplots(figsize=(10,10))

wordcloud = WordCloud(

    background_color="white",

    width = 512,

    height=384,

).generate(" ".join(x2011))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()



import plotly.figure_factory as ff

dataframe=timesdata[timesdata.year ==2015]

data2015=dataframe.loc[:,["research","international", "total_score"]]

data2015["index"]=np.arange(1,len(data2015)+1)

fig=ff.create_scatterplotmatrix(data2015, diag='box', index='index',colormap='Portland',

                                  colormap_type='cat',

                                  height=700, width=700)

iplot(fig)
trace1 = go.Scatter3d(

    x=dataframe.world_rank,

    y=dataframe.research,

    z=dataframe.citations,

    mode='markers',

    marker=dict(

        size=10,

        color='cyan',                    

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