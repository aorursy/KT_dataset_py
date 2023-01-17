

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# plotly

# import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

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
timesData = pd.read_csv("../input//world-university-rankings/timesData.csv")
timesData.info()
timesData.head(10)
df = timesData.iloc[:100,:]



import plotly.graph_objs as go



trace1 = go.Scatter(

    x = df.world_rank,

    y = df.citations,

    mode = "lines",

    name = "citations",

    marker = dict(color = "blue"),

    text = df.university_name

)

trace2 = go.Scatter(

    x = df.world_rank,

    y = df.teaching,

    mode = "lines+markers",

    name = "teaching",

    marker = dict(color  = "purple"),

    text = df.university_name

)

data = [trace1,trace2]

layout = dict(

    title = "Citation and Teaching",

    xaxis = dict(title = "World Rank",ticklen = 5,zeroline = False)

)

fig = dict(data = data,layout = layout)

iplot(fig)
df = timesData.iloc[:100,:]



trace1 = go.Scatter(

    x = df.world_rank,

    y = df.citations,

    text = df.university_name,

    name = "citations",

    marker = dict(color = "red"),

    mode = "markers"

)

trace2 = go.Scatter(

    x = df.world_rank,

    y = df.teaching,

    text = df.university_name,

    name = "teaching",

    marker = dict(color = "black"),

    mode = "markers"

)

data = [trace1,trace2]

layout = dict(

    title = "Citation and Teaching vs World Rank of Top 100 Universitie",

    xaxis = dict(title = "World Rank",zeroline = False,ticklen = 5),

    yaxis = dict(title = "Citation and Teaching",zeroline = False,ticklen = 5)

)

fig = dict(layout = layout,data=data)

iplot(fig)
df2014 = timesData[timesData.year == 2014].iloc[:3,:]

trace1 = go.Bar(

    x = df2014.university_name,

    y = df2014.citations,

    name = "Citations",

    marker = dict(color = "purple",line = dict(color = "blue",width = 1.5)),

    text = df2014.country

)

trace2 = go.Bar(

    x = df2014.university_name,

    y = df2014.teaching,

    name = "teaching",

    marker = dict(color = "blue",line = dict(color = "red",width = 1.5)),

    text = df2014.country

)

data = [trace1,trace2]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data,layout = layout)

iplot(fig)
df2014 = timesData[timesData.year == 2014].iloc[:3,:]



x = df2014.university_name



trace1 = {

    'x':x,

    'y':df2014.citations,

    'name':'citation',

    'type':'bar'

};

trace2 = {

    'x':x,

    'y':df2014.teaching,

    'name':'teaching',

    'type':'bar'

};

data = [trace1,trace2];

layout = {

    'xaxis':{'title':'Top 3 Universities'},

    'barmode':'relative',

    'title':'citations and teaching of top 3 universities in 2014'

};

fig = go.Figure(data = data,layout = layout)

iplot(fig)
df2016 = timesData[timesData.year == 2016].iloc[:7,:]

pie1 = df2016.num_students

pie1_list = [float(each.replace(',','.')) for each in df2016.num_students]

labels = df2016.university_name



fig = {

    "data":[

        {

            "values":pie1_list,

            "labels":labels,

            "domain":{"x":[0,.5]},

            "name":"Number Of Students Rates",

            "hoverinfo":"label+percent+name",

            "hole":.3,

            "type":"pie"

        },

    ],

    "layout":{

        "title":"Universities Number of Students rates",

        "annotations":[

            {

                "font":{"size":20},

                "showarrow":False,

                "text":"Number of Students",

                "x":0.20,

                "y":1

            },

        ]

    }

}

iplot(fig)
df2016 = timesData[timesData.year == 2016].iloc[:20,:]

num_students_size = [float(each.replace(',','.')) for each in df2016.num_students]

international_color = [float(each) for each in df2016.international]

data = [

    {

        'y':df2016.teaching,

        'x':df2016.world_rank,

        'mode':'markers',

        'marker':{

            'color':international_color,

            'size':num_students_size,

            'showscale': True

        },

        'text':df2016.university_name

    }

]

iplot(data)
x2011 = timesData.student_staff_ratio[timesData.year == 2011]

x2012 = timesData.student_staff_ratio[timesData.year == 2012]

trace1 = go.Histogram(

    x = x2011,

    opacity = 0.75,

    name = "2011",

    marker = dict(color = "blue")

)

trace2 = go.Histogram(

    x = x2012,

    opacity = 0.75,

    name = "2012",

    marker = dict(color = "purple")

)

data = [trace1,trace2]

layout = go.Layout(barmode = "overlay",

                  title = "Students-Staff Ratio in 2011 and 2012",

                  xaxis = dict(title = "Students-Staff Ratio"),

                  yaxis = dict(title = "Count"),

                  )

fig = go.Figure(data = data,layout = layout)

iplot(fig)

x2011 = timesData.country[timesData.year == 2011]

plt.subplots(figsize = (8,8))

wordcloud = WordCloud(

    background_color = "white",

    width = 512,

    height = 384,

).generate(" ".join(x2011))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')

plt.show()
x2015 = timesData[timesData.year == 2015]

trace0 = go.Box(

    y = x2015.total_score,

    name = "Total Score of Universities in 2015",

    marker = dict(color = "red")

)

trace1 = go.Box(

    y = x2015.research,

    name = "Research of universities in 2015",

    marker = dict(color = "blue")

)

data = [trace0,trace1]

iplot(data)
import plotly.figure_factory as ff

dataframe = timesData[timesData.year == 2015]

data2015 = dataframe.loc[:,["research","international","total_score"]]

data2015["index"] = np.arange(1,len(data2015)+1)

fig = ff.create_scatterplotmatrix(data2015,diag = "box",index = "index",colormap = "Portland",

                                 colormap_type = "cat",height = 700,width = 700)

iplot(fig)
trace1 = go.Scatter(

    x = dataframe.world_rank,

    y = dataframe.teaching,

    name = "teaching",

    marker = dict(color = "green")

)

trace2 = go.Scatter(

    x = dataframe.world_rank,

    y = dataframe.income,

    xaxis = 'x2',

    yaxis = 'y2',

    name = "income",

    marker = dict(color = "yellow")

)

data = [trace1,trace2]

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

fig = go.Figure(data = data,layout = layout)

iplot(fig)