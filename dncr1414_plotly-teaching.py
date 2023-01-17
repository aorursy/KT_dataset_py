# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import plotly.plotly as py

from plotly.offline import init_notebook_mode ,iplot 

init_notebook_mode(connected=True)

import plotly.graph_objs as go



from wordcloud import WordCloud



import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
timesData=pd.read_csv("../input/timesData.csv")
timesData.head()
timesData.info()
df=timesData.iloc[:100,:]
df.tail()
trace1=go.Scatter(

x=df.world_rank,

y=df.citations,

mode="lines",

name="citations",

marker=dict(color='rgb(12,125,10,0.8)'),

text=df.university_name

)

trace2=go.Scatter(

x=df.world_rank,

y=df.teaching,

mode="lines+markers",

name="teaching",

marker=dict(color='rgb(125,12,10,0.7)'),

text=df.university_name)



data=[trace1,trace2]

layout=dict(title="",

           xaxis=dict(title='world rank',ticklen=5,zeroline=False))

fig=dict(data=data,layout=layout)

iplot(fig)
df14=timesData[timesData.year==2014].iloc[:100,:]

df15=timesData[timesData.year==2015].iloc[:100,:]

df16=timesData[timesData.year==2016].iloc[:100,:]



trace1=go.Scatter(

    x=df14.world_rank,

    y=df14.citations,

    mode="markers",

    name="2014",

    marker=dict(color='rgb(125,125,25,0.7)'),

    text=df14.university_name)



trace2=go.Scatter(

    x=df15.world_rank,

    y=df15.citations,

    mode="markers",

    name="2015",

    marker=dict(color='rgb(25,125,125)'),

    text=df15.university_name)



trace3=go.Scatter(

    x=df16.world_rank,

    y=df16.citations,

    mode="markers",

    name="2016",

    marker=dict(color='rgb(125,25,125)'),

    text=df16.university_name)



data=[trace1,trace2,trace3]

layout=dict(title="2014-2015-2016",

           xaxis=dict(title="world rank",ticklen=10,zeroline=True),

           yaxis=dict(title="Citations",ticklen=5,zeroline=False))

fig=dict(data=data, layout=layout)

iplot(fig)
df2014=timesData[timesData.year==2014].iloc[:3,:]



trace1=go.Bar(

x=df2014.university_name,

y=df2014.citations,

name="citations",

marker=dict(color='rgb(100,100,75,0.8)'),

text=df2014.country

)



trace2=go.Bar(

    x=df2014.university_name,

    y=df2014.teaching,

    name="teaching",

    marker=dict(color='rgb(45,125,180,0.7)'),

    text=df2014.country)



data=[trace1,trace2]



layout=go.Layout(barmode="group")

fig=go.Figure(data=data,layout=layout)

iplot(fig)
x=df2014.university_name



trace1={

    'x':x,

    'y':df2014.citations,

    'name':'citation',

    'type':'bar'

};



trace2={

    'x':x,

    'y':df2014.teaching,

    'name':'teaching',

    'type':'bar'

};



data=[trace1,trace2];

layout=go.Layout(barmode='relative')

fig=go.Figure(data=data,layout=layout)

iplot(fig)
df2016=timesData[timesData.year==2016].iloc[:7,:]

pie1=df2016.num_students

pie1_list=[float(each.replace(',','.')) for each in df2016.num_students]

labels=df2016.university_name



fig={

    "data":[

        {

        "values":pie1_list,

        "labels":labels,

        "domain":{"x":[0,.5]},

        "name":"Number of Students Rates",

        "hoverinfo":"label+percent+name",

        "hole":0.2,

        "type":"pie"

    },],

    "layout":{

        "title":"Universities Number of Students Rates",

        "annotations":[

            {

                "font":{"size":20},

                "showarrow":False,

                "text":"Number of Stundets",

                "x":0.20,

                "y":1

            }

        ]

    }

}

iplot(fig)
df2016=timesData[timesData.year==2016].iloc[:20,:]

num_student_size=[float(each.replace(',','.'))for each in df2016.num_students]

international_color=[float(each) for each in df2016.international]



data=[

    {

        'x':df2016.world_rank,

        'y':df2016.teaching,

        'mode':'markers',

        'marker':{

            'color':international_color,

            'size':num_student_size,

            'showscale':True

            

        },

        'text':df2016.university_name

    }

]

iplot(data)
x2013=timesData.student_staff_ratio[timesData.year==2013]

x2014=timesData.student_staff_ratio[timesData.year==2014]



trc1=go.Histogram(

    x=x2013,

    opacity=0.7,

    name="2013",

    marker=dict(color='rgba(125,7,21,0.8)')

    

)



trc2=go.Histogram(

    x=x2014,

    opacity=0.6,

    name="2014",

    marker=dict(color='rgba(21,25,147,0.2)')

)

data=[trc1,trc2]

layout=go.Layout(barmode="overlay",

                title="Students-Staff Ratio 2013-2014",

                xaxis=dict(title='Students-Staff Ratio'),

                yaxis=dict(title="Count"))

fig=go.Figure(data=data,layout=layout)

iplot(fig)
x2016=timesData.country[timesData.year==2016]

plt.subplots(figsize=(8,8))

wordCloud=WordCloud(

background_color='white',

width=512,

height=380,

).generate(" ".join(x2016))

plt.imshow(wordCloud)

plt.axis=('off')



x2015=timesData[timesData.year==2015]



trace1=go.Box(

    y=x2015.total_score,

    name="Total Score of Universities in 2015",

    marker=dict(color='rgba(36,25,14,0.8)')

)



trace2=go.Box(

    y=x2015.research,

    name="research of Universities in 2015",

    marker=dict(color='rgba(14,25,136,0.6)')

)

data=[trace1,trace2]

iplot(data)
import plotly.figure_factory as ff

dataframe=timesData[timesData.year==2015]

data2015=dataframe.loc[:,["research","international","total_score"]]

data2015["index"]=np.arange(1,len(data2015)+1)



fig=ff.create_scatterplotmatrix(data2015,diag="box",index="index",colormap="Portland",

                                colormap_type='cat',

                               height=700, width=800)

iplot(fig)
trace1=go.Scatter(

x=dataframe.world_rank,

y=dataframe.teaching,

name="teaching",

marker=dict(color='rgba(210,10,15,0.7)')

)



trace2=go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.income,

    xaxis="x2",

    yaxis="y2",

    name="income",

    marker=dict(color='rgba(15,210,10,0.8)')

)

data=[trace1,trace2]



layout=go.Layout(

    xaxis2 =dict(

        domain=[0.6,0.95],

    anchor='y2'

    ),

    yaxis2=dict(

    domain=[0.6,0.95],

    anchor='x2'),

    

    title="income and Teaching vs World Rank of Universities"

)



fig=go.Figure(data=data, layout=layout)

iplot(fig)
trace1=go.Scatter3d(

    x=dataframe.world_rank,

    y=dataframe.research,

    z=dataframe.citations,

    mode="markers",

    marker=dict(

        size=5,

        color="green",

    )

)



data=[trace1]



layout=go.Layout(margin=dict(

l=0,

r=0,

b=0,

t=0))



fig=go.Figure(data=data, layout=layout)



iplot(fig)
trc1=go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.research,

    name="research"

)



trc2=go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.citations,

    xaxis="x2",

    yaxis="y2",

    name="citations"

)

trc3=go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.income,

    xaxis="x3",

    yaxis="y3",

    name="income"



)



trc4=go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.total_score,

    xaxis="x4",

    yaxis="y4",

    name="total_score"

)



data=[trc1,trc2,trc3,trc4]



layout=go.Layout(

    xaxis=dict(

    domain=[0,0.45]

    ),

    

    yaxis=dict(domain=[0,0.45]),

    

    xaxis2=dict(domain=[0.55,1]),

    

    xaxis3=dict(domain=[0,0.45],

                anchor="y3"),

    

    xaxis4=dict(domain=[0.55,1],

               anchor="y4"),

    

    yaxis2=dict(domain=[0,0.45],

               anchor="x2"),

    



    yaxis3=dict(domain=[0.55,1]),

    



    yaxis4=dict(domain=[0.55,1],

               anchor="x4"),

    

    title="Research, Citation, İncome and Total Score vs World Rank"

    

)

fig=go.Figure(data=data,layout=layout)

iplot(fig)