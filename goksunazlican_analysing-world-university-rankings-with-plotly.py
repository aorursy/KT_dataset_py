# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# word cloud library

from wordcloud import WordCloud



# matplotlib

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
timesData = pd.read_csv('../input/timesData.csv')
timesData.info()
timesData.head()
df=timesData.iloc[:100:]



import plotly.graph_objs as go



trace1= go.Scatter(

                    x=df.world_rank,

                    y=df.citations,

                    mode="lines",

                    name="citations",

                    marker=dict(color='rgb(16,112,4,0.8)'),

                    text=df.university_name

                    )



trace2 = go.Scatter(

                    x=df.world_rank,

                    y=df.teaching,

                    mode="lines+markers",

                    name="teaching",

                    marker=dict(color='rgb(100,2,5,0.5)'),

                    text=df.university_name

                    )



data=[trace1,trace2]#liste

layout=dict(title='Citation and Teaching vs World Rank of Top 100 Universities',xaxis=dict(title='World Rank',ticklen=5,zeroline=False))

fig=dict(data=data,layout=layout)

iplot(fig)
df2011=timesData[timesData.year==2011].iloc[:100,:]

df2012=timesData[timesData.year==2012].iloc[:100,:]

df2013=timesData[timesData.year==2013].iloc[:100,:]





import plotly.graph_objs as go



trace1 = go.Scatter(

                    x=df2011.world_rank,

                    y=df2011.citations,

                    mode="markers",

                    name="2011",

                    marker=dict(color='rgb(255,10,1,0.8)'),

                    text=df2011.university_name

                    )





trace2 = go.Scatter(

                    x=df2012.world_rank,

                    y=df2012.citations,

                    mode="markers",

                    name="2012",

                    marker=dict(color='rgb(20,210,1,0.8)'),

                    text=df2012.university_name

                    )



trace3 = go.Scatter(

                    x=df2013.world_rank,

                    y=df2013.citations,

                    mode="markers",

                    name="2013",

                    marker=dict(color='rgb(20,2,221,0.8)'),

                    text=df2013.university_name

                    )



data=[trace1,trace2,trace3]



layout=dict(title='Citation vs World Rank of Top 100 Universities with 2011,2012,2013',

           xaxis=dict(title='World Rank',ticklen=5,zeroline=False),

            yaxis=dict(title='Citation',ticklen=5,zeroline=False)

           )



fig=dict(data=data,layout=layout)

iplot(fig)
df2011=timesData[timesData.year==2011].iloc[:3,:]

df2011



import plotly.graph_objs as go

trace1= go.Bar(

                x=df2011.university_name,

                y=df2011.citations,

                name='citations',

                marker=dict(color='rgba(255,8,95,0.5)',

                           line=dict(color='rgb(0,0,0)',width=1.5))  ,

                text=df2011.country

                )



trace2=go.Bar(

                x=df2011.university_name,

                y=df2011.teaching,

                name='teaching',

                marker=dict(color='rgba(0,255,85,0.5)',

                            line=dict(color='rgb(25,2,158)',width=1.5)),

                text=df2011.country

            

)

data=[trace1,trace2]

layout=go.Layout(barmode='group')

fig=go.Figure(data=data,layout=layout)

iplot(fig)
import plotly.graph_objs as go



x=df2011.university_name



trace1={

    'x':x,

    'y':df2011.citations,

    'name':'citations',

    'type':'bar'

};



trace2={

    'x':x,

    'y':df2011.teaching,

    'name':'teaching',

    'type':'bar'

};



data=[trace1,trace2];



layout={

    'xaxis':{'title':'Top 3 Universities'},

    'barmode':'relative',

    'title':'citations and teaching of top 3 universities in 2011'

};



fig=go.Figure(data=data,layout=layout)

iplot(fig)
df2014=timesData[timesData.year==2014].iloc[:7,:]

pie1=df2014.num_students

pie1_list=[float(each.replace(',','.')) for each in df2014.num_students] 

labels=df2014.university_name



fig={

    

    "data":[

            {

               "values":pie1_list,

                "labels":labels,

                "domain":{"x":[0, .5]},

                "name":"number of students rates",

                "hoverinfo":"label+percent+name",

                "hole":.2 ,

                "type":"pie"

            },

    ],

    

    "layout":{

        "title":"Universities Number of Students",

        "annotations":[

            { "font":{"size":20},

              "showarrow": False,

              "text":"Number of Students",

                "x":0.20,

                "y":1

            },

        ]

            

    }

}

iplot(fig)