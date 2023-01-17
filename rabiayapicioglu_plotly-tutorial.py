# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#from plotly.offline import plot # this solves the problem of PlotRequestError:No Message

#import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

from plotly.graph_objs import *

init_notebook_mode()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

timesData=pd.read_csv("../input/timesData.csv")
timesData.info()
timesData.head(10)


#from  plotly.offline import plot

#import plotly.graph_objs as go



# these two lines allow your code to show up in a notebook

import plotly.graph_objs as go

# Creating trace1

df2016=timesData[ timesData['year']==2016].iloc[:100,:]

trace1 = go.Scatter(

                    x = df2016.world_rank,

                    y = df2016.citations,

                    mode = "lines",

                    name = "citations",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= df2016.university_name)

# Creating trace2

trace2 = go.Scatter(

                    x = df2016.world_rank,

                    y = df2016.teaching,

                    mode = "lines+markers",

                    name = "teaching",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                    text= df2016.university_name)

data = [trace1, trace2]

layout = dict(title = 'Citation and Teaching vs World Rank of Top 100 Universities',

              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Values',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)

df2014=timesData[ timesData['year']==2014].iloc[:100,:]

df2015=timesData[ timesData['year']==2015].iloc[:100,:]

df2016=timesData[ timesData['year']==2016].iloc[:100,:]



trace1=go.Scatter(

                  x=df2014.world_rank,

                  y=df2014.citations,

                  mode="markers",

                  name="2015",

                  marker=dict( color='cyan'),

                  text=df2014.university_name )



trace2=go.Scatter(

                  x=df2015.world_rank,

                  y=df2015.citations,

                  mode="markers",

                  marker=dict( color='rgba(255,128,2,0.8)'),

                  text=df2015.university_name )



trace3=go.Scatter(

                  x=df2016.world_rank,

                  y=df2016.citations,

                  mode="markers",

                  marker=dict( color='rgba(255,128,200,0.8)'),

                  text=df2016.university_name )



data_combined=[trace1,trace2,trace3]



layout=dict( title='Citation vs world rank top 100 universities with 2014,2015 and 2016',

             xaxis=dict( title="World Rank",ticklen=5,zeroline=False ),

             yaxis=dict( title="Citation",ticklen=5,zeroline=False ),)

           

fig=dict( data=data_combined,layout=layout)

iplot( fig )

    















# prepare data frames

df2014 = timesData[timesData.year == 2014].iloc[:3,:]

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
df2014=timesData[ timesData.year==2014 ].iloc[:3,:]



import plotly.graph_objs as go



x=df2014.university_name



trace1={ 

        'x':x,

        'y':df2014.citations,

        'name':'citation',

        'type':'bar' }

trace2={

        'x':x,

        'y':df2014.teaching,

        'name':'teaching',

        'type':'bar'

                     }

data=[trace1,trace2]

layout={

         'xaxis':{'title':'Top 3 universities'},

         'barmode':'relative',

         'title':'citations and teaching of top 3 unis in 2014'

}



fig=go.Figure( data=data,layout=layout )

iplot( fig )

# data preparation

df2016 = timesData[timesData.year == 2016].iloc[:7,:]

pie1 = df2016.num_students

pie1_list = [float(each.replace(',', '.')) for each in df2016.num_students]  # str(2,4) => str(2.4) = > float(2.4) = 2.4

labels = df2016.university_name

# figure

fig = {

  "data": [

    {

      "values": pie1_list,

      "labels": labels,

      "domain": {"x": [0, .5]},

      "name": "Number Of Students Rates",

      "hoverinfo":"label+percent+name",

      "hole": .3,

      "type": "pie"

    },],

  "layout": {

        "title":"Universities Number of Students rates",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": True,

              "text": "Number of Students",

                "x": 0.20,

                "y": 1

            },

        ]

    }

}

iplot(fig)

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
x2011=timesData.student_staff_ratio[ timesData.year==2011 ]

x2012=timesData.student_staff_ratio[ timesData.year==2012 ]



trace1=go.Histogram(

   x=x2011,

   opacity=0.75,

   name='2011',

   marker=dict( color='rgba(171,50,96,0.6)'))

trace2=go.Histogram(

   x=x2012,

   opacity=0.75,

   name='2012',

   marker=dict( color='rgba(12,50,196,0.6)') )

data=[trace1,trace2]

layout=go.Layout( barmode='overlay',title='students-staff rati in 2011 and 2012',

                  xaxis=dict( title='students-staff ratio'),

                  yaxis=dict( title='Count')

                )

fig=go.Figure( data=data,layout=layout )

iplot( fig )
import matplotlib.pyplot as plt

from wordcloud import WordCloud 

x2011=timesData.country[ timesData.year==2011 ]

plt.subplots( figsize=(8,8))

wordcloud=WordCloud(

                       background_color='white',

                       width=512,

                       height=384).generate(" ".join(x2011))

plt.imshow( wordcloud )

plt.axis('off')



plt.show()
x2015 = timesData[timesData.year == 2015]



trace0 = go.Box(

    y=x2015.total_score,

    name = 'total score of universities in 2015',

    marker = dict(

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

dataframe = timesData[timesData.year == 2015]

data2015 = dataframe.loc[:,["research","international", "total_score"]]

data2015["index"] = np.arange(1,len(data2015)+1)

# scatter matrix

fig = ff.create_scatterplotmatrix(data2015, diag='box', index='index',colormap='Portland',

                                  colormap_type='cat',

                                  height=700, width=700)

iplot(fig)
# first line plot

trace1 = go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.teaching,

    name = "teaching",

    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

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



fig = go.Figure(data=data, layout=layout)

iplot(fig)
# create trace 1 that is 3d scatter

trace1 = go.Scatter3d(

    x=dataframe.world_rank,

    y=dataframe.research,

    z=dataframe.citations,

    mode='markers',

    marker=dict(

        size=10,

        color='rgb(255,0,0)',  

        

        # set color to an array/list of desired values      

    )

)



data = [trace1]

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0  

    ),

     

    

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import numpy as np



ax3d = plt.figure().gca(projection='3d')



arrayx = np.array([[0.7], [7.1], [7.5], [0.6], [0.5], [0.00016775708773695687]])

arrayy = np.array([[0.1], [2], [3], [6], [5], [16775708773695687]])

arrayz = np.array([[1], [2], [3], [4], [5], [6]])



labels = ['one', 'two', 'three', 'four', 'five', 'six']



arrayx = arrayx.flatten()

arrayy = arrayy.flatten()

arrayz = arrayz.flatten()



ax3d.scatter(arrayx, arrayy, arrayz)



#give the labels to each point

for x_label, y_label, z_label, label in zip(arrayx, arrayy, arrayz, labels):

    ax3d.text(x_label, y_label, z_label, label)



plt.title("Data")

plt.show()
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