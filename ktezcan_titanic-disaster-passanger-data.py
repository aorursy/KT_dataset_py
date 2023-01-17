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
titanicdata = pd.read_csv("../input/titanic-extended/full.csv")



# prepare data frame

df = titanicdata.iloc[:100,:]



# import graph objects as "go"

import plotly.graph_objs as go



# Creating trace1

trace1 = go.Scatter(

                    x = df.Name                                                                                                                                                                      ,

                    y = df.Age_wiki                                     ,

                    mode = "lines",

                    name = "Age           ",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= df.Destination                                                                                                      )

# Creating trace2

trace2 = go.Scatter(

                    x = df.Name                                                                                                                                                                  ,

                    y = df.Survived                                                                                                                                                                                                                                                                                                     ,

                    mode = "lines+markers",

                    name = "Survival Status                   ",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                    text= df.Destination                                                                                                       )

data = [trace1, trace2]

layout = dict(title =  "Passangers' Name, Age, Destination & Survival Status",

              xaxis= dict(title= '',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
# prepare data frames

dfClass1 = titanicdata[titanicdata.Class == 1].iloc[:100,:]

dfClass2 = titanicdata[titanicdata.Class == 2].iloc[:100,:]

dfClass3 = titanicdata[titanicdata.Class == 3].iloc[:100,:]

# import graph objects as "go"

import plotly.graph_objs as go

# creating trace1

trace1 =go.Scatter(

                    x = dfClass1.PassengerId                                                                                                                                                   ,

                    y = dfClass1.Age_wiki                                                                                                                          ,

                    mode = "markers",

                    name = "First",

                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),

                    text= dfClass1.Survived                                                                         )

# creating trace2

trace2 =go.Scatter(

                    x = dfClass2.PassengerId                                                                                                                              ,

                    y = dfClass2.Age_wiki                                                                                                                         ,

                    mode = "markers",

                    name = "Second",

                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),

                    text= dfClass2.Survived                                                                         )

# creating trace3

trace3 =go.Scatter(

                    x = dfClass3.PassengerId                                                                                                                                            ,

                    y = dfClass3.Age_wiki                                                                                                                ,

                    mode = "markers",

                    name = "Third                    ",

                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),

                    text= dfClass3.Survived                                                                                )

data = [trace1, trace2, trace3]

layout = dict(title = 'Survival rate According to Class',

              xaxis= dict(title= 'PassengerId                      ',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Age',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
# prepare data frames

df_survived_1 = titanicdata[titanicdata.Survived                                           == 1            ].iloc[:20,:]

# import graph objects as "go"

import plotly.graph_objs as go

# create trace1 

trace1 = go.Bar(

                x = df_survived_1.Name                                                                                                                                      ,

                y = df_survived_1.Age_wiki                                                                                                                          ,

                name = "Age                ",

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df_survived_1.Sex                                                                                               )

# create trace2 

trace2 = go.Bar(

                x = df_survived_1.Name                                                                                                                                      ,

                y = df_survived_1.Fare                                                                                                      ,

                name = "Fare                       ",

                marker = dict(color = 'rgba(255, 255, 128, 0.5)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df_survived_1.Sex                                                                                               )

data = [trace1, trace2]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
# prepare data frames

df_survived_1 = titanicdata.iloc[:20,:]

# import graph objects as "go"

import plotly.graph_objs as go



x = titanicdata.Name



trace1 = {

  'x': x,

  'y': df_survived_1.Pclass,

  'name': 'Pclass',

  'type': 'bar'

};

trace2 = {

  'x': x,

  'y': df_survived_1.SibSp,

  'name': 'SibSp',

  'type': 'bar'

};

data = [trace1, trace2];

layout = {

  'xaxis': {'title': ''},

  'barmode': 'relative',

  'title': 'Pclass & SibSp Numbers of First Twenty Passanger '

};

fig = go.Figure(data = data, layout = layout)

iplot(fig)
df_survived_1 = titanicdata[titanicdata.Survived== 1] 

labels = df_survived_1.Class

pie1_list=df_survived_1.Survived

# figure

fig = {

  "data": [

    {

      "values": pie1_list,

      "labels": labels,

      "domain": {"x": [0, .5]},

      "name": "Fare Rates",

      "hoverinfo":"label+percent+name",

      "hole": .3,

      "type": "pie"

    },],

  "layout": {

        "title":"Classes of Survived Passangers",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

              "text": "",

                "x": 0.20,

                "y": 1

            },

        ]

    }

}

iplot(fig)
# data preparation

df_survived_1 = titanicdata[titanicdata.Survived                                           == 1            ].iloc[:20,:]



pie1_list=df_survived_1.Fare

international_color = [float(each) for each in df_survived_1.Fare]

data = [

    {

        'y': df_survived_1.Age_wiki,

        'x': df_survived_1.Name,

        'mode': 'markers',

        'marker': {

            'color': international_color,

            'size': pie1_list,

            'showscale': True

        },

        "text" :  df_survived_1.Name    

    }

]

iplot(data)
df_survived_1 = titanicdata.Pclass [titanicdata.Survived    == 1  ]        

df_survived_0 = titanicdata.Pclass [titanicdata.Survived    == 0   ]       

trace1 = go.Histogram(

    x=df_survived_1,

    opacity=0.75,

    name = "Survived",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))

trace2 = go.Histogram(

    x=df_survived_0,

    opacity=0.75,

    name = "Dead",

    marker=dict(color='rgba(12, 50, 196, 0.6)'))



data = [trace1, trace2]

layout = go.Layout(barmode='overlay',

                   title=' Surviving Ratio According to Class',

                   xaxis=dict(title='Class'),

                   yaxis=dict( title='Passangers'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
# data prepararion

df_survived_1 = titanicdata.Name                  [titanicdata.Survived    == 1  ]

plt.subplots(figsize=(8,8))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(df_survived_1))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()
df_survived_1 = titanicdata[titanicdata.Survived== 1  ]

boolean = titanicdata.Fare < 60

titanicdata[boolean]

trace0 = go.Box(

    y=df_survived_1.Age_wiki       ,

    name = 'Age',

    marker = dict(

        color = 'rgb(12, 12, 140)',

    )

)

trace1 = go.Box(

    y=titanicdata[boolean].Fare,

    name = 'Fare',

    marker = dict(

        color = 'rgb(12, 128, 128)',

    )

)

data = [trace0, trace1]

iplot(data)


# import figure factory

import plotly.figure_factory as ff

# prepare data

dataframe = titanicdata[titanicdata.Survived== 0  ]

data1 = dataframe.loc[:,["Age_wiki","WikiId","Fare"]]

data1["index"] = np.arange(1,len(data1)+1)

# scatter matrix

fig = ff.create_scatterplotmatrix(data1, diag='box', index='index',colormap='Portland',

                                  colormap_type='cat',

                                  height=700, width=700)

iplot(fig)
# first line plot

trace1 = go.Scatter(

    x=dataframe.PassengerId                   ,

    y=dataframe.Age_wiki       ,

    name = "Age",

    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

)

# second line plot

trace2 = go.Scatter(

    x=dataframe.PassengerId                   ,

    y=dataframe.Fare,

    xaxis='x2',

    yaxis='y2',

    name = "Fare",

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

    title = 'Age and Fare of Dead Passangers'



)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
# create trace 1 that is 3d scatter

trace1 = go.Scatter3d(

    x=dataframe.PassengerId        ,

    y=dataframe.Age_wiki       ,

    z=dataframe.Fare           ,

    mode='markers',

    marker=dict(

        size=10,

        color='rgb(255,0,0)',                # set color to an array/list of desired values      

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

    x=dataframe.PassengerId           ,

    y=dataframe.Embarked                              ,

    name = "Embarked"

)

trace2 = go.Scatter(

    x=dataframe.PassengerId           ,

    y=dataframe.Age_wiki       ,

    xaxis='x2',

    yaxis='y2',

    name = "Age"

)

trace3 = go.Scatter(

    x=dataframe.PassengerId           ,

    y=dataframe.Fare           ,

    xaxis='x3',

    yaxis='y3',

    name = "Fare"

)

trace4 = go.Scatter(

    x=dataframe.PassengerId           ,

    y=dataframe.Boarded                            ,

    xaxis='x4',

    yaxis='y4',

    name = "Boarded        "

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

    title = 'Embarked,Age,Fare and Boarded'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)