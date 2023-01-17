# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



#plotly library

import plotly.plotly as py

from plotly.offline import init_notebook_mode ,iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



#word cloud

from wordcloud import WordCloud



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
winemag130_data = pd.read_csv("../input/winemag-data-130k-v2.csv")

winemag130_data.rename( columns={'Unnamed: 0':'ID'}, inplace=True )



winemag150_data = pd.read_csv("../input/winemag-data_first150k.csv")

winemag150_data.rename( columns={'Unnamed: 0':'ID'}, inplace=True )
winemag130_data.info()
winemag130_data.head(10)
#Plotly line Plot



df = winemag130_data.iloc[:100,:]



# import graph objects as "go"

import plotly.graph_objs as go



# Creating trace1

trace1 = go.Scatter(

                    x = df.ID,

                    y = df.points,

                    mode = "lines",

                    name = "points",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= df.variety)

# Creating trace2

trace2 = go.Scatter(

                    x = df.ID,

                    y = df.price,

                    mode = "lines+markers",

                    name = "price",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                    text= df.variety)

data = [trace1, trace2]

layout = dict(title = 'Points and Price vs ID of Top 100 Variety',

              xaxis= dict(title= 'ID',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)

#unique columns list

winemag130_data["points"].unique()
#Plotly scatter plot



df87 = winemag130_data[winemag130_data.points == 87].iloc[:50,:]

df90 = winemag130_data[winemag130_data.points == 90].iloc[:50,:]

df93 = winemag130_data[winemag130_data.points == 93].iloc[:50,:]

df96 = winemag130_data[winemag130_data.points == 96].iloc[:50,:]

df99 = winemag130_data[winemag130_data.points == 99].iloc[:50,:]



df = winemag130_data.iloc[:100,:]



import plotly.graph_objs as go

# creating trace1

trace1 =go.Scatter(

                    x = df87.points,

                    y = df.price,

                    mode = "markers",

                    name = "87",

                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),

                    text= df87.variety)

# creating trace2

trace2 =go.Scatter(

                    x = df90.points,

                    y = df.price,

                    mode = "markers",

                    name = "90",

                    marker = dict(color = 'rgba(240, 128, 255, 0.8)'),

                    text= df90.variety)

# creating trace3

trace3 =go.Scatter(

                    x = df93.points,

                    y = df.price,

                    mode = "markers",

                    name = "93",

                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),

                    text= df90.variety)

# creating trace4

trace4 =go.Scatter(

                    x = df96.points,

                    y = df.price,

                    mode = "markers",

                    name = "96",

                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),

                    text= df96.variety)

# creating trace5

trace5 =go.Scatter(

                    x = df99.points,

                    y = df.price,

                    mode = "markers",

                    name = "99",

                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),

                    text= df99.variety)

data = [trace1, trace2, trace3,trace4,trace5]

layout = dict(title = 'Points vs world rank of top 50 points with 87, 90,93,96 and 99 points',

              xaxis= dict(title= 'ID',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Price',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
#plotly bar plot



df99 = winemag130_data[winemag130_data.points == 99].iloc[:50,:]



import plotly.graph_objs as go

# create trace1 

trace1 = go.Bar(

                x = df99.variety,

                y = df99.price,

                name = "price",

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df99.region_1)

# create trace2 

trace2 = go.Bar(

                x = df99.variety,

                y = df99.points,

                name = "points",

                marker = dict(color = 'rgba(255, 255, 128, 0.5)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df99.region_1)

data = [trace1, trace2]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
#Plotly bar plot



df87 = winemag130_data[winemag130_data.points == 87].iloc[:10,:]



import plotly.graph_objs as go



x = df87.variety



trace1 = {

  'x': x,

  'y': df87.price,

  'name': 'price',

  'type': 'bar'

};

trace2 = {

  'x': x,

  'y': df87.points,

  'name': 'points',

  'type': 'bar'

};

data = [trace1, trace2];

layout = {

  'xaxis': {'title': 'Top 3 universities'},

  'barmode': 'relative',

  'title': 'price and points of top 10 variety in 87'

};

fig = go.Figure(data = data, layout = layout)

iplot(fig)

#plotly pie plot



df87 = winemag130_data[winemag130_data.points == 87].iloc[1:8,:]



value=df87.price

labels=df87.title



fig = {

  "data": [

    {

      "values": value,

      "labels": labels,

      "domain": {"x": [0, .5]},

      "name": "Wine names by price",

      "hoverinfo":"label+percent+name",

      "hole": .3,

      "type": "pie"

    },],

  "layout": {

        "title":"Wine names by price",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

              "text": "Wine Reviews Title",

                "x": 0.20,

                "y": 1

            },

        ]

    }

}

iplot(fig)
df87 = winemag130_data[winemag130_data.points == 87].iloc[1:21,:]

df87.info()
#plotly bubble plot



df87 = winemag130_data[winemag130_data.points == 87].iloc[1:21,:]



df=df87.fillna(0)



color=df.price



data = [

    {

        'y': df.price,

        'x': df.ID,

        'mode': 'markers',

        'marker': {

            'color': color,

            'size': color,

            'showscale': True

        },

        "text" :  df.variety    

    }

]

iplot(data)
winemag150_data.info()
winemag150_data.head()
winemag150_data["points"].unique()
#plotly histograms plot



df96 = winemag130_data[winemag130_data.points == 96].iloc[:50,:]

df100 = winemag130_data[winemag130_data.points == 100].iloc[:50,:]



import plotly.graph_objs as go



trace1 = go.Histogram(

    x=df96.price,

    opacity=0.75,

    name = "96 points",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))

trace2 = go.Histogram(

    x=df100.price,

    opacity=0.75,

    name = "100 points",

    marker=dict(color='rgba(12, 50, 196, 0.6)'))



data = [trace1, trace2]

layout = go.Layout(barmode='overlay',

                   title=' Wine Reviews price in 96 and 100 points',

                   xaxis=dict(title='price'),

                   yaxis=dict( title='Count'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)

#plotly cumulative histograms plot



df87 = winemag130_data[winemag130_data.points == 87].iloc[:100,:]



import plotly.graph_objs as go







trace2 = go.Histogram(

    x=df87.price,

    cumulative=dict(enabled=True))



data = [trace2]

layout = go.Layout(barmode='overlay',

                   title=' Wine Reviews price in 87 points',

                   xaxis=dict(title='price'),

                   yaxis=dict( title='Count'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)

#WorldCloud



df87 = winemag130_data[winemag130_data.points == 87].iloc[:160,:]

df87_new=df87.country[df87.points==87]



plt.subplots(figsize=(10,10))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(df87_new))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()
#plotly box plot



df99 = winemag130_data[winemag130_data.points == 99].iloc[:100,:]



df100 = winemag130_data[winemag130_data.points == 100].iloc[:100,:]



trace0 = go.Box(

    y=df99.price,

    name = 'total score of price in 99',

    marker = dict(

        color = 'rgb(12, 12, 140)',

    )

)

trace1 = go.Box(

    y=df100.price,

    name = 'research of price in 100',

    marker = dict(

        color = 'rgb(12, 128, 128)',

    )

)



data = [trace0, trace1]

iplot(data)

#Scatter plot matrix



df100=winemag150_data[winemag150_data.points == 100].iloc[:100,:]



import plotly.figure_factory as ff

# prepare data



df100 = df100.loc[:,["points","price", "ID"]]

df100["index"] = np.arange(1,len(df100)+1)

# scatter matrix

fig = ff.create_scatterplotmatrix(df100, diag='box', index='index',colormap='Portland',

                                  colormap_type='cat',

                                  height=700, width=700)

iplot(fig)
#Plotly inset plot



df100=winemag150_data[winemag150_data.points == 100]



trace1 = go.Scatter(

    x=df100.ID,

    y=df100.price,

    name = "price",

    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

)

# second line plot

trace2 = go.Scatter(

    x=df100.ID,

    y=df100.points,

    xaxis='x2',

    yaxis='y2',

    name = "points",

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

    title = 'Points and Price vs ID of Wine Reviews'



)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
#plotly 3D scatter plot



df99=winemag150_data[winemag150_data.points == 99]

df100=winemag150_data[winemag150_data.points == 100]



trace1 = go.Scatter3d(

    x=df99.ID,

    y=df99.price,

    z=df99.points,

    mode='markers',

    marker=dict(

        size=10,

        color='rgb(255, 0, 0)',                # set color to an array/list of desired values      

    )

)

trace2 = go.Scatter3d(

    x=df100.ID,

    y=df100.price,

    z=df100.points,

    mode='markers',

    marker=dict(

        size=10,

        color='rgb(127, 127, 127)',                # set color to an array/list of desired values      

    )

)



data = [trace1,trace2]

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