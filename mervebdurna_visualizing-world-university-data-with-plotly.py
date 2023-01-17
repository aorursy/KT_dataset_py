#import library

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



import os

print(os.listdir("../input"))

#loading data

df = pd.read_csv("../input/timesData.csv")
df.head(10)
df.info()
#missing data

total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head()

#cleaning data

df.dropna(inplace=True)



total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head()
#Line Charts



#selecting top 100 university of 2011

df_top_100= df.iloc[:100,:]



trace1 = go.Scatter(

    x=df_top_100["world_rank"],

    y=df_top_100["teaching"],

    mode="lines+markers",

    name = "teaching",

    marker = dict(color = 'rgba(210, 86, 99, 0.8)'),

    text= df_top_100["university_name"]

)



trace2 = go.Scatter(

    x=df_top_100["world_rank"],

    y=df_top_100["total_score"],

    mode="lines+markers",

    name = "total score",

    marker = dict(color = 'rgba(42, 39, 89, 0.8)'),

    text= df_top_100["university_name"]

)



data = [trace1, trace2]

layout = dict(title = 'Teaching and Total Score vs World Rank of Top 100 Universities',

              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
# Scatter Charts



df_us_universities = df_top_100[df_top_100["country"]=="United States of America"]

df_germany_universities = df_top_100[df_top_100["country"]=="Germany"]

df_uk_universities = df_top_100[df_top_100["country"]=="United Kingdom"]



trace1 =go.Scatter(

                    x = df_us_universities.world_rank,

                    y = df_us_universities.total_score,

                    mode = "markers",

                    name = "USA",

                    marker = dict(color = 'rgba(255, 120, 100, 0.8)'),

                    text= df_us_universities["university_name"])



trace2 =go.Scatter(

                    x = df_germany_universities.world_rank,

                    y = df_germany_universities.total_score,

                    mode = "markers",

                    name = "Germany",

                    marker = dict(color = 'rgba(30, 200, 150, 0.8)'),

                    text= df_germany_universities["university_name"])



trace3 =go.Scatter(

                    x = df_uk_universities.world_rank,

                    y = df_uk_universities.total_score,

                    mode = "markers",

                    name = "United Kingdom",

                    marker = dict(color = 'rgba(0, 30, 200, 0.8)'),

                    text= df_uk_universities["university_name"])

data = [trace1, trace2, trace3]

layout = dict(title = 'The total score of universities  in Usa, Germany and Uk in the top 100',

              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Total Score',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
# Bar Charts



df2015 = df[df.year == 2015].iloc[:5,:]



trace1 = go.Bar(

                x = df2015["university_name"],

                y = df2015["total_score"],

                name = "total_score",

                marker = dict(color = 'rgba(250, 0, 30, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df2015.country)

# create trace2 

trace2 = go.Bar(

                x = df2015["university_name"],

                y = df2015["citations"],

                name = "citations",

                marker = dict(color = 'rgba(10, 150, 0, 0.5)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df2015.country)

data = [trace1, trace2]

layout = go.Layout(barmode = "group",title = 'Top 5 universities in 2015')

fig = go.Figure(data = data, layout = layout)

iplot(fig)
# Bar Charts 2  



df2013 = df[df["year"] == 2013].iloc[:5,:]



trace1 = {

  'x': df2013["university_name"],

  'y': df2013["citations"],

  'name': 'citations',

  'type': 'bar'

};

trace2 = {

  'x': df2013["university_name"],

  'y': df2013["teaching"],

  'name': 'teaching',

  'type': 'bar'

};

trace3 = {

  'x': df2013["university_name"],

  'y': df2013["research"],

  'name': 'research',

  'type': 'bar'

};

data = [trace1, trace2,trace3];

layout = {

  'xaxis': {'title': 'Top 5 universities'},

  'barmode': 'relative',

  'title': 'Top 5 universities in 2013'

};

fig = go.Figure(data = data, layout = layout)

iplot(fig)

#Pie Charts



df2011 = df[df.year == 2011].iloc[:100,:]



values = df2011["country"].value_counts().values

labels = df2011["country"].value_counts().index

# figure

fig = {

  "data": [

    {

      "values": values,

      "labels": labels,

      "domain": {"x": [0, .5]},

      "name": "Distribution of the first 100 universities by country",

      "hoverinfo":"label+percent",

      "hole": .3,

      "type": "pie"

    },],

  "layout": {

        "title":"Distribution of the first 100 universities by country",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

              "text" :"",

              "x": 0.50,

              "y": 1

            },

        ]

    }

}

iplot(fig)

# Bubble Charts



df2016 = df[df.year == 2016].iloc[:20,:]

students_number_size  = [float(each.replace(',', '.')) for each in df2016["num_students"]]

international_students_percent_color = [float(each.replace('%', '')) for each in df2016["international_students"]]

data = [

    {   'x': df2016["university_name"],

        'y': df2016["total_score"],

        'mode': 'markers',

        'marker': {

            'color': international_students_percent_color,

            'size': students_number_size,

            'showscale': True

        },

        "text" :df2016["university_name"]

    }

]

iplot(data)
# Histogram

df2011 = df[df["year"] == 2011].iloc[:100,:]

df2016 = df[df["year"] == 2012].iloc[:100,:]



trace1 = go.Histogram(

    x=df2011["country"],

    opacity=0.75,

    name = "2011",

    marker=dict(color='rgba(231, 50, 5, 0.6)'))

trace2 = go.Histogram(

    x=df2016["country"],

    opacity=0.75,

    name = "2016",

    marker=dict(color='rgba(2, 50, 196, 0.6)'))



data = [trace1, trace2]

layout = go.Layout(barmode='group',#overlay

                   title=' Number of countries universities in 2011 and 2016',

                   xaxis=dict(title='country'),

                   yaxis=dict( title='Count'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
# Word Cloud

from wordcloud import WordCloud

plt.subplots(figsize=(10,10))

wordcloud = WordCloud(background_color='black',

                      width=512,

                      height=384

                      ).generate(" ".join(df["country"]))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()
#Box Plot



df2015 = df[df.year == 2015]



trace1 = go.Box(

    y=df2015["total_score"],

    name = 'total score of universities in 2015',

    marker = dict(

        color = 'rgb(200, 10, 10)',

    )

)

trace2 = go.Box(

    y=df2015["research"],

    name = 'research of universities in 2015',

    marker = dict(

        color = 'rgb(10, 200, 10)',

    )

)

    

trace3 = go.Box(

    y=df2015["citations"],

    name = 'citations of universities in 2015',

    marker = dict(

        color = 'rgb(10, 10, 200)',

    )

)

data = [trace1,trace2,trace3]

iplot(data)
# Scatter Plot Matrix

#covariance and relation between more than 2 features



# import figure factory

import plotly.figure_factory as ff



df2015 = df[df["year"] == 2015]

df2015 = df2015.loc[:,["research","international", "total_score"]]

df2015["index"] = np.arange(1,len(df2015)+1)

# scatter matrix

fig = ff.create_scatterplotmatrix(df2015, diag='histogram', index='index',colormap='Viridis',

                                  colormap_type='cat',

                                  height=700, width=700)

iplot(fig)
#3d scatter plot

trace1 = go.Scatter3d(

    x=df.world_rank,

    y=df.research,

    z=df.citations,

    mode='markers',

    marker=dict(

        size=10,

        color='rgb(150,0,150)',                # set color to an array/list of desired values      

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