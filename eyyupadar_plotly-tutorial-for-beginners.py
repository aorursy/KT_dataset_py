

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


timesData = pd.read_csv("../input/timesData.csv")


timesData.info()
timesData.head(10)
df = timesData.iloc[:100,:]



# import graph objects as "go"

import plotly.graph_objs as go



# Creating trace1

trace1 = go.Scatter(

                    x = df.world_rank,

                    y = df.citations,

                    mode = "lines+markers",

                    name = "citations",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= df.university_name)

# Creating trace2

trace2 = go.Scatter(

                    x = df.world_rank,

                    y = df.teaching,

                    mode = "markers+lines",

                    name = "teaching",

                    marker = dict(color = 'rgba(90, 26, 80, 0.8)'),

                    text= df.university_name)

data = [trace1, trace2]

layout = dict(title = 'Citation and Teaching vs World Rank of Top 100 Universities',

              xaxis= dict(title= 'World Rank',ticklen= 3,zeroline= True)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
timesData.head()
df2014 = timesData[timesData.year == 2014].iloc[:100,:]

df2015 = timesData[timesData.year == 2015].iloc[:100,:]

df2016 = timesData[timesData.year == 2016].iloc[:100,:]

import plotly.graph_objs as go



trace1 =go.Scatter(

                    x = df2014.world_rank,

                    y = df2014.citations,

                    mode = "markers",

                    name = "2014",

                    marker = dict(color = 'rgba(1, 140, 10, 0.9)'),

                    text= df2014.university_name)

                   

trace2 =go.Scatter(

                    x = df2015.world_rank,

                    y = df2015.citations,

                    mode = "markers",

                    name = "2015",

                    marker = dict(color = 'rgba(10, 240, 52, 0.8)'),

                    text= df2015.university_name)



trace3 =go.Scatter(

                    x = df2016.world_rank,

                    y = df2016.citations,

                    mode = "markers",

                    name = "2016",

                    marker = dict(color = 'rgba(2, 182, 130, 0.7)'),

                    text= df2016.university_name)

data =[trace1,trace2,trace3]



layout = dict(title="Citation and Teaching vs World Rank of Top 100 Universities with 2014 ,2015 and 2016 years",

                        xaxis = dict(title ="world_rank", ticklen = 4,zeroline =True),

                        yaxis = dict(title= "citations",ticklen = 4,zeroline =True)                        

             )

fig =dict(data = data, layout = layout)

iplot(fig)

df2014 = timesData[timesData.year == 2014].iloc[:50,:]

df2015 = timesData[timesData.year == 2015].iloc[:50,:]

df2016 = timesData[timesData.year == 2016].iloc[:50,:]

import plotly.graph_objs as go



trace1 =go.Scatter(

                    x = df2014.world_rank,

                    y = df2014.total_score,

                    mode = "markers",

                    name = "2014",

                    marker = dict(color = 'rgba(1, 140, 10, 0.9)'),

                    text= df2014.university_name)

                   

trace2 =go.Scatter(

                    x = df2015.world_rank,

                    y = df2015.total_score,

                    mode = "markers",

                    name = "2015",

                    marker = dict(color = 'rgba(10, 240, 52, 0.8)'),

                    text= df2015.university_name)



trace3 =go.Scatter(

                    x = df2016.world_rank,

                    y = df2016.total_score,

                    mode = "markers",

                    name = "2016",

                    marker = dict(color = 'rgba(2, 182, 130, 0.7)'),

                    text= df2016.university_name)

data =[trace1,trace2,trace3]



layout = dict(title="Citation and Teaching vs World Rank of Top 100 Universities with 2014 ,2015 and 2016 years",

                        xaxis = dict(title ="world_rank", ticklen = 4,zeroline =True),

                        yaxis = dict(title= "citations",ticklen = 4,zeroline =True)                        

             )

fig =dict(data = data, layout = layout)

iplot(fig)

df2014 = timesData[timesData.year == 2014].iloc[: 3,:]

df2014
df2014 = timesData[timesData.year == 2014].iloc[: 3,:]



import plotly.graph_objs as go



trace1 =go.Bar(

               

                x = df2014.university_name,

                y = df2014.citations,

                 name ="citations",

                 marker = dict(color='rgba(180,220,190,0.5)', line =dict(color='rgb(0,1,1)', width =1.5)),

                text = df2014.country)





trace2 =go.Bar(

               

                x = df2014.university_name,

                y = df2014.teaching,

                name ="teaching",

                marker = dict(color='rgba(220,80,10,0.5)', line =dict(color='rgb(0.5,0.5,0.5)', width =1.5)),

                text = df2014.country)

data =[trace1,trace2]

layout =go.Layout(barmode= "group")



fig =go.Figure(data=data,layout=layout)

iplot(fig)









df2015.head(3)
df2014 =timesData[timesData.year==2015].iloc[: 3,:]



import plotly.graph_objs as go



trace1 = go.Bar(

                  

                 x = timesData.university_name,

                 y =timesData.num_students,

                name = "num_students",

              

                marker = dict(color ="rgba(255,10,15,0.5)", line  =dict(color ="rgb(100,50,0)", width =5)),

                text =timesData.country

                 )



trace2 = go.Bar(

                  

                 x = timesData.university_name,

                 y =timesData.international_students,

                name = "international_students",

          

                marker = dict(color ="rgba(10,80,250,0.8)", line  =dict(color ="rgb(10,5,200)", width =5)),

                text =timesData.country)



data = [trace1,trace2]



layuot = go.Layout(barmode="group")



fig =dict(data=data, layout=layout)

iplot(fig)

import plotly.graph_objs as go

df2016 = timesData[timesData.year == 2016].iloc[: 7,:]

pie1 =df2016.num_students

pie1_list = [float(each.replace(",",".")) for each in  df2016.num_students]

labels = df2016.university_name



fig = {

    "data": [

        {

            "values":pie1_list,

            "labels":labels,

            "domain":{"x": [0, .5]},

            "name":"Number of .students Rates",

            "hoverinfo": "label + percent+ name",

            "hole": .3,

            "type":"pie"

        },        

    ],

   "layout": {

       "title": "University Number of Students Rates",

       "annotations":[{

            "font":{"size":20},

           "showarrow":False,

           "text":"Number of Students",

           "x":.20,

           "y":1

       },

          

       ]

   }

    

    

}

iplot(fig)





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







x2011 =timesData.student_staff_ratio[timesData.year == 2011]

x2012 =timesData.student_staff_ratio[timesData.year == 2012]



trace1 = go.Histogram(



    x=x2011,

    name ="2011",

    marker =dict(color ="rgba(178,56,25,0.6)"),

    opacity= 0.65)

trace2 = go.Histogram(

    x = x2012,

    name ="2012",

    marker =dict(color ="rgba(20,155,23,0.6)"), 

    opacity =0.65)

data =[trace1,trace2]



layout =go.Layout(barmode="overlay",

                 

                  title="students staff ration in 2011 and 2012",

                  xaxis =dict(title="Student Staff Ratio"),

                  yaxis = dict(title="count"),)



fig = go.Figure(data=data,layout=layout)

iplot(fig)





from wordcloud import WordCloud

x2016 = timesData.country[timesData.year == 2016]

plt.subplots(figsize=(10,10))

wordcloud = WordCloud(

                          background_color="pink",

                          width=500,

                          height=400

                         ).generate(" ".join(x2016))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()





x2016 = timesData[timesData.year== 2016]



trace0 = go.Box(

               y = x2016.total_score,

               name = "total score of universities in 2016",

               marker=dict(color="rgba(23,43,150,0.6)",

                          )            

              )

trace2 =go.Box(   

              y =x2016.research,

              name = "research of universities in 2016",

               marker= dict(color="rgba(200,87,67,0.6)",

                           )  

                  )



data =[trace0,trace2]

iplot(data)





df.head(1)

import plotly.figure_factory as ff



df = timesData[timesData.year == 2016]

dt2016 = df.loc[:,["research","international","total_score"]]

dt2016["index"] =np.arange(1, len(dt2016)+1)



fig = ff.create_scatterplotmatrix(dt2016, diag="box",index="index", colormap="Portland",

                            colormap_type ="cat",

                            height = 700, width= 700

                           )

iplot(fig)







trace3 =go.Scatter(

     x =df.world_rank,

     y =df.teaching,

     name = "teaching",

     marker =dict(color ="rgba(200,150,10,0.6)")

    

)



trace4 =go.Scatter(

                x= df.world_rank,

                y =df.income,

                marker = dict(color="rgba(100,250,255,0.6)"),

                name ="income",

                xaxis ="x2",

                yaxis = "y2",)

data = [trace3,trace4]



layout =go.Layout(

            xaxis2= dict(

            domain =[0.7, 0.98], anchor ="y2",

                       ),

            yaxis2 =dict(

                       domain =[0.7,0.98], anchor ="x2"

                   

            ),

            title ="income and teaching vs in world rank of universities",

)

fig =go.Figure(data=data, layout=layout)

iplot(fig)





trace0 = go.Scatter3d(

    x =df.world_rank,

    y =df.research,

    z= df.citations,

    mode = "markers",

    marker = dict(

                  size =10,

                  color = "rgb(255,0 ,0)"

                

                 

        

                 )



)



data = [trace0]

layout =go.Layout(

   

    margin = dict(l =0,r=0, b=0,t=0)

    

 

)

fig = go.Figure( data = data, layout = layout)

iplot(fig)

























trace1 = go.Scatter(

    x= df.world_rank,

    y = df.research,

    name ="research"



)

trace3 = go.Scatter(

    x = df.world_rank,

    y = df.income,

    name = "income",

    xaxis = "x3",

    yaxis = "y3"



)

trace4 = go.Scatter(

    x = df.world_rank,

    y = df.total_score,

    name = "total_score",

    xaxis = "x4",

    yaxis = "y4"



)

data =[trace1,trace3,trace4]



layout = go.Layout(



    xaxis =dict( domain =[0, 0.45]),

    yaxis =dict( domain =[0, 0.45]),

    xaxis3 =dict(domain =[0,0.45], anchor ="y3"),

    xaxis4 =dict(domain =[0.55, 1], anchor ="y4"),

    

   

    yaxis3 = dict(domain =[0.55,1], anchor ="x3"),

    yaxis4 = dict(domain =[0,0.45], anchor ="x4"),

    

    title ="Research, income and total score VS World Rank of Universities"

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)




