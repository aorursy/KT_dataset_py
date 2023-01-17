import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

# word cloud library

from wordcloud import WordCloud

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# dataset load

mall_customer = pd.read_csv('../input/Mall_Customers.csv')

mall_customer.head()
mall_customer.info()
# Line Charts

import plotly.graph_objs as go

df = mall_customer.iloc[:100,:]

x = df.CustomerID

y1 = df['Annual Income (k$)']

y2 = df['Spending Score (1-100)']

#question1

# create treaces

trace1 = go.Scatter(

                    x = x,

                    y = y1,

                    mode = 'lines',

                    name = 'Income',

                    marker = dict(color = 'rgba(17,120,20,0.8)'),

                    )

trace2 = go.Scatter(

                    x = x,

                    y = y2,

                    mode = 'lines+markers',

                    name = 'Score',

                    marker = dict(color = 'rgba(118, 26, 17, 0.8)'),

                    )

data = [trace1,trace2]

layout = dict(title = 'income and score for customers', 

              xaxis = dict(title = 'Customers number', ticklen = 5, zeroline = False)

             )

fig = dict(data=data, layout=layout)

iplot(fig)

#Scatter Charts

import plotly.graph_objs as go

df = mall_customer.iloc[:100,:]

x = df.CustomerID

y1 = df['Annual Income (k$)']

y2 = df['Spending Score (1-100)']

#question1

# create treaces

trace1 = go.Scatter(

                    x = x,

                    y = y1,

                    mode = 'markers',

                    name = 'Income',

                    marker = dict(color = 'rgba(10,12,200,0.8)'),

                    )

trace2 = go.Scatter(

                    x = x,

                    y = y2,

                    mode = 'markers',

                    name = 'Score',

                    marker = dict(color = 'rgba(80, 26, 17, 0.8)'),

                    )

data = [trace1,trace2]

layout = dict(title = 'income and score for customers', 

              xaxis = dict(title = 'müşteri numaraları', ticklen = 5, zeroline = False)

             )

fig = dict(data=data, layout=layout)

iplot(fig)
mall_customer.head()
#Bar Charts

df1 = mall_customer[mall_customer.Gender == 'Male']

df2 = mall_customer[mall_customer.Gender == 'Female']

male_age =  df1.Age 

male_income = df1['Annual Income (k$)']

male_score = df1['Spending Score (1-100)']

female_age = df2.Age

female_income = df2['Annual Income (k$)']

female_score = df2['Spending Score (1-100)']

trace1 = go.Bar(

    x = male_age,

    y = male_income,

    name = 'male income',

    marker = dict(color = 'rgba(20,90,5,0.8)',line=dict(color='rgb(0,0,0)',width=1.5))

)

trace2 = go.Bar(

    x = male_age,

    y = male_score,

    name = 'male score',

    marker = dict(color = 'rgba(200,50,125,0.8)',line=dict(color='rgb(0,0,0)',width=1.5))

)

data = [trace1, trace2]

layout = dict(title = 'skore and income for male customer',

              xaxis = dict(title = 'Age', zeroline = False, ticklen = 5),

              yaxis = dict(title = 'skore and income', zeroline = False, ticklen = 5),

              barmode = 'group')

fig = dict(data = data, layout = layout)

iplot(fig)
#bar as one on the top of the other

import plotly.graph_objs as go



df1 = mall_customer[mall_customer.Gender == 'Male']

df2 = mall_customer[mall_customer.Gender == 'Female']

male_age =  df1.Age 

male_income = df1['Annual Income (k$)']

male_score = df1['Spending Score (1-100)']

female_age = df2.Age

female_income = df2['Annual Income (k$)']

female_score = df2['Spending Score (1-100)']



trace1 = {

  'x' : female_age,

  'y' : female_income,

  'name' : 'income',

  'type' : 'bar'

};

trace2 = {

  'x' : female_age,

  'y' : female_score,

  'name' : 'score',

  'type' : 'bar'

};



data = [trace1,trace2];

layout = {

    'xaxis' : {'title':'age'},

    'barmode' : 'relative',

    'title' : 'skore and income for female customer'

}

fig = go.Figure(data = data, layout = layout)

iplot(fig)
# pie chart

df1 = mall_customer[mall_customer.Gender == 'Male']

df2 = mall_customer[mall_customer.Gender == 'Female']

male_age =  df1.Age 

male_income = df1['Annual Income (k$)']

male_score = df1['Spending Score (1-100)']

female_age = df2.Age

female_income = df2['Annual Income (k$)']

female_score = df2['Spending Score (1-100)']



fig = {

    "data" : [

        {

            "values" : male_age,

            "labels" : male_income,

            "domain" : {"x": [1, .5]},

            "name" : "Age of male",

            "hoverinfo" : "label+percent+name",

            "hole" : .5,

            "type" : "pie"

        

    },],

    "layout": {

        "title" : "Income ratio for age of male",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

              "text": "Number of male",

                "x": 0.10,

                "y": 1

            },

        ]  

    }

}

iplot(fig)

# buble chart

df1 = mall_customer[mall_customer.Gender == 'Male']

df2 = mall_customer[mall_customer.Gender == 'Female']

male_age =  df1.Age 

male_income = df1['Annual Income (k$)']

male_score = df1['Spending Score (1-100)']

female_age = df2.Age

female_income = df2['Annual Income (k$)']

female_score = df2['Spending Score (1-100)']

data = [

    {

        'x' : female_age,

        'y' : female_score,

        'mode' :'markers',

        'marker' : {

            'color' : female_income,

            'size' : female_age, 

            'showscale' :True

        },

        

    }

]

iplot(data)
#histogram

df1 = mall_customer[mall_customer.Gender == 'Male']

df2 = mall_customer[mall_customer.Gender == 'Female']

male_age =  df1.Age 

male_income = df1['Annual Income (k$)']

male_score = df1['Spending Score (1-100)']

female_age = df2.Age

female_income = df2['Annual Income (k$)']

female_score = df2['Spending Score (1-100)']

trace1 = go.Histogram(

    x = male_age,

    opacity = 0.75,

    name = "male age",

    marker =dict(color='rgba(171,20,85,0.8)'))

trace2 = go.Histogram(

    x = female_age,

    opacity = 0.75,

    name = "female age",

    marker = dict(color='rgba(20,15,110,0.8)'))

data = [trace1, trace2]

layout = go.Layout(barmode = 'overlay',

                   title = 'male-female age ratio',

                   xaxis = dict(title='age ratio for Female-male '),

                   yaxis = dict(title='ratio')

                  )

fig = go.Figure(data=data, layout=layout)

iplot(fig)
mall_customer.head()
# word cloud

list_color = ['purple','blue','purple','yellow','purple','pink','purple','blue','blue','yellow','green','purple','white','purple','purple']

plt.subplots(figsize=(8,8))

wordcloud = WordCloud(

                        background_color = 'white',

                        width = 512,

                        height = 384

                        ).generate(" ".join(list_color))

# generate yan yana yazılı olanları ayırır ve en çok yazılanı büyük harfle plot ettirir.

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')

plt.show()



#box plot

df1 = mall_customer[mall_customer.Gender == 'Male']

df2 = mall_customer[mall_customer.Gender == 'Female']

male_age =  df1.Age 

male_income = df1['Annual Income (k$)']

male_score = df1['Spending Score (1-100)']

female_age = df2.Age

female_income = df2['Annual Income (k$)']

female_score = df2['Spending Score (1-100)']



trace1 = go.Box(

     y = male_income,

     name = 'male income',

     marker = dict(color = 'rgba(254,24,0,0.8)')

)

trace2 = go.Box(

     y = female_income,

     name = 'female income',

     marker = dict(color = 'rgba(5,140,20,0.8)')

)

data = [trace1,trace2]

iplot(data)
#scatter plot

mall = mall_customer.iloc[:,2:]

import plotly.figure_factory as ff

mall['index'] = np.arange(1,len(mall)+1)

fig = ff.create_scatterplotmatrix(mall, diag ='box', index = 'index' , colormap =['rgb(10, 10, 255)', '#F0963C', 'rgb(51, 255, 153)'],

                                 colormap_type = 'seq', height = 700, width = 700)

iplot(fig)
#inset plot

df1 = mall_customer[mall_customer.Gender == 'Male']

df2 = mall_customer[mall_customer.Gender == 'Female']

male_age =  df1.Age 

male_income = df1['Annual Income (k$)']

male_score = df1['Spending Score (1-100)']

female_age = df2.Age

female_income = df2['Annual Income (k$)']

female_score = df2['Spending Score (1-100)']



trace1 = go.Scatter(

    y = male_age, 

    x = mall_customer['Annual Income (k$)'],

    name = 'male age',

    marker = dict(color = 'rgba(160, 112, 2, 0.8)')

)

trace2 = go.Scatter(

    y = female_age,

    x = mall_customer['Annual Income (k$)'],

    xaxis = 'x2',

    yaxis =  'y2',

    name =  'female age',

    marker = dict(color = 'rgba(200,20,154,0.8)')

)



data = [trace1, trace2]

layout = go.Layout(

    xaxis = dict(title = 'income'),

    yaxis = dict(title = 'age'),

    xaxis2 = dict(

            domain = [0.6, 0.95],

            anchor = 'y2',

    ),

    yaxis2 = dict(

            domain = [0.6, 0.95],

    ),

    title = 'female and male age for income'

)

fig = go.Figure(data = data, layout=layout)

iplot(fig)
# 3D Scatter Plot

df1 = mall_customer[mall_customer.Gender == 'Male']

df2 = mall_customer[mall_customer.Gender == 'Female']

male_age =  df1.Age 

male_income = df1['Annual Income (k$)']

male_score = df1['Spending Score (1-100)']

female_age = df2.Age

female_income = df2['Annual Income (k$)']

female_score = df2['Spending Score (1-100)']

trace1 = go.Scatter3d(

    x = female_age,

    y = female_income,

    z = female_score,

    mode = 'markers',

    marker = dict(

            size=[15,20,25],

            color = male_age)

)

data = [trace1]

layout = go.Layout(

   margin = dict(

       l = 0,

       r = 0,

       b = 0,

       t = 0

   )

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
#multiple subplots



list_1 = mall_customer.Gender.unique()

trace1 = go.Scatter(

    x = list_1,

    y = mall_customer['Annual Income (k$)'],

    name = "income",

    marker = dict(color = 'purple')

)

trace2 = go.Scatter(

    x = list_1,

    y = mall_customer['Spending Score (1-100)'],

    xaxis = 'x2',

    yaxis = 'y2',

    name = "score",

    marker = dict(color='navy')

)



data = [trace1,trace2]

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

    title = 'income and score'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)