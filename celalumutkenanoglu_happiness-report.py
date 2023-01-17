# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# plotly

# import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go







# matplotlib

import matplotlib.pyplot as plt

%matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_Happiness2019 = pd.read_csv("/kaggle/input/world-happiness/2019.csv")

df_Happiness2018 = pd.read_csv("/kaggle/input/world-happiness/2018.csv")

df_Happiness2017 = pd.read_csv("/kaggle/input/world-happiness/2017.csv")
df_Happiness2019.info()
df_Happiness2019.head(10)
df_Happiness2018.head()
df_Happiness2019_100 = df_Happiness2019.iloc[:100,:] 



plt.figure(figsize=(20,10))

sns.barplot(x=df_Happiness2019_100["Country or region"], y=df_Happiness2019_100['Score'])

plt.xticks(rotation= 90)

plt.xlabel('Country or Region')

plt.ylabel('Score')

plt.title('Country or Region vs Score')
df_Happiness2019_3 = df_Happiness2019.iloc[:3,:]

df_Happiness2018_3 = df_Happiness2018.iloc[:3,:]



# import graph objects as "go"

import plotly.graph_objs as go

# create trace1 

trace1 = go.Bar(

                x = df_Happiness2019_3["Country or region"],

                y = df_Happiness2019_3["Score"],

                name = "Score for 2019",

                marker = dict(color = 'rgba(220, 150, 175, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df_Happiness2019_3["Overall rank"])

# create trace2 

trace2 = go.Bar(

                x = df_Happiness2018_3["Country or region"],

                y = df_Happiness2018_3["Score"],

                name = "Score for 2018",

                marker = dict(color = 'rgba(50, 40, 30, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df_Happiness2018_3["Overall rank"])

trace3 = go.Bar(

                x = df_Happiness2018_3["Country or region"],

                y = df_Happiness2018_3["Score"],

                name = "Score for 2018",

                marker = dict(color = 'rgba(120, 98, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df_Happiness2018_3["Overall rank"])

data = [trace1, trace2,trace3]

layout = go.Layout(title = 'Score and Ranking for first 3 country for 2017,2018 and 2019',

    barmode = "group")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
data = go.Scatter(

                    x = df_Happiness2019_100["Overall rank"],

                    y = df_Happiness2019_100["Social support"],

                    mode = "lines",

                    name = "Social Sport",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= df_Happiness2019_100["Country or region"])

layout = dict(title = "Social Sport",

              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)

             )

fig = dict(data =data , layout = layout)

iplot(fig)
plt.figure(figsize=(9,15))

sns.barplot(x=df_Happiness2019_100["Perceptions of corruption"],y=df_Happiness2019_100["Country or region"] ,color='green',alpha = 0.5,label='Yellow' )
plt.figure(figsize=(9,15))

sns.barplot(x=df_Happiness2019_100["Generosity"],y=df_Happiness2019_100["Country or region"] ,color='red',alpha = 0.5,label='Yellow' )
df_Happiness2018_100 = df_Happiness2018.iloc[:100,:]

trace1 =go.Scatter(

                    x = df_Happiness2019_100["Overall rank"],

                    y = df_Happiness2019_100["GDP per capita"],

                    mode = "markers",

                    name = "2019 GDP per Capita",

                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),

                    text= df_Happiness2019_100["Country or region"])

trace2 =go.Scatter(

                    x = df_Happiness2018_100["Overall rank"],

                    y = df_Happiness2018_100["GDP per capita"],

                    mode = "markers",

                    name = "2018 GDP per Capita",

                    marker = dict(color = 'rgba(0, 255, 10, 0.8)'),

                    text= df_Happiness2018_100["Country or region"])

                   

data = [trace1, trace2]

layout = dict(title = 'GDP per capita with 2019 and 2018',

              xaxis= dict(title= 'Overall Rank',ticklen= 5,zeroline= False),

              yaxis= dict(title= "GDP per Capita",ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
# data preparation



num_size  = df_Happiness2019_100["Score"]*5

international_color = df_Happiness2019_100["Freedom to make life choices"]

data = [

    {

        'y': df_Happiness2019_100["Generosity"],

        'x': df_Happiness2019_100["Overall rank"],

        'mode': 'markers',

        'marker': {

            'color': international_color,

            'size': num_size,

            'showscale': True

        },

        "text" :  df_Happiness2019_100["Country or region"]    

    }

]

iplot(data)




# visualize

f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x=df_Happiness2019_100["Country or region"],y=df_Happiness2019_100["GDP per capita"],color='lime',alpha=0.8)

sns.pointplot(x=df_Happiness2019_100["Country or region"],y=df_Happiness2019_100["Social support"],color='red',alpha=0.8)

plt.text(40,0.6,'GDP per capita',color='red',fontsize = 17,style = 'italic')

plt.text(40,0.55,'Social Support',color='lime',fontsize = 18,style = 'italic')

plt.xticks(rotation= 90)

plt.xlabel('States',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.title('GDP per Capita  VS  Social Support',fontsize = 20,color='blue')

plt.grid()
trace1 = go.Scatter(

                    x = df_Happiness2019_100["Overall rank"],

                    y = df_Happiness2019_100["Healthy life expectancy"],

                    mode = "lines",

                    name = "Social Sport",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= df_Happiness2019_100["Country or region"])

trace2 = go.Scatter(

                    x = df_Happiness2019_100["Overall rank"],

                    y = df_Happiness2019_100["Freedom to make life choices"],

                    mode = "lines",

                    name = "Social Sport",

                    marker = dict(color = 'rgba(180, 3, 240, 0.8)'),

                    text= df_Happiness2019_100["Country or region"])

layout = dict(title = "Health Life Expectancy and Freedom to Make Life Choices",

              xaxis= dict(title= 'Overall Rank',ticklen= 5,zeroline= False)

             )

data=[trace1,trace2]

fig = dict(data =data , layout = layout)

iplot(fig)
# data preparation





trace0 = go.Box(

    y=df_Happiness2019_100["Score"],

    name = 'total score in 2019',

    marker = dict(

        color = 'rgb(12, 12, 140)',

    )

)

trace1 = go.Box(

    y=df_Happiness2018_100["Score"],

    name = 'total score in 2018',

    marker = dict(

        color = 'rgb(90, 250, 2)',

    )

)

data = [trace0, trace1]

iplot(data)