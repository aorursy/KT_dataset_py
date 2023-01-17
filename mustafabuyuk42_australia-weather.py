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
weather = pd.read_csv("../input/weatherAUS.csv")
weather.tail()
df = weather.iloc[:,:]

lc =df.Location.unique()

df["year"] = [int(each.split('-')[0]) for each in df.iloc[:,0]] 

trace1 = go.Scatter( x = lc,

                     y = df.MinTemp,

                     mode = 'lines',

                     name = 'MinTemp',

                     marker = dict (color ='rgba(255,125,65,0.8)'),

                     text =   df.year)

trace2 = go.Scatter( x = lc,

                     y = df.MaxTemp,

                     mode = 'lines', 

                     name = 'MaxTemp',

                     marker =dict(color = 'rgba(136,255,255,0.7)'),

                     text = df.year)

data = [trace1,trace2]

layout = dict( title = ' Max ve Min Temperature values',xaxis = dict(title = 'Australian Cities',ticklen= 5,zeroline= False), yaxis = dict(title = 'Celcius'))

fig =dict(data= data , layout = layout)

iplot(fig)



weather.columns
weather["year"] =[int(each.split('-')[0]) for each in weather.iloc[:,0]]

# preperaing data frames 

df2013 = weather[weather.year == 2013].iloc[:,:]

df2015 = weather[weather.year == 2015].iloc[:,:]

df2017 = weather[weather.year == 2017].iloc[:,:]



trace1 = go.Scatter(x = df2013.Location.unique(),

                    y = df2013.MaxTemp,

                    mode = 'markers',

                    name = '2013',

                    marker =dict(color ='rgba(200,0,125,0.8)'),

                    text =df2013.year)

trace2 = go.Scatter(x = df2015.Location.unique(),

                    y = df2015.MaxTemp,

                    mode = 'markers',

                    name = '2015',

                    marker = dict(color = 'rgba(20,155,22,0.8)'),

                    text = df2015.year )

trace3 = go.Scatter(x = df2017.Location.unique(),

                    y = df2017.MaxTemp,

                    mode = 'markers',

                    name = '2017',

                    marker = dict(color = 'rgba(121,121,0,0.8)'),

                    text =df2017.year)

data = [trace1,trace2,trace3]

layout = dict(title = 'Temperatures at different years',

              xaxis = dict(title = 'Cities',ticklen= 5,zeroline= False),

              yaxis = dict(title = 'Celcius',ticklen= 5,zeroline= False)  )

fig = go.Figure(data = data,layout =layout)

iplot(fig)
df2014 = weather[weather.year == 2014].iloc[:,:]

trace1 = go.Bar(x = df2014.Location.unique(),

                y = df2014.MaxTemp,

                name = 'MaxTemp',

                marker = dict(color = 'rgba(255, 174, 255, 0.5)'),   

                text = df2014.year)

     

trace2 = go.Bar(x = df2014.Location.unique(),

                y = df2014.MinTemp,

                name = 'MinTemp',

                marker = dict(color = 'rgba(255,45,44,0.9)',line = dict(color ='rgba(0,0,0)',width = 1.5)),   

                text = df2014.year)

             



data =[trace1,trace2]

layout = go.Layout(barmode = 'group')

fig = go.Figure(data = data ,layout = layout)

iplot(fig)
weather.head()
df2015 = weather[weather.year == 2015].iloc[:7,:]

df2017 = weather[weather.year == 2017].iloc[:7,:]



values =[list(df2015.iloc[:,2]),list(df2017.iloc[:,2])]

labels = list(df2015.Location)+list(df2017.Location)
trace = go.Pie(labels = labels, values = values)

data = [trace]

layout = dict ( title = 'hj')

fig = dict(data = data ,layout = layout)

iplot(fig)
weather["year"] =[int(each.split('-')[0]) for each in weather.iloc[:,0]]

df2015 = weather.MaxTemp[weather.year == 2015]

df2014 = weather.MaxTemp[weather.year == 2014]



trace1 = go.Histogram(x = df2015,

                      opacity = 0.6, 

                      name = '2015',    

                      marker=dict(color='rgba(171, 50, 96, 0.8)'))

trace2 = go.Histogram( x= df2014,

                      opacity = 0.7,

                      name = '2014',    

                      marker=dict(color='rgba(1, 250, 96, 0.8)'))



data =[trace1,trace2]

layout = go.Layout(barmode='overlay',

                   title='Max temperature in 2014 and 2015',

                   xaxis=dict(title='max temp (celcius)'),

                   yaxis=dict( title='count'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
weather.head()
df2016 = weather[weather.year == 2016]

trace1 = go.Box(y =df2016.Humidity9am,

                name = 'Humidity9am',

                marker = dict(color ='rgba(255,255,2,0.8)'))

trace2 = go.Box( y= df2016.Humidity3pm,

                 name = 'Humidity3pm',

                 marker = dict(color = 'rgba(0,10,250,0.4)'))

data =[trace1,trace2]

iplot(data)
import plotly.figure_factory as ff

df = weather[weather.year == 2015].iloc[:1000,:]

data2015 = df.loc[:,["MinTemp","MaxTemp","Humidity9am"]]

data2015["index"] = np.arange(1,len(data2015)+1)

fig = ff.create_scatterplotmatrix(data2015, diag='box', index='index',colormap='Portland',

                                  colormap_type='cat',

                                  height=700, width=700)

iplot(fig)
trace1 = go.Scatter(x = weather.Location.unique(),

                    y = weather.MaxTemp,

                    name = 'MaxTemp')

trace2 = go.Scatter(x = weather.Location.unique(),

                    y = weather.MinTemp,

                    xaxis='x2',

                    yaxis='y2',

                    name = 'MinTemp')

trace3 = go.Scatter(x = weather.Location.unique(),

                    y = weather.Temp3pm,

                    xaxis ='x3',

                    yaxis ='y3',

                    name  = 'Temp3pm')

trace4 = go.Scatter(x =weather.Location.unique(),

                    y = weather.Humidity9am,

                    xaxis='x4',

                    yaxis='y4',

                    name = 'Temp9am')

data = [trace1,trace2,trace3,trace4]

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

    title = 'Air condition in Australia')

fig =go.Figure(data = data,layout=layout)

iplot(fig)