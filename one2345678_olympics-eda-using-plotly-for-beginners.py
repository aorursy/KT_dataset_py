import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
data=pd.read_csv('../input/athlete_events.csv')
#read file as data
#have a look on data 
data.info()
#some values are NAN in some columns like Medals 
#so sad that not all of the participants were able to grab a medal
#ohh my God 
#minimum age of a participant is 10 and maximum is 97 
#also have a look on weight and height 
data.describe()
data.describe(include='O')
# 10 top rows of the data
data.head(10)
#lets check skewness of some numerical columns 
#not that much skewed 
#I was expecting more:)
data['Age'].skew()
data['Weight'].skew()
data['Height'].skew()
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import scipy
import numpy as np
#I used get_dummies() OHE can also be used 
# NAN values will be dropped 
df=pd.get_dummies(data['Medal'])
df.head()
data=pd.concat([data,df],axis=1)
data['total']= (data['Gold']+data['Silver']+data['Bronze'])
data.drop('Medal',axis=1,inplace=True)
#ohh the it looks little nice 
data.head()
teamc=data.groupby(['Team'])[['Bronze','Gold','Silver']].sum().astype(np.int32).reset_index()
teamc.head(20)
teamc['total']= (teamc['Gold']+teamc['Silver']+teamc['Bronze'])
teamc=teamc.sort_values(by=['total'])
# sort the dataframe on the basis of total medal 
#team United States with the highest medals 
teamc.tail(10)
#male players won approximately 3 times more medals than female players 
pd.crosstab(data.Sex,data.Gold)
pd.crosstab(data.Sex,data.Silver)
pd.crosstab(data.Sex,data.Bronze)
#check which olympic year got highest medals 
game=data.groupby(['Games'])[['Bronze','Gold','Silver']].sum().astype(np.int32).reset_index()
game.head()
game['total']= (game['Gold']+game['Silver']+game['Bronze'])
game=game.sort_values(by=['total'])
game.head()
#2008 Summer
game.tail()
#Plotly charts could be created offline. It  means that you do not need an API or anything else.
#look mind blowing plotly examples
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from subprocess import check_output
# you can use your favorite colors
df=game
df = pd.DataFrame(df)
Games=df['Games']
trace = go.Bar(
    x=df['Games'],
    y=df['total'],
    marker=dict(
        color = 'rgb(255,0,0)',
        colorscale='Jet',
        showscale=False
    ),
)
layout = go.Layout(
    title='total medals distribution', yaxis = dict(title = 'total medals', zeroline = False,range=[1,3000])
)
data1 = [trace]
fig = go.Figure(data=data1, layout=layout)
py.iplot(fig, filename="total")
trace = go.Bar(
    x=df['Games'],
    y=df['Gold'],
    marker=dict(
        color = 'rgb(0,121,120)',
        colorscale='Jet',
        showscale=False
    ),
)
layout = go.Layout(
    title='Gold medals  distribution', yaxis = dict(title = 'Gold', zeroline = False,range=[1,1000])
)
data1 = [trace]
fig = go.Figure(data=data1, layout=layout)
py.iplot(fig, filename="Gold")
trace = go.Bar(
    x=df['Games'],
    y=df['Silver'],
    marker=dict(
        color = 'rgb(255,0,130)',
        colorscale='Jet',
        showscale=False
    ),
)
layout = go.Layout(
    title='Silver medals distribution', yaxis = dict(title = 'Silver', zeroline = False,range=[1,1000])
)
data1 = [trace]
fig = go.Figure(data=data1, layout=layout)
py.iplot(fig, filename="Silver")
name=data.groupby(['Name'])[['Bronze','Gold','Silver']].sum().astype(np.int32).reset_index()
name.tail()
name['total']= (name['Gold']+name['Silver']+name['Bronze'])
name=name.sort_values(by=['total'])
name=name.reset_index()
name.tail(10)
# see highest medals holder let's search a bit more about him
data.loc[data['Name'] == 'Michael Fred Phelps, II']
# again United States and it's swimming 28 medals 
data.loc[data['Name'] == 'Larysa Semenivna Latynina (Diriy-)']
# Gymnastics 
# where is usain bolt  :(
noc=pd.crosstab(data.NOC,data.Sex)
noc['Ratio']= (noc['M']/noc['F'])
noc=noc.sort_values(by=['Ratio'])
noc.head(20)
season=pd.crosstab(data.Season,data.total)
season
event=pd.crosstab(data.Event,data.total)
event.reset_index()
event.index = event.index.set_names(['index'])
event.tail()
# genderwise participation 
gender = data['Sex'].value_counts()
label = (np.array(gender.index))
size = gender.values

colors = ['skyblue', 'orange']
trace = go.Pie(labels=label, values=size, marker=dict(colors=colors))
layout = go.Layout(
    title='Gender Distribution'
)
data1 = [trace]
fig = go.Figure(data=data1, layout=layout)
py.iplot(fig, filename="data")
#agewise medal distribution
data['Age']=data['Age'].fillna(0)
df=data
df = pd.DataFrame(df)
age = df['Age'].value_counts()
trace = go.Bar(
    x=age.index,
    y=age.values,
    marker=dict(
        color = age.values,
        colorscale='Jet',
        showscale=True
    ),
)
layout = go.Layout(
    title='Age distribution', yaxis = dict(title = 'age of participants', zeroline = False)
)
data1 = [trace]
fig = go.Figure(data=data1, layout=layout)
py.iplot(fig, filename="age")
#weightwise medal distribution
data['Weight']=data['Weight'].fillna(0)
df=data
df = pd.DataFrame(df)
Weight = df['Weight'].value_counts()
trace = go.Bar(
    x=Weight.index,
    y=Weight.values,
    marker=dict(
        color = Weight.values,
        colorscale='Jet',
        showscale=False
    ),
)
layout = go.Layout(
    title='Weight distribution', yaxis = dict(title = 'Weight of participants', zeroline = False,range=[1,10000])
)
data1 = [trace]
fig = go.Figure(data=data1, layout=layout)
py.iplot(fig, filename="Weight")
#last Game wise 
#everyone was impressing the girl named medals:) some got Gold some not..
data['Games']=data['Games'].fillna('.')
df=data
df = pd.DataFrame(df)
Games = df['Games'].value_counts()
trace = go.Bar(
    x=Games.index,
    y=Games.values,
    marker=dict(
        color = Games.values,
        colorscale='Jet',
        showscale=True
    ),
)
layout = go.Layout(
    title='Participants in Games', yaxis = dict(title = 'Games', zeroline = False,range=[1,15000])
)
data1 = [trace]
fig = go.Figure(data=data1, layout=layout)
py.iplot(fig, filename="Games")




