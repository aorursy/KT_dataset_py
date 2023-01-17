import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import os

import warnings

warnings.filterwarnings('ignore')

import plotly.graph_objects as go

pd.set_option('display.max_rows',2000, 'display.max_columns',10)
estonia = pd.read_csv("../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv")



AliveOrNot = []



for i in estonia['Survived']:

    if i == 0.0:

        AliveOrNot.append('Not Survived')

    else:

        AliveOrNot.append('Survived')

estonia['AliveOrNot'] = AliveOrNot



estonia['Survived'] = estonia['Survived'].astype(float)



estonia.head()
#fig = px.pie(estonia,names='Country')

fig = go.Figure(data=[go.Pie(labels=estonia['Country'], hole=.3)])

fig.update_traces(textposition='inside')

fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

fig.show()
estonia['Age'] = estonia['Age'].astype(float)

fig = px.treemap(estonia, path=['Country','Firstname','Lastname','Age'],color='Country')

fig.update_layout(

    title='Passengers with country,name and age travelling on Estonia')

fig.show()
fig = px.histogram(estonia, x="Age",color='Sex' )

fig.update_layout(

    title='Age of Male and Female Passengers on Estonia')

fig.show()
fig = px.histogram(estonia, x="Country",color='AliveOrNot')

fig.update_layout(

    title='Countrywise People Survived and Not Survived on Estonia')

fig.show()
fig = px.histogram(estonia, x="Country",color='Category')

fig.update_layout(

    title='Countrywise Passengers travelling on P and C class in Estonia')

fig.show()
#Survived People with Class

estonia_survived = estonia[estonia['AliveOrNot']=='Survived']

estonia_survived_detail=estonia_survived.groupby(['Country','Sex','Age','Category'])['AliveOrNot'].count().reset_index(name='Survivor')

fig = px.sunburst(estonia_survived_detail,path=['Country','Sex','Age','Category'],values='Survivor')

fig.update_layout(

    title='People Survivor Details')

fig.show()
#Survived People with Class

estonia_survived = estonia[estonia['AliveOrNot']=='Not Survived']

estonia_survived_detail=estonia_survived.groupby(['Country','Sex','Age','Category'])['AliveOrNot'].count().reset_index(name='Not_Survived')

fig = px.sunburst(estonia_survived_detail,path=['Country','Sex','Age','Category'],values='Not_Survived')

fig.update_layout(

    title='People Not Survived Details')

fig.show()
#Category for Travelling Class wrt to Age

fig = px.box(estonia, x="Category", y="Age")

fig.update_layout(

    title='Category of Travelling Class')

fig.show()
#Category for Travelling Class wrt to Age

fig = px.box(estonia, x="Country", y="Age",color='Sex')

fig.update_layout(

    title='Countrywise Traveller Age')

fig.show()
#Category for Travelling Class wrt to Age

fig = px.box(estonia, x="Country", y="Age",color='Category')

fig.update_layout(

    title='Countrywise Traveller Age with their Travelling Category')

fig.show()
estonia_survived = estonia[estonia['AliveOrNot']=='Survived']

#Category for Travelling Class wrt to Age

fig = px.box(estonia_survived, x="Country", y="Age",color='Sex')

fig.update_layout(

    title='Survived Traveller Age')

fig.show()
estonia_survived = estonia[estonia['AliveOrNot']=='Not Survived']

#Category for Travelling Class wrt to Age

fig = px.box(estonia_survived, x="Country", y="Age",color='Sex')

fig.update_layout(

    title='Traveller Age that did not survived')

fig.show()