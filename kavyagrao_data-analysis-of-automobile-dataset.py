import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly

plotly.offline.init_notebook_mode()

import plotly.express as px

import plotly.graph_objects as go

df1 = pd.read_csv('../input/autompg-dataset/auto-mpg.csv')

df1
df1.head(10)
df1.tail(10)
df1.dtypes
df1.describe()
df1.shape
df1.columns
len(df1.columns)
df1.info
df1['horsepower'].fillna(value=df1['horsepower'].replace('?',0))
df1.loc[(df1['weight']<2000) & (df1['mpg']>35) & (df1['cylinders']>=4) ,['car name']]
df1.groupby(by='car name',sort=False)['cylinders'].max().head(10)
sns.set(style='whitegrid')

ax=sns.countplot(df1.cylinders,palette=['#2A363B','#FECBAB','#FF847C','#E84A5F','#99B898'])

ax.set(title="Bar Plot")

ax.set(xlabel='No of cylinders')

ax.set(ylabel='No of Cars')
sns.set(style='whitegrid')

ax=sns.scatterplot(x=df1.mpg,y=df1.weight,hue=df1.cylinders,palette=['green','orange','brown','dodgerblue','red'])

ax.set(title="Scatter Plot")

ax.set(xlabel="Miles / Gallon")

ax.set(ylabel="Weight of the Car")
car_hp_mpg = pd.melt(df1, id_vars=["car name"],

             value_vars=["horsepower", 'mpg']).sort_values(by=["car name","variable"],ignore_index=True)

car_hp_mpg
max_hp = pd.pivot_table(df1, values='horsepower', index='car name',

            columns='cylinders', aggfunc=np.max)

max_hp
cars = df1

fig = px.pie(cars, values=cars['horsepower'], names=cars['mpg'],

             title='Horse powers vs/ Mileage (mpg)',

            hole=.2, width = 600, height=600)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
fig = px.treemap(cars, path=['car name','horsepower'], values=cars['cylinders'], height=700,

                 width=700, title='Cars: Hp v/s No of cylinders',branchvalues = "total",

                 color_discrete_sequence = px.colors.sequential.Agsunset)

fig.data[0].textinfo = 'label+text+value'

fig.show()