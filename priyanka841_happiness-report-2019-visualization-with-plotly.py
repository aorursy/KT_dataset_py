
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly_express as px

%matplotlib inline


df = pd.read_csv('../input/2019-world-happiness-report-csv-file/2019.csv')
df.head()
df.tail()
df.shape
df.info()
# we observe that there are no null values and also see the data types
df.describe()
#get the score of top 10 ranking countries and plot them using Plotly
top_10 =df.iloc[ 0:10, 0:3]
top_10
fig = px.pie(top_10, values='Score', names='Country or region', color_discrete_sequence=px.colors.sequential.RdBu, 
             title='Top 10 Country and their score',
             hover_data=['Overall rank'], labels={'Overall rank':'Overall rank'})
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
df1 = df.iloc[0:20]
df1.head()
df1.shape
fig = px.pie(df1, values='GDP per capita', names='Country or region', 
             color_discrete_sequence=px.colors.sequential.RdBu, 
             title='Top 20 Country with GDP score',
             hover_data=['Overall rank'])
             #labels={'Perceptions of corruption':'Perceptions of corruption'}) 
                                                                       
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
fig = px.pie(df1, values='Perceptions of corruption', names='Country or region', 
             color_discrete_sequence=px.colors.sequential.dense, 
             title='Top 20 Country with their Corruption score',
             hover_data=['Overall rank'])
             #labels={'Perceptions of corruption':'Perceptions of corruption'}) 
                                                                       
fig.update_traces(textposition='inside', textinfo='value+label')
fig.show()
df2 =  df.iloc[136:156]
df2.head()
df2.shape
fig = px.pie(df2, values='Perceptions of corruption'
                          , names='GDP per capita', 
             #color='GDP per capita',
             color_discrete_sequence=px.colors.sequential.matter, 
             title='Bottom 20 Country with their GDP and  Corruption score',
             hover_data=['Overall rank'], hover_name='Country or region',
             labels={'Country or region':'Country or region'}) 
                                                                       
fig.update_traces(textposition='inside', textinfo='label')
fig.show()
data_ss = df[df['Overall rank']>=140]
fig = px.bar(data_ss, x='GDP per capita', y='Social support',
             hover_data=['Healthy life expectancy', 'Freedom to make life choices', 'Overall rank'],
             color='Country or region',
             title='Bottom 17 countries with their details' ,
             height=400)
fig.show()
data_top = df[df['Overall rank']<=20]
data_top.shape
fig = px.bar(data_top, x='GDP per capita', y='Social support',
             hover_data=['Healthy life expectancy', 'Freedom to make life choices', 'Overall rank'],
             color='Country or region',
             title='Top 20 countries with their details' ,
             color_discrete_sequence=px.colors.sequential.thermal,                          
             height=600)
fig.show()