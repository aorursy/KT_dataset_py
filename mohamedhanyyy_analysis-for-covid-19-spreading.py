# import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data = pd.read_excel('../input/covid19-worldspreading/Covid-19_.xlsx')
df = pd.DataFrame(data)
df
# Country/Region contain all world Countries 

# Confirmed contain all confirmed covid-19 cases 

# Country Abbr 2 contain every country with the abbreviation of 2 letter

# Country Abbr 3 contain every country with the abbreviation of 3 letter

# this 2 columns are useful to use Choropleth with plotly to make the world map

df.columns
# check for null Values

df.isnull().sum()
plt.scatter(df['Confirmed'].iloc[:10],df['Country/Region'].iloc[:10],color='r')
fig = plt.figure(figsize=(7,16))

fig = plt.plot(df['Confirmed'].loc[:50],df['Country/Region'].loc[:50],color='green',lw = 2,ls='--',marker= 'o',markerfacecolor='red')

fig
# if you want to search for a specific country 

df.loc[df['Country/Region'] == 'France']
# Plotly is not working on kaggle notebook , this graph will show the spreading of the covid-19 in world map (choropleth)

import plotly.plotly as py

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

data_ = dict(type ='choropleth',locations =df['Country Abbr 3'],z = df['Confirmed'],

        colorbar = {'title': 'Covid-19 Spreading'})

layout = dict(title = 'Covid-19 World Spreading Analysis', geo = dict(showframe = False))

chronomap = go.Figure(data=[data_], layout=layout)

plot(chronomap)

# it will look like the image bellow