# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot # plotly offline mode

init_notebook_mode(connected=True)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#read csv file

data = pd.read_csv("../input/world-happiness-report-2019.csv")
# get some informations about data

data.info()
data.head()
data.tail()
data.describe()
data.columns
#i want to rename columns for easy coding, i prefer change the columns names but you can use the real names of columns

data.rename(columns={'Country (region)':'Country',

                    'SD of Ladder':'SDOfLadder',

                    'Positive affect':'PositiveAffect',

                    'Negative affect':'NegativeAffect',

                    'Social support':'SocialSupport',

                    'Log of GDP\nper capita':'GDPCapita',

                    'Healthy life\nexpectancy':'HealthyLife'},inplace=True)

#new names of columns

data.columns
# make the corrolation map of data

data.corr()
#plot the correlation map

fig,ax=plt.subplots(figsize=(12,6))

sns.heatmap(data.corr(),annot=True,ax=ax,fmt='.1f')# annot= show numbers on squares, fmt= how many number comes after comma

plt.show()

# this correlation map gives the direct proportion between variables(columns) of data
data.plot(subplots=True,grid=True,figsize=(15,15))

plt.show()
# list of countries

data
# make the map of unhappy countries

map_data = [go.Choropleth( 

           locations = data['Country'], #Sets the coordinates via location names

           locationmode = 'country names',# Determines the set of locations used to match entries in `locations` to regions on the map.

           z = data["Ladder"], #Sets the color values

           text = data['Country'],#Sets the text elements associated with each location.

           colorbar = {'title':'Ladder Rank'})]



layout = dict(title = 'Unhappiest Level Of Countries', 

             geo = dict(showframe = True, 

                       projection = dict(type = 'equirectangular')))



world_map = go.Figure(data=map_data, layout=layout)

iplot(world_map)
data.GDPCapita.plot(kind="line",grid=True,color="blue",label="GDP problem",linestyle=":",alpha=0.75,figsize=(7,7))

plt.legend()

plt.xlabel("Ladder Rank")

plt.show()
gdpCapita = data.sort_values(by="GDPCapita")

gdpCapita
map_data = [go.Choropleth( 

           locations = data['Country'],

           locationmode = 'country names',

           z = data["GDPCapita"], 

           text = data['Country'],

           colorbar = {'title':'Ladder Rank'})]



layout = dict(title = 'GDP Problem Of Countries', 

             geo = dict(showframe = True, 

                       projection = dict(type = 'equirectangular')))



world_map = go.Figure(data=map_data, layout=layout)

iplot(world_map)
data.Freedom.plot(kind="line",grid=True,color="red",label="Freedom Problem",linestyle=":",alpha=0.75,figsize=(7,7))

plt.legend()

plt.xlabel("Ladder Rank")

plt.show()
freedom_data= data.sort_values(by="Freedom")

freedom_data
map_data = [go.Choropleth( 

           locations = data['Country'],

           locationmode = 'country names',

           z = data["Freedom"], 

           text = data['Country'],

           colorbar = {'title':'Ladder Rank'})]



layout = dict(title = 'Countries With Least Freedom', 

             geo = dict(showframe = True, 

                       projection = dict(type = 'equirectangular')))



world_map = go.Figure(data=map_data, layout=layout)

iplot(world_map)
data.SocialSupport.plot(kind="line",grid=True,color="green",label="Social Support Problem",linestyle=":",alpha=0.75,figsize=(7,7))

plt.legend()

plt.xlabel("Ladder Rank")

plt.show()
# sort the data

social_data = data.sort_values(by="SocialSupport")

social_data
map_data = [go.Choropleth( 

           locations = data['Country'],

           locationmode = 'country names',

           z = data["SocialSupport"], 

           text = data['Country'],

           colorbar = {'title':'Ladder Rank'})]



layout = dict(title = 'Countries With Least Social Support', 

             geo = dict(showframe = True, 

                       projection = dict(type = 'equirectangular')))



world_map = go.Figure(data=map_data, layout=layout)

iplot(world_map)
data.Corruption.plot(kind="line",grid=True,color="purple",label="Corruption",linestyle=":",alpha=0.75,figsize=(7,7))

plt.legend()

plt.xlabel("Ladder Rank")

plt.show()
# sort the data

corruption_data = data.sort_values(by="Corruption")

corruption_data
map_data = [go.Choropleth( 

           locations = data['Country'],

           locationmode = 'country names',

           z = data["Corruption"], 

           text = data['Country'],

           colorbar = {'title':'Ladder Rank'})]



layout = dict(title = 'Corruption Level Of Countries', 

             geo = dict(showframe = True, 

                       projection = dict(type = 'equirectangular')))



world_map = go.Figure(data=map_data, layout=layout)

iplot(world_map)
data.HealthyLife.plot(kind="line",grid=True,color="navy",label="Health Problem",linestyle=":",alpha=0.75,figsize=(7,7))

plt.legend()

plt.xlabel("Ladder Rank")

plt.show()
health_data=data.sort_values(by="HealthyLife")

health_data
map_data = [go.Choropleth( 

           locations = data['Country'],

           locationmode = 'country names',

           z = data["HealthyLife"], 

           text = data['Country'],

           colorbar = {'title':'Ladder Rank'})]



layout = dict(title = 'Health Problems Of Countries', 

             geo = dict(showframe = True, 

                       projection = dict(type = 'equirectangular')))



world_map = go.Figure(data=map_data, layout=layout)

iplot(world_map)