# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

import plotly.graph_objs as go

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print()

print()

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Let's read the dataset.

try:

    df_country = pd.read_csv('../input/GlobalLandTemperaturesByCountry.csv')

    df_state = pd.read_csv('../input/GlobalLandTemperaturesByState.csv')

    df_global = pd.read_csv('../input/GlobalTemperatures.csv')

    df_city = pd.read_csv('../input/GlobalLandTemperaturesByCity.csv')

    df_major_city = pd.read_csv('../input/GlobalLandTemperaturesByMajorCity.csv')



except Exception as e :

    pass

df_country.head()
df_state.head()
df_city.head()
df_global.head()
df_global['year'] = pd.to_datetime(df_global['dt']).dt.year

by_year = df_global.groupby('year')['LandAverageTemperature'].mean().reset_index()

by_year.iplot(kind='scatter', x='year', y='LandAverageTemperature', title='Temperature trend',

              xTitle='Year', yTitle='Temperature')
hot_country = df_country.groupby('Country')['AverageTemperature'].mean().reset_index()

hot_country.sort_values('AverageTemperature', ascending=False, inplace=True)

hot_country.reset_index(drop=True, inplace=True)

hot_country = hot_country[:10]





code = ['DJI', 'ML','BFA','SEN','ABW','ARE','MRT','GMB','NER','CUW']

country = hot_country['Country']

temp = hot_country['AverageTemperature']



data = dict(

type = 'choropleth',

locations = code,

z = temp,

text = country,

colorbar = {'title' : 'Temperature'},

)



layout = dict(

title = 'Top 10 hottest Countries',

geo = dict(

showframe = False,

projection = {'type':'natural earth'}

)

)



choromap = go.Figure(data = [data],layout = layout)

iplot(choromap)



hot_country.T
cool_country = df_country.groupby('Country')['AverageTemperature'].mean().reset_index()

cool_country.sort_values('AverageTemperature', inplace=True)

cool_country.reset_index(drop=True, inplace=True)

cool_country = cool_country[:10]





code = ['GRL','DNK','SJM','RUS','CAN','MNG','NOR','FIN','SGS','ISL']

country = cool_country['Country']

temp = cool_country['AverageTemperature']



data = dict(

type = 'choropleth',

locations = code,

z = temp,

text = country,

colorbar = {'title' : 'Temperature'},

)



layout = dict(

title = 'Top 10 Coolest Countries',

geo = dict(

showframe = False,

projection = {'type':'natural earth'}

)

)



choromap = go.Figure(data = [data],layout = layout)

iplot(choromap)



cool_country.T
country_top = ['India','United States','China','Japan','Russia']

new_df = df_country[df_country['Country'].isin(country_top)]

#print(new_df['Country'].unique())



new_df['year'] = pd.to_datetime(new_df['dt']).dt.year



new_df = new_df.groupby(['Country','year'])['AverageTemperature'].mean().reset_index()

new_df



new_pivot = new_df.pivot_table(values='AverageTemperature', index='year', columns='Country')

new_pivot.iplot(kind='scatter', title='Temperature trend in Top 5 Economic Countries.',

                xTitle='Year', yTitle='Temperature')
states = df_state.groupby(['State','Country'])['AverageTemperature'].mean().reset_index()

states.sort_values('AverageTemperature',inplace=True, ascending=False)

print(states[:10])



name_states = states['State'][:10]

name_states



temp_df = df_state[df_state['State'].isin(name_states)]

temp_df['year'] = pd.to_datetime(temp_df['dt']).dt.year

state_pivot = temp_df.pivot_table(values='AverageTemperature', index='year', columns='State')

state_pivot.iplot(kind='scatter', title='Temperature trend of top 10 states', 

                  xTitle='Year', yTitle='Temperature')

india = df_country[df_country['Country']=='India']

india['year'] = pd.to_datetime(india['dt']).dt.year



new_india = india.groupby('year')['AverageTemperature'].mean().reset_index()

new_india.iplot(kind='scatter', x='year', y='AverageTemperature', title='Temperature trend in India',

               xTitle='Year', yTitle='Temperature')
state = df_state[df_state['Country']=='India']

state = state.groupby('State')['AverageTemperature'].mean().reset_index()

state.sort_values('AverageTemperature',inplace=True, )

state = state[:10]

state.iplot(kind='bar', x='State', y='AverageTemperature', title='Top 10 Coolest States',

           xTitle='State', yTitle='Temperature', color='deepskyblue')
state = df_state[df_state['Country']=='India']

state = state.groupby('State')['AverageTemperature'].mean().reset_index()

state.sort_values('AverageTemperature',inplace=True, ascending=False)

state = state[:10]

state.iplot(kind='bar', x='State', y='AverageTemperature', title='Top 10 Hotest States',

           xTitle='State', yTitle='Temperature')
temp_df = df_city[df_city['City']== 'Jodhpur'   ]

temp_df['year'] = pd.to_datetime(temp_df['dt']).dt.year



by_year = temp_df.groupby('year')['AverageTemperature'].mean().reset_index()

by_year.iplot(kind='scatter', x='year', y='AverageTemperature', title='Temperature trend of Jodhpur City in India',

             xTitle='Year', yTitle='Temperature', legend=True)
new_df = df_major_city[df_major_city['Country']=='India']

new_df['year'] = pd.to_datetime( new_df['dt']).dt.year # Converting date into year and making new column.



by_new = new_df.groupby(['City','year'] )['AverageTemperature'].mean().reset_index()

new_pivot = by_new.pivot_table(values='AverageTemperature', index='year', columns='City')

new_pivot.iplot(kind='scatter')
city = df_major_city[df_major_city['Country']=='India']

city['month'] = pd.to_datetime(city['dt']).dt.month

city['month'] = city['month'].map({1:'January', 2:'Februay', 3:'March',4:'April', 5:'May', 6:'June', 

                                  7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'})

city_pivot = city.pivot_table(values='AverageTemperature', index='month', columns='City')

city_pivot.iplot(kind='heatmap',colorscale= 'YlOrRd')
city = df_major_city[df_major_city['Country']=='India']

by_city = city.groupby(['City'])['AverageTemperature'].mean().reset_index()

#print(by_city)



name = by_city['City']

temp = by_city['AverageTemperature']

lat = [23.0225, 12.9716, 19.0760, 22.5726, 28.7041, 17.3850, 26.9124, 26.4499, 26.8467, 13.0827, 21.1458, 28.6139, 18.5204,21.1702 ]

lon = [72.5714, 77.5946, 72.8777, 88.3639, 77.1025, 78.4867, 75.7873, 80.3319, 80.9462, 80.2707, 79.0882, 77.2090, 73.8567, 72.8311]





data = [{'lat': lat ,

  'lon': lon ,       

  'marker': {'color': temp ,

   'line': {'color': 'rgb(40,40,40)', 'width': 0.5},

   'size': 5,

   'sizemode': 'diameter',

    'colorbar': dict(

            title = 'Temperature', 

            thickness = 10,           

            outlinecolor = "rgba(68, 68, 68, 0)",            

            ticklen = 3,                       

            dtick = 0.1      )        },

  'text': name.astype(str) + '  ->  ' ,

  'type': 'scattergeo',

  

      }]





layout = go.Layout(

    title = 'Average Temperature of Major Cities of India',

    showlegend = True,

    geo = dict(

            scope='asia',

            projection=dict( type = 'natural earth'),

            showland = True,

            landcolor = 'rgb(217, 217, 217)',

            subunitwidth=1,

            countrywidth=1,

            subunitcolor="rgb(255, 255, 255)",

            countrycolor="rgb(255, 255, 255)"

        ),)



fig =  go.Figure(layout=layout, data=data)

iplot( fig, validate=False)