import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.offline as offline

offline.init_notebook_mode()



# lib for the map

import folium



import os

print(os.listdir("../input"))
# import population dataset and display header

population_df = pd.read_csv("../input/population.csv")

population_df.head()

population_df.info()
total_pop = population_df.loc[population_df['Year'] == 2017]['Number'].sum()

f'Total Population of Barcelona = {total_pop}'
temp_df = population_df.loc[population_df['Year'] == 2017].groupby(['Gender'])['Number'].sum()



trace = go.Pie(labels = temp_df.index,

               values = temp_df.values,

               marker = dict(colors=['#E53916','#2067AD'], line = dict(color='#FFFFFF', width=2.5))

              )



data = [trace]

layout = go.Layout(title="Gender-Wise Distribution for Year-2017")

fig = go.Figure(data=data, layout=layout)



plotly.offline.iplot(fig)
male = population_df.loc[population_df['Gender'] == 'Male'].groupby(['Year'])['Number'].sum()

female = population_df.loc[population_df['Gender'] == 'Female'].groupby(['Year'])['Number'].sum()



trace0 = go.Bar(x = male.index,

                y= male.values,

                name = "Male",

                marker = dict(color='rgb(236,154,41)'),

                opacity = 0.8

               )



trace1 = go.Bar(x = female.index,

                y = female.values,

                name = "Female",

                marker = dict(color='rgb(168,32,26)'),

                opacity = 0.8

               )



data = [trace0,trace1]

layout = go.Layout(barmode = 'group',

                   xaxis = dict(tickangle=-30),

                   title="Gender-Wise Distribution Across the Years",

                      )

fig = go.Figure(data=data,layout=layout)



plotly.offline.iplot(fig)
dist_df = population_df.loc[population_df['Year'] == 2017].groupby(['District.Name'])['Number'].sum()



trace0 = go.Bar(x = dist_df.index,

                y = dist_df.values,

                marker = dict(color=list(dist_df.values),

                                  colorscale='Reds'),

                )



data = [trace0]

layout = go.Layout(xaxis = dict(tickangle=-30),

                   title="District-Wise Distribution of population (Year 2017)",

                      )

fig = go.Figure(data=data,layout=layout)



plotly.offline.iplot(fig)
transport_df =pd.read_csv("../input/transports.csv")

transport_df.head()
metro = transport_df.loc[transport_df['Transport'] == 'Underground']

metro = metro[['Latitude','Longitude','Station']]



barcelona_coordinates = [41.3851, 2.1734]



map_metro = folium.Map(location=barcelona_coordinates, tiles='OpenStreetMap', zoom_start=12)



for elem in metro.iterrows():

    folium.CircleMarker([elem[1]['Latitude'],elem[1]['Longitude']],

                        radius=5,

                        color='blue',

                        popup=elem[1]['Station'],

                        fill=True).add_to(map_metro)

map_metro
busstop_df =pd.read_csv("../input/bus_stops.csv")

busstop_df.head()
bus = busstop_df.loc[busstop_df['Transport'] == 'Day bus stop']



map_bus = folium.Map(location=barcelona_coordinates, tiles='OpenStreetMap', zoom_start=12)



for elem in bus[:100].iterrows():

    folium.Marker([elem[1]['Latitude'],elem[1]['Longitude']],

                  popup=str(elem[1]['Bus.Stop']),

                  icon=folium.Icon(color='blue', icon='stop')).add_to(map_bus)

    

map_bus