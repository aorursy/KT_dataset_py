import numpy as np

import pandas as pd 

import plotly.offline as pyo

import plotly.plotly as py

from plotly.graph_objs import *

pyo.offline.init_notebook_mode()
df = pd.read_csv('../input/bikes.csv')

df.head()
print(df[df['banking']==False])

print(df[df['bonus']==True])
df =df[df.status=='OPEN']

df['useage_percent'] = 1 - df['available_bikes']/df['bike_stands']
df2 = df.groupby(by='name')['bike_stands'].mean()

pd.DataFrame(df2).sort_values(by=['bike_stands'], ascending=False)
df1 = df.groupby(by='name')['useage_percent'].mean()

pd.DataFrame(df1).sort_values(by=['useage_percent'], ascending=False)
df['month'] = pd.to_datetime(df['last_update'],unit='ms').dt.date

df['month'] = df['month'].apply(lambda x: int(str(x)[5:7]))

Month2 = df.groupby(by=['month','number'])['available_bikes'].mean().reset_index()

traces =[]

for col in Month2.number.unique().tolist():

    traces.append({'type': 'scatter',

     'mode': 'lines',

     'name': col,

     'x': list(Month2['month'].unique()),

     'y': list(Month2[Month2['number']==col]['available_bikes'])})

data = Data(traces)

layout = {'title': ' Sharing Bike Data in Santander for 17 stations from May to November ',

         'xaxis' : {'title' : 'Month'},

         'yaxis' : {'title' : 'Available Bikes'}}

fig = Figure(data=data, layout=layout)

pyo.iplot(fig)
Month1 = df.groupby(by=['month','number'])['useage_percent'].mean().reset_index()

traces =[]

for col in Month1.number.unique().tolist():

    traces.append({'type': 'scatter',

     'mode': 'lines',

     'name': col,

     'x': list(Month1['month'].unique()),

     'y': list(Month1[Month1['number']==col]['useage_percent'])})

data = Data(traces)

layout = {'title': ' Sharing Bike Data in Santander for 17 stations from May to November ',

         'xaxis' : {'title' : 'Month'},

         'yaxis' : {'title' : 'Useage Percent'}}

fig = Figure(data=data, layout=layout)

pyo.iplot(fig)
Bike = pd.DataFrame()

Bike['name'] = list(df2.index)

Bike['bike_stands'] = list(df2)

Bike['useage_percent'] = list(df1)

Bike['lat'] = list(df.groupby(by='name')['lat'].mean())

Bike['lon'] = list(df.groupby(by='name')['lng'].mean())

mapbox_access_token = 'pk.eyJ1IjoibXRhbmc0OSIsImEiOiJjajl6N2pwOHQ4b28yMndzNHhmM3FvcTZqIn0.SitEQvdYa_z81ZEJx14aUQ'

scl = [[0, 'rgb(150,0,90)'],[0.125, 'rgb(0, 0, 200)'],[0.25,'rgb(0, 25, 255)'],[0.375,'rgb(0, 152, 255)'],[0.5,'rgb(44, 255, 150)'],[0.625,'rgb(151, 255, 0)'],[0.75,'rgb(255, 234, 0)'],[0.875,'rgb(255, 111, 0)'],[1,'rgb(255, 0, 0)']]

data = Data([

    Scattermapbox(

        lat=Bike['lat'],

        lon=Bike['lon'],

        mode='markers',

        marker=dict(

              color=Bike['useage_percent'],

              colorscale=scl,

              cmin=0,

              cmax=1,

              reversescale=True,

              opacity=0.9,

              size=11,

             colorbar=dict(

            thickness=10,

            titleside='right',

            outlinecolor='rgba(68,68,68,0)',

            ticks='outside',

            ticklen=3,

            ticksuffix=' Useage Percentage',

            dtick=0.1

             )

        ),

        text=Bike['name'],        

    )

])



layout = Layout(

    autosize=True,

    hovermode='closest',

    mapbox=dict(

        style='light',

        accesstoken=mapbox_access_token,

        bearing=0,

        center=dict(

            lat=43.47,

            lon=-3.788

        ),

        pitch=0,

        zoom=10

    ),

)

fig = dict(data=data, layout=layout)

pyo.iplot(fig, filename='Multiple Mapbox')
scl = [[0, 'rgb(150,0,90)'],[5, 'rgb(0, 0, 200)'],[10,'rgb(0, 25, 255)'],[15,'rgb(0, 152, 255)'],[20,'rgb(44, 255, 150)'],[25,'rgb(151, 255, 0)'],[30,'rgb(255, 234, 0)'],[35,'rgb(255, 111, 0)'],[40,'rgb(255, 0, 0)']]

data = Data([

    Scattermapbox(

        lat=Bike['lat'],

        lon=Bike['lon'],

        mode='markers',

        marker=dict(

              color=Bike['bike_stands'],

              colorscale=scl,

              cmin=0,

              cmax=40,

              reversescale=True,

              opacity=0.9,

              size=11,

             colorbar=dict(

            thickness=10,

            titleside='right',

            outlinecolor='rgba(68,68,68,0)',

            ticks='outside',

            ticklen=3,

            ticksuffix='Bike Stands',

            dtick=5

             )

        ),

        text=Bike['name'],        

    )

])



layout = Layout(

    autosize=True,

    hovermode='closest',

    mapbox=dict(

        style='light',

        accesstoken=mapbox_access_token,

        bearing=0,

        center=dict(

            lat=43.47,

            lon=-3.788

        ),

        pitch=0,

        zoom=10

    ),

)

fig = dict(data=data, layout=layout)

pyo.iplot(fig, filename='Multiple Mapbox')