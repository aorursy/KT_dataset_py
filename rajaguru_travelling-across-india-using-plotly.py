import pandas as pd
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
from IPython.display import HTML, Image

import numpy as np
import os
import sys 
pd.options.display.html.table_schema = True
pd.options.display.max_rows = None
init_notebook_mode(connected = True)

os.listdir("../input")
DIR=r"../input"
indiancities=os.path.join(DIR,'cities_r2.csv')
indianCities=pd.read_csv(indiancities)
indianCities.head()
trace1 = go.Scatter(
                    x = indianCities.name_of_city,
                    y = indianCities.population_male,
                    mode = "lines",
                    name = "citations",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text= indianCities.name_of_city) 
# Creating trace2
trace2 = go.Scatter(
                    x = indianCities.name_of_city,
                    y = indianCities.population_female,
                    mode = "lines+markers",
                    name = "teaching",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    )
data = [trace1, trace2]
layout = dict(title = 'Citation and Teaching vs World Rank of Top 100 Universities',
              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)

indianCities.columns.tolist() 
lat=[]
lon=[]
for x in indianCities['location']:
  a=x.split(',')
  lat.append(float(a[0]))
  lon.append(float(a[1]))
  
mapbox_access_token="pk.eyJ1IjoiZ3VydW5hdGgwNSIsImEiOiJjanNuYmpyeDIwYWx2NGFsanZoejJsMzRwIn0.LhyRdspSIhxn-1s5R_9KLQ"
data = [
    go.Scattermapbox(
        lat=lat,
        lon=lon,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=indianCities['population_total']//170000,
            color=indianCities['state_code'],
        ),
        
        text=indianCities['state_name']+" :- "+indianCities['name_of_city'] +" Total population "+indianCities['population_total'].astype(str)
                                    +" Male population "+indianCities['population_male'].astype(str)+" Female population "+
                                    indianCities['population_female'].astype(str),
    )
]

layout = go.Layout(
    autosize=True,
    width=1000,
    height=1000,
    hovermode='closest',
    mapbox=go.layout.Mapbox(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=go.layout.mapbox.Center(
            lat=20.5937,
            lon=78.9629,
        ),
        pitch=0,
        zoom=4,
    ),
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)

indianCities['state_name'].value_counts()
indianCities.columns[4:].tolist()
indianCities[indianCities['name_of_city'].str.contains('Chennai')].values[:,4:].ravel().tolist()
data=[go.Scatterpolar(
      name = "Tambaram",
      r = indianCities[indianCities['name_of_city'].str.contains('Tambaram')].values[:,4:].ravel().tolist(),
      theta =indianCities.columns[4:].tolist(),
      fill = "toself",
    ),
      
    go.Scatterpolar(
      name = "Pallavaram",
      r = indianCities[indianCities['name_of_city'].str.contains('Pallavaram')].values[:,4:].ravel().tolist(),
      theta =indianCities.columns[4:].tolist(),
      fill = "toself",
    ),
     ]
layout = go.Layout(
    polar = dict(
        radialaxis = dict(
            visible = True,
        )
    ),
    showlegend = True
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
numerical_columns=indianCities.select_dtypes('int64').columns.tolist()+indianCities.select_dtypes('float64').columns.tolist()
indianStates=indianCities.groupby('state_name').sum()[numerical_columns].reset_index()
indianStates.head()
indianStates.columns[3:].tolist()
data=[go.Scatterpolar(
      name = state,
      r = indianStates[indianStates['state_name'].str.contains(state)].values[:,3:].ravel().tolist(),
      theta =indianStates.columns[3:].tolist(),
      fill = "toself",
    ) for state in indianStates['state_name']]
layout = go.Layout(
    title="All states comparison",
    polar = dict(
        radialaxis = dict(
            visible = False,
        )
    ),
    showlegend = True
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
data = [
    go.Scattermapbox(
        lat=lat,
        lon=lon,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=indianCities['population_total']//170000,
            color=indianCities['state_code'],
        ),
        
        text=indianCities['state_name']+" :- "+indianCities['name_of_city'] +" Total population "+indianCities['population_total'].astype(str)
                                    +" Male population "+indianCities['population_male'].astype(str)+" Female population "+
                                    indianCities['population_female'].astype(str),
    )
]

layout = go.Layout(
    autosize=True,
    width=1000,
    height=1000,
    hovermode='closest',
    mapbox=go.layout.Mapbox(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=go.layout.mapbox.Center(
            lat=11.059821,
            lon=78.387451,
        ),
        pitch=0,
        zoom=4,
    ),
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)

tamilnaduCities=indianCities[indianCities['state_name']=='TAMIL NADU']
tamilnaduCities.head()
data=[go.Scatterpolar(
      name = city,
      r = tamilnaduCities[tamilnaduCities['name_of_city'].str.contains(city)].values[:,3:].ravel().tolist(),
      theta =tamilnaduCities.columns[3:].tolist(),
      fill = "toself",
    ) for city in tamilnaduCities['name_of_city']]
layout = go.Layout(
    title="Tamil Nadu cities comparison",
    polar = dict(
        radialaxis = dict(
            visible = False,
        )
    ),
    showlegend = True
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
reduced_columns=['population_total','literates_total','sex_ratio','effective_literacy_rate_total','total_graduates']
data=[go.Scatterpolar(
      name = city,
      r = tamilnaduCities[tamilnaduCities['name_of_city'].str.contains(city)][reduced_columns].values.ravel().tolist(),
      theta =reduced_columns,
      fill = "toself",
    ) for city in tamilnaduCities['name_of_city']]
layout = go.Layout(
    title="Tamil Nadu cities comparison",
    polar = dict(
        radialaxis = dict(
            visible = False,
        )
    ),
    showlegend = True
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
indianStates
indianStates
indianStates
indianStates
indianCities
indianCities
indianCities
indianStates
indianStates
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('abc').getOrCreate()
india = spark.read.csv("/home/user/windstream_dtnx/WS_Accuracy/experiments/practice_data_science-master/practice_datasets/cities_r2.csv",header=True)
india.show()
