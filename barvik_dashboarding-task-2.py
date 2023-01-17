import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv ('../input/road-weather-information-stations.csv')

print(df.columns)
#two words about data

df.head(5)
#get hist for correct values based on analysis in previous kernel

mask = df['RoadSurfaceTemperature'] > 24 

mask &= df['AirTemperature'] > 30

mask &= df['AirTemperature'] < 100

rst = df[mask]

rst.hist(column = ['RoadSurfaceTemperature', 'AirTemperature'], sharex=True, sharey=True)
rst['Date'] = pd.to_datetime(rst['DateTime'])

temp_ave = rst[['AirTemperature', 'RoadSurfaceTemperature']].groupby([rst.Date.dt.year, rst.Date.dt.month, rst.Date.dt.day]).agg('mean')

temp_ave['date'] = temp_ave.index

temp_ave['date'] = pd.to_datetime(temp_ave['date'], format="(%Y, %m, %d)")

temp_ave = temp_ave.reset_index(drop = True)

temp_ave.head()
import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot



init_notebook_mode()



data = [

    go.Scatter(x=temp_ave.date, y=temp_ave.AirTemperature, name='Air')

    , go.Scatter(x=temp_ave.date, y=temp_ave.RoadSurfaceTemperature, name='Road Surface')

]

layout = dict(title='Average Temperature',

              xaxis=dict(title='Date'),

              yaxis=dict(title='Temperature, F'))

fig = dict(data = data, layout = layout)

iplot(fig)
loc = rst['StationLocation']

coord = pd.DataFrame(list(loc.apply(lambda x: (float((eval(x))['longitude']), float((eval(x))['latitude'])))), columns=['longitude','latitude'])

coord = coord.groupby('latitude').agg('min')

coord['latitude'] = coord.index

coord.reset_index(drop=True)
coord.tail(3)
import folium

MAP_COORDINATES = (coord['latitude'].max(), coord['longitude'].max())



map = folium.Map(location=MAP_COORDINATES, zoom_start=10)

for c in coord.iterrows() :

    folium.Marker([c[1]['latitude'],c[1]['longitude']]).add_to(map)

    

display (map)