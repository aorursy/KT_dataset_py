import os
import bq_helper
import pandas as pd
import plotly.plotly as py
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
air_table = bq_helper.BigQueryHelper("bigquery-public-data","openaq", "global_air_quality")
air_table.list_tables()
air_table.table_schema("global_air_quality")
air_table.head("global_air_quality")
ww_co_query = """
SELECT latitude, longitude, value
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE pollutant = 'co'
"""
worldwide_co = air_table.query_to_pandas_safe(ww_co_query)
worldwide_co.tail()
worldwide_co.describe()
# Replace negative values with zero
num = worldwide_co.value._get_numeric_data()

num[num < 0] = 0
worldwide_co.value = num
worldwide_co.describe()
# Normalize value by subtracting the mean and dividing by standard deviation
worldwide_co['value'] = (worldwide_co['value'] - worldwide_co['value'].mean()) / worldwide_co['value'].std()
worldwide_co.describe()
fig, ax = plt.subplots(figsize=(26,12))
earth = Basemap(ax=ax)
earth.drawcoastlines(color='#556655', linewidth=0.5)
ax.scatter(worldwide_co['longitude'], worldwide_co['latitude'], s=worldwide_co['value'], 
           c='red', alpha=1, zorder=10)
ax.set_title("CO Worldwide", fontsize=18)
fig.savefig('worldwide_carbon_levels.png')
# Now lets check out the state feature
import collections

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


a = worldwide_co.value.values
counter=collections.Counter(a)

key = list(counter.keys())
population = list(counter.values())

scale=[[0, '#84FFFF'], [0.25, '#00E5FF'], [0.65, '#40C4FF'],[1, '#01579B']]


dataa = [ dict(
        type='choropleth',
        colorscale = scale,
        locations = key,
        z = population,
        locationmode = "USA-states",
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Number of Companies")
        ) ]

layout = dict(
        title = 'Frequency of companies by state<br>(Hover for number of cos)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=dataa, layout=layout )
py.iplot( fig, filename='statefreq' )
import plotly.plotly as py
import pandas as pd

df = worldwide_co 

data = [ dict(
        type = 'choropleth',
        locations = df['CODE'],
        z = df['GDP (BILLIONS)'],
        text = df['COUNTRY'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '$',
            title = 'GDP<br>Billions US$'),
      ) ]

layout = dict(
    title = '2014 Global GDP<br>Source:\
            <a href="https://www.cia.gov/library/publications/the-world-factbook/fields/2195.html">\
            CIA World Factbook</a>',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='d3-world-map' )
x = worldwide_co['longitude']
y = worldwide_co['latitude']
xx, yy = np.meshgrid(x, y, sparse=True)
listy = [xx,yy]
listy[1]
import os
import matplotlib.pyplot as plt
from netCDF4 import Dataset as netcdf_dataset
import numpy as np

from cartopy import config
import cartopy.crs as ccrs

ax = plt.axes(projection=ccrs.Mollweide())
ax.stock_img()

plt.contour(worldwide_co['longitude'].T, worldwide_co['latitude'], worldwide_co.value.values, transform=ccrs.Mollweide())
#plt(x, y, 'bo')

'''x = worldwide_co['longitude']
y = worldwide_co['latitude']
xx, yy = np.meshgrid(x, y, sparse=True)
z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
h = plt.contourf(x,y,z)'''
plt.figure()

ax = plt.axes(projection=ccrs.Mollweide())
#ax.gridlines(color='gray', linestyle='--')
ax.stock_img()
ax.set_global()

plt.pcolormesh([worldwide_co['longitude'], worldwide_co['latitude']], worldwide_co['value'].values)
plt.tight_layout()
plt.show()
low_co_cities = """
SELECT city, country, pollutant, value, latitude, longitude
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE 
pollutant = 'co' AND
value < 1
"""
lowest_co_cities = air_table.query_to_pandas_safe(low_co_cities)
lowest_co_cities[0:100]
fig, ax = plt.subplots(figsize=(26,12))
earth = Basemap(ax=ax)
earth.drawcoastlines(color='#556655', linewidth=0.5)
ax.scatter(lowest_co_cities['longitude'], lowest_co_cities['latitude'], lowest_co_cities['value'], 
           c='green', alpha=.5, zorder=10)
ax.set_title("Lowest CO Cities", fontsize=18)
fig.savefig('low_carbon_cities.png')
high_co_cities = """
SELECT city, country, pollutant, value, latitude, longitude
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE 
pollutant = 'co'
ORDER BY value DESC
LIMIT 50
"""
highest_co_cities = air_table.query_to_pandas_safe(high_co_cities)
highest_co_cities.head()
fig, ax = plt.subplots(figsize=(26,12))
earth = Basemap(ax=ax)
earth.drawcoastlines(color='#556655', linewidth=0.5)
ax.scatter(highest_co_cities['longitude'], highest_co_cities['latitude'], highest_co_cities['value'], 
           c='red', alpha=0.5, zorder=10)
ax.set_title("Highest CO Cities", fontsize=18)
fig.savefig('high_carbon_cities.png')
