#Importing Libraries



import pandas as pd

from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

import numpy as np

# Connecting to BigQuery to be able to use the dataset



from google.cloud import bigquery

client = bigquery.Client()

ds_ref = client.dataset('chicago_taxi_trips', project='bigquery-public-data')

ds = client.get_dataset(ds_ref)

tbl = client.get_table(ds.table('taxi_trips'))
sql = """

SELECT

    pickup_latitude,

    pickup_longitude,

    dropoff_latitude,

    dropoff_longitude

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

group by pickup_latitude,

         pickup_longitude,

         dropoff_latitude,

         dropoff_longitude



LIMIT 100000

"""

taxi_raw = client.query(sql).to_dataframe()
lat_max=taxi_raw['pickup_latitude'].max()

lat_min=taxi_raw['pickup_latitude'].min()

long_max=taxi_raw['pickup_longitude'].max()

long_min=taxi_raw['pickup_longitude'].min()



fig = plt.figure(figsize=(12,9))



map = Basemap(projection='mill',

           llcrnrlat = lat_min,

           urcrnrlat = lat_max,

           llcrnrlon = long_min,

           urcrnrlon = long_max,

           resolution = 'c')



map.drawcoastlines()

map.drawmapboundary()

map.drawcountries(linewidth=2)

map.drawstates()



map.scatter(taxi_raw["pickup_longitude"].tolist(),taxi_raw["pickup_latitude"].tolist(),latlon=True)



plt.title('Pick-Up Basemap', fontsize=20)



plt.show()
lat_max=taxi_raw['dropoff_latitude'].max()

lat_min=taxi_raw['dropoff_latitude'].min()

long_max=taxi_raw['dropoff_longitude'].max()

long_min=taxi_raw['dropoff_longitude'].min()



fig = plt.figure(figsize=(12,9))



map = Basemap(projection='mill',

           llcrnrlat = lat_min,

           urcrnrlat = lat_max,

           llcrnrlon = long_min,

           urcrnrlon = long_max,

           resolution = 'c')



map.drawcoastlines()

map.drawmapboundary()

map.drawcountries(linewidth=2)

map.drawstates()



map.scatter(taxi_raw["dropoff_longitude"].tolist(),taxi_raw["dropoff_latitude"].tolist(),latlon=True)



plt.title('dropoff Basemap', fontsize=20)



plt.show()