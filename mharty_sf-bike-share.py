import pandas as pd

import folium

from folium.plugins import MarkerCluster



import bq_helper

from bq_helper import BigQueryHelper



import geopandas as gpd



sf = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="san_francisco")
sf.list_tables()
sf.head('bikeshare_stations')
sf.head('bikeshare_status')
sf.head('bikeshare_trips')
query_0 = """SELECT count(station_id)

             FROM `bigquery-public-data.san_francisco.bikeshare_stations`

             """

response_0 = sf.query_to_pandas_safe(query_0)

response_0
query_1 = """SELECT * FROM `bigquery-public-data.san_francisco.bikeshare_stations`"""

stations_df = sf.query_to_pandas_safe(query_1)





stations_df['installation_date'] = pd.to_datetime(stations_df['installation_date'])
stations_df.info()
stations_df.groupby('landmark')['station_id'].count().sort_values()
stations_df.resample('Y', on='installation_date')['station_id'].count()
stations_df.resample('M', on='installation_date')['station_id'].count().plot();
m = folium.Map(location=[37.65, -121.9], zoom_start=9)



def add_station_markers_to_map(df):

    popup_text = 'Station: {}\nTotal Docks: {}'

    folium.Marker(location=[df['latitude'], df['longitude']],

                  popup=popup_text.format(df['name'], df['dockcount']),

                  tooltip=popup_text.format(df['name'], df['dockcount'])

                 ).add_to(m)



stations_df.apply(add_station_markers_to_map, axis=1)

m
m = folium.Map(location=[37.65, -121.9], zoom_start=9)



def add_station_clusters_to_map(df):

    popup_text = 'Station: {}\nTotal Docks: {}'

    

    folium.Marker(location=[df['latitude'], df['longitude']],

                  popup=popup_text.format(df['name'], df['dockcount']),

                  tooltip=popup_text.format(df['name'], df['dockcount'])

                 ).add_to(marker_cluster)





marker_cluster = MarkerCluster().add_to(m)

stations_df.apply(add_station_clusters_to_map, axis=1)

m