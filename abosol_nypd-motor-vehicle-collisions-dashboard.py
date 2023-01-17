import pandas as pd
import matplotlib.pyplot as plt
import folium
import datetime
df = pd.read_csv("../input/nypd-motor-vehicle-collisions.csv", dtype=str)
#df.info()
idx = df['LATITUDE'].isna() | (df['LATITUDE'] == '0.0000000')
df['time'] = pd.to_datetime(df['DATE'] + " " + df['TIME'])
df.drop(['DATE', 'TIME'], axis=1, inplace=True)
df.columns
numeric_columns = ['LATITUDE',
                   'LONGITUDE',
                   'NUMBER OF PERSONS INJURED',
                   'NUMBER OF PERSONS KILLED',
                   'NUMBER OF PEDESTRIANS INJURED',
                   'NUMBER OF PEDESTRIANS KILLED',
                   'NUMBER OF CYCLIST INJURED',
                   'NUMBER OF CYCLIST KILLED',
                   'NUMBER OF MOTORIST INJURED',
                   'NUMBER OF MOTORIST KILLED']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce') 
df.drop(['LOCATION', 'UNIQUE KEY'], axis=1, inplace=True)
df.info()
df1 = df.loc[~idx] # where LAT/LON are known 
#min_lat, max_lat = df1['LATITUDE'].min(), df1['LATITUDE'].max()
#min_lon, max_lon = df1['LONGITUDE'].min(), df1['LONGITUDE'].max()
#print(min_lat, max_lat, min_lon, max_lon)
now = datetime.datetime.now()
earliest_date = now - datetime.timedelta(hours=24*38)
#print(year_ago)
stage_df = df1[df1['time'] > earliest_date]
stage_df.shape[0]
#import gc
#gc.collect()
# Create a Map instance
m = folium.Map(location=[40.6971494,-74.2598745], tiles='Stamen Toner',
                   zoom_start=10, control_scale=True)

from folium.plugins import MarkerCluster

mc = MarkerCluster()

for each in stage_df.iterrows():
    mc.add_child(folium.Marker(
        location = [each[1]['LATITUDE'],each[1]['LONGITUDE']])) #, 
        #clustered_marker = True)

m.add_child(mc)

display(m)