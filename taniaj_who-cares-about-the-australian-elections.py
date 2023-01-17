# Install latest folium because popup doesn't seem to work with current Kaggle folium 0.5.0

!pip install folium==0.9.0
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib

import folium

from folium import IFrame



TWEET_FILE = 'auspol2019.csv'

GEOCODE_FILE = 'location_geocode.csv'

INPUT_PATH = '../input/'

LOCATION_GEOCODE_DATA = '../data/location_geocode.csv'
# Read in the tweet file

df = pd.read_csv(INPUT_PATH + TWEET_FILE)

df.shape
# Get a count of tweets grouped by location

loc_grouped_df = df[['user_location', 'id']].groupby('user_location', sort=False).count().sort_values(['id'], ascending=False)

loc_grouped_df.columns=['count']

loc_grouped_df.reset_index(inplace=True)

loc_grouped_df.head(10)
# Read in the geocode file

geocode_df = pd.read_csv(INPUT_PATH + GEOCODE_FILE)

geocode_df.shape
# Add the count column to the geocode dataframe

geocode_df = geocode_df.merge(loc_grouped_df, how='left', left_on='name', right_on='user_location')

geocode_df.shape
# Number of locations that were not found

geocode_df[geocode_df['lat'].isnull()].shape
# Remove nulls

geocode_df.dropna(inplace=True)

geocode_df.shape
# Remove non-alphanum characters from names to avoid breaking folium

geocode_df['name'] = geocode_df['name'].str.replace('[^a-zA-Z\s,]', '')
# Make an empty Folium map



m = folium.Map(location=[20,0], tiles="Mapbox Bright", zoom_start=2)



# Add markers one by one

for i in range(0,len(geocode_df)):

    folium.Circle(

      # .item() to avoid json errors by converting numpy.int64 to regular ints.

      location=[geocode_df.iloc[i]['lat'].item(), geocode_df.iloc[i]['long'].item()], 

      #popup=geocode_df.iloc[i]['name'] + ' ' +str(geocode_df.iloc[i]['count'].item()),

      radius=geocode_df.iloc[i]['count'].item()*100,

      color='crimson',

      fill=True,

      fill_color='crimson'

    ).add_to(m)



# Save and display it

m.save('ausvotes_map.html')

m
