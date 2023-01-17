# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing necessery libraries for future analysis of the dataset

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import folium
from folium.plugins import HeatMapWithTime, TimestampedGeoJson
# Pandas to read Covid Tweets dataset
tweets = pd.read_csv('../input/covid19-tweets/covid19_tweets.csv')
# Examining the dataset from begining
tweets.head()
# Checking the shape of dataset
tweets.shape

tweets.info()
# World City Dataset

cities = pd.read_csv('../input/world-cities-datasets/worldcities.csv')
# Exploring city dataset
cities.head()
## Empty Columns of Latitude and Longitudes are added in Tweets Dataset

tweets["lat"] = np.NaN
tweets["lng"] = np.NaN
tweets["location"] = tweets["user_location"]

user_location = tweets['location'].fillna(value='').str.split(',')
# AVG Location Dataset

avg_countries_loc = pd.read_csv('https://gist.githubusercontent.com/tadast/8827699/raw/3cd639fa34eec5067080a61c69e3ae25e3076abb/countries_codes_and_coordinates.csv')
avg_countries_loc.head()
# Make a list of all countries in Avg Location Dataset
codes = avg_countries_loc['Alpha-2 code'].str.replace('"','').str.strip().to_list()
world_city_iso2 = []
for c in cities['iso2'].str.lower().str.strip().values.tolist():
    if c not in world_city_iso2:
        world_city_iso2.append(c)
        
# Try to identify if both external share same countries codes for tracking between them
l_codes = [c.lower() for c in codes]
for a in world_city_iso2:
    if a not in l_codes:
        print(a)
# Adding the missing country codes manually

codes = avg_countries_loc['Alpha-2 code'].str.replace('"','').str.strip().to_list() + ['XW','SX', 'CW','XK']
code_lat = avg_countries_loc['Latitude (average)'].str.replace('"','').to_list() + ['31.953112', '18.0255', '12.2004', '42.609778']
code_lng = avg_countries_loc['Longitude (average)'].str.replace('"','').to_list() + ['35.301170', '-63.0450', '-69.0200', '20.918062']
lat = cities['lat'].fillna(value = '').values.tolist()
lng = cities['lng'].fillna(value = '').values.tolist()


# Getting all alpha 3 codes into  a list
world_city_iso3 = []
for c in cities['iso3'].str.lower().str.strip().values.tolist():
    if c not in world_city_iso3:
        world_city_iso3.append(c)
        
# Getting all alpha 2 codes into  a list    
world_city_iso2 = []
for c in cities['iso2'].str.lower().str.strip().values.tolist():
    if c not in world_city_iso2:
        world_city_iso2.append(c)
        
# Getting all countries into  a list        
world_city_country = []
for c in cities['country'].str.lower().str.strip().values.tolist():
    if c not in world_city_country:
        world_city_country.append(c)

# Getting all amdin names into  a list
world_states = []
for c in cities['admin_name'].str.lower().str.strip().tolist():
    world_states.append(c)


# Getting all cities into  a list
world_city = cities['city'].fillna(value = '').str.lower().str.strip().values.tolist()



for each_loc in range(len(user_location)):
    ind = each_loc
    order = [False,False,False,False,False]
    each_loc = user_location[each_loc]
    for each in each_loc:
        each = each.lower().strip()
        if each in world_city:
            order[0] = world_city.index(each)
        if each in world_states:
            order[1] = world_states.index(each)
        if each in world_city_country:
            order[2] = world_city_country.index(each)
        if each in world_city_iso2:
            order[3] = world_city_iso2.index(each)
        if each in world_city_iso3:
            order[4] = world_city_iso3.index(each)

    if order[0]:
        tweets['lat'][ind] = lat[order[0]]
        tweets['lng'][ind] = lng[order[0]]
        continue
    if order[1]:
        tweets['lat'][ind] = lat[order[1]]
        tweets['lng'][ind] = lng[order[1]]
        continue
    if order[2]:
        try:
            tweets['lat'][ind] = code_lat[codes.index(world_city_iso2[order[2]].upper())]
            tweets['lng'][ind] = code_lng[codes.index(world_city_iso2[order[2]].upper())]
        except:
            pass
        continue
    if order[3]:
        tweets['lat'][ind] = code_lat[codes.index(world_city_iso2[order[3]].upper())]
        tweets['lng'][ind] = code_lng[codes.index(world_city_iso2[order[3]].upper())]
        continue
    if order[4]:
        tweets['lat'][ind] = code_lat[codes.index(world_city_iso2[order[4]].upper())]
        tweets['lng'][ind] = code_lng[codes.index(world_city_iso2[order[4]].upper())]
        continue

# Null values of location in tweets
all_tweets = len(tweets)
bad_tweets_without_location = tweets['user_location'].isnull().sum()
tweets_unrecovered_location = tweets['lat'].isnull().sum()

print(all_tweets, bad_tweets_without_location, tweets_unrecovered_location)
print('\nPercentage of recovering Tweet Locations using extrenal datasets...')
print((all_tweets-(tweets_unrecovered_location))/(all_tweets-bad_tweets_without_location))

map_df = tweets[['lat','lng','user_location','date']].dropna()
map_df.head()
dates = map_df['date'].str.split(' ').str.get(0).unique().tolist()
print('Number of Days in dataset:', len(dates))

daily_tweets = folium.Map(tiles='cartodbpositron', min_zoom=2) 

# Ensure you're handing it floats
map_df['lat'] = map_df['lat'].astype(float)
map_df['lng'] = map_df['lng'].astype(float)
map_df['date'] = map_df['date'].str.split(' ').str.get(0)


# List comprehension to make out list of lists
heat_data = [[[row['lat'],row['lng']] for index, row in map_df[map_df['date'] == i].iterrows()] for i in dates]

# Plot it on the map
hm = HeatMapWithTime(data=heat_data, name=None, radius=7, min_opacity=0, max_opacity=0.8, 
                     scale_radius=False, gradient=None, use_local_extrema=False, auto_play=False, 
                     display_index=True, index_steps=1, min_speed=0.1, max_speed=10, speed_step=0.1, 
                     position='bottomleft', overlay=True, control=True, show=True)
hm.add_to(daily_tweets)
# Display the map
daily_tweets.save('daily_tweets.html')
daily_tweets
def geojson_features(map_df):
    features = []
    for _, row in map_df.iterrows():
        feature = {
            'type': 'Feature',
            'geometry': {
                'type':'Point', 
                'coordinates':[row['lng'],row['lat']]
            },
            'properties': {
                'time': row['date'],
                'style': {'color' : 'red'},
                'icon': 'circle',
                'iconstyle':{
                    'fillColor': 'red',
                    'fillOpacity': 0.5,
                    'stroke': 'true',
                    'radius': 3
                }
            }
        }
        features.append(feature)
    return features


map_df = tweets[['lat','lng','user_location','date']].dropna()
timely_tweets = folium.Map(tiles='cartodbpositron', min_zoom=2) 

# Ensure you're handing it floats
map_df['lat'] = map_df['lat'].astype(float)
map_df['lng'] = map_df['lng'].astype(float)
map_df['date'] = map_df['date']


# List comprehension to make out list of lists
heat_data = [[[row['lat'],row['lng']] for index, row in map_df[map_df['date'] == i].iterrows()] for i in dates]

# Plot it on the map
hm = TimestampedGeoJson(geojson_features(map_df), transition_time=200, loop=True, auto_play=False, add_last_point=True, 
                   period='P1D', min_speed=0.1, max_speed=10, loop_button=False, date_options='YYYY-MM-DD HH:mm:ss', 
                   time_slider_drag_update=False, duration=None)
hm.add_to(timely_tweets)
# Display the map
timely_tweets

!apt install git
!git clone https://github.com/AkhilRam7/Covid19Tweets.git
%cd Covid19Tweets
!pip install flask-ngrok

from flask_ngrok import run_with_ngrok
from flask import Flask, render_template
app = Flask(__name__)
run_with_ngrok(app)   #starts ngrok when the app is run
@app.route('/')
def index():
    return render_template('daily_tweets.html')
app.run()
from IPython.display import IFrame

#Add NGROK SERVING address below in src

IFrame(src='http://705d513a190c.ngrok.io/', width=700, height=600)