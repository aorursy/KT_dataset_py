import requests # library to handle requests
import pandas as pd # library for data analsysis
import numpy as np # library to handle data in a vectorized manner
import random # library for random number generation

#uncomment next line if need to install latest version of geopy
#!conda install -c conda-forge geopy --yes 
from geopy.geocoders import Nominatim # module to convert an address into latitude and longitude values

# libraries for displaying images
from IPython.display import Image 
from IPython.core.display import HTML 
    
# tranforming json file into a pandas dataframe library
from pandas import json_normalize

#uncomment next line if need to install latest version of folium
#!conda install -c conda-forge folium=0.5.0 --yes
import folium # plotting library

print('Folium installed')
print('Libraries imported.')
CLIENT_ID = 'your-client-ID' # enter your Foursquare ID here!
CLIENT_SECRET = 'your-client-secret' # enter your Foursquare Secret here!

VERSION = '20180604' # what version of Foursquare you want to use
LIMIT = 20 # max limit is 50 
print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)
CLIENT_ID = '0IBRWROBN4BHTVCXD5J43BB3JQARQ4V1DRETPONPPMVY205B'
CLIENT_SECRET = 'KTMQCKPDDQFYN41SU0R1OG3DTW4DHHZARAAKNLSJF0DXJSYH' 
# Grand Central Terminal Address
address = '89 E 42nd St, New York, NY 10017'

geolocator = Nominatim(user_agent="foursquare_agent")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print("The latitude and longitude coordinates are:")
print(latitude, longitude)
search_query = 'Pizza'
radius = 500 #Radius of search in meters

url = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&query={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION, search_query, radius, LIMIT)
results = requests.get(url).json()
# assign relevant part of JSON to venues
venues = results['response']['venues']

# tranform venues into a pandas dataframe
dataframe = json_normalize(venues)
dataframe.head()
# keep only columns that include venue name, and anything that is associated with location
filtered_columns = ['name', 'categories'] + [col for col in dataframe.columns if col.startswith('location.')] + ['id']
dataframe_filtered = dataframe.loc[:, filtered_columns]

# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']

# filter the category for each row
dataframe_filtered['categories'] = dataframe_filtered.apply(get_category_type, axis=1)

# clean column names by keeping only last term
dataframe_filtered.columns = [column.split('.')[-1] for column in dataframe_filtered.columns]

dataframe_filtered
venue_id = '4c7d96bbd65437043defc0a2' # ID of closest pizza joint
url = 'https://api.foursquare.com/v2/venues/{}?client_id={}&client_secret={}&v={}'.format(venue_id, CLIENT_ID, CLIENT_SECRET, VERSION)

result = requests.get(url).json()
result['response']['venue']
print(result['response']['venue']['name'])
try:
    print(result['response']['venue']['rating'])
except:
    print('This venue has not been rated yet.')
venue_id = '4a8c31aef964a520410d20e3' # ID of second closest
url = 'https://api.foursquare.com/v2/venues/{}?client_id={}&client_secret={}&v={}'.format(venue_id, CLIENT_ID, CLIENT_SECRET, VERSION)

result = requests.get(url).json()
print(result['response']['venue']['name'])
try:
    print(result['response']['venue']['rating'])
except:
    print('This venue has not been rated yet.')
venue_id = '4d012c08ba1da1cd3cb68c28' # ID of third closest
url = 'https://api.foursquare.com/v2/venues/{}?client_id={}&client_secret={}&v={}'.format(venue_id, CLIENT_ID, CLIENT_SECRET, VERSION)

result = requests.get(url).json()
print(result['response']['venue']['name'])
try:
    print(result['response']['venue']['rating'])
except:
    print('This venue has not been rated yet.')
venues_map = folium.Map(location=[latitude, longitude], zoom_start=16) # generate map centred around the Grand Central Terminal

# add a red circle marker to represent Grand Central Terminal
folium.features.CircleMarker(
    [latitude, longitude],
    radius=10,
    color='red',
    popup='Grand Central Terminal',
    fill = True,
    fill_color = 'red',
    fill_opacity = 0.6
).add_to(venues_map)

# add the pizza joints as blue circle markers
for lat, lng, label in zip(dataframe_filtered.lat, dataframe_filtered.lng, dataframe_filtered.categories):
    folium.features.CircleMarker(
        [lat, lng],
        radius=5,
        color='blue',
        popup=label,
        fill = True,
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(venues_map)

# display map
venues_map
