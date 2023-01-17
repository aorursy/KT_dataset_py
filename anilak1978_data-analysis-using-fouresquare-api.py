# import neccessary libraries

!conda install -c conda-forge geopy --yes 

from geopy.geocoders import Nominatim # module to convert an address into latitude and longitude values

import requests # library to handle requests

import pandas as pd # library for data analsysis

import numpy as np # library to handle data in a vectorized manner

import random # library for random number generation



# libraries for displaying images

from IPython.display import Image 

from IPython.core.display import HTML 

    

# tranforming json file into a pandas dataframe library

from pandas.io.json import json_normalize



!conda install -c conda-forge folium=0.5.0 --yes

import folium # plotting library



print('Folium installed')

print('Libraries imported.')
# Define Foursquare Credentials and Version



CLIENT_ID = 'SLDDVV4ZMUHC2TGZ113CKT1XF3MJZRBZNRL4QSYVDRGLW3XJ' # your Foursquare ID

CLIENT_SECRET = 'VDOU35CORKHKU3PPHIMZDFZDLYSHH4HRZAQGVZGYIENPRHNA' # your Foursquare Secret

VERSION = '20180604'

LIMIT = 30

print('Your credentails:')

print('CLIENT_ID: ' + CLIENT_ID)

print('CLIENT_SECRET:' + CLIENT_SECRET)
address = '365 5th Ave, New York, NY 10016'



geolocator = Nominatim()

location = geolocator.geocode(address)

latitude = location.latitude

longitude = location.longitude

print(latitude, longitude)
search_query = 'Japanese'

radius = 500

print(search_query + ' .... OK!')
# create the api url

url = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&query={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION, search_query, radius, LIMIT)

url
# get the data result in json

results = requests.get(url).json()

results
# assign relevant part of JSON to venues

venues = results['response']['venues']



# tranform venues into a dataframe

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
dataframe_filtered.info()
dataframe_filtered.describe()
dataframe_filtered.name
venues_map = folium.Map(location=[latitude, longitude], zoom_start=13) # generate map centred around the CUNY



# add a red circle marker to represent the CUNY

folium.features.CircleMarker(

    [latitude, longitude],

    radius=10,

    color='red',

    popup='CUNY',

    fill = True,

    fill_color = 'red',

    fill_opacity = 0.6

).add_to(venues_map)



# add the Japanese restaurants as blue circle markers

for lat, lng in zip(dataframe_filtered.lat, dataframe_filtered.lng):

    folium.features.CircleMarker(

        [lat, lng],

        radius=5,

        color='blue',

        fill = True,

        fill_color='blue',

        fill_opacity=0.6

    ).add_to(venues_map)



# display map

venues_map
venue_id = '4e4e4ad5bd4101d0d7a7002d' # ID of Ten Sushi Restaurant

url = 'https://api.foursquare.com/v2/venues/{}?client_id={}&client_secret={}&v={}'.format(venue_id, CLIENT_ID, CLIENT_SECRET, VERSION)

url
result = requests.get(url).json()

print(result['response']['venue'].keys())

result['response']['venue']
try:

    print(result['response']['venue']['rating'])

except:

    print('This venue has not been rated yet.')
venue_id = '4c44df40429a0f47c660491e' # ID of Sariku Japanese Restaurant

url = 'https://api.foursquare.com/v2/venues/{}?client_id={}&client_secret={}&v={}'.format(venue_id, CLIENT_ID, CLIENT_SECRET, VERSION)



result = requests.get(url).json()

try:

    print(result['response']['venue']['rating'])

except:

    print('This venue has not been rated yet.')
# get the number of tips

result['response']['venue']['tips']['count']
## Sariku Tips - create the url

limit = 15 # set limit to be greater than or equal to the total number of tips

url = 'https://api.foursquare.com/v2/venues/{}/tips?client_id={}&client_secret={}&v={}&limit={}'.format(venue_id, CLIENT_ID, CLIENT_SECRET, VERSION, limit)



# get the url in json

results = requests.get(url).json()

results
tips = results['response']['tips']['items']



tip = results['response']['tips']['items'][0]

tip.keys()
pd.set_option('display.max_colwidth', -1)



tips_df = json_normalize(tips) # json normalize tips



# columns to keep

filtered_columns = ['text', 'agreeCount', 'disagreeCount', 'id', 'user.firstName', 'user.lastName', 'user.gender', 'user.id']

tips_filtered = tips_df.loc[:, filtered_columns]



# display tips

tips_filtered
user_id = '8785316' # user ID of Andrew Buck



# create the url

url = 'https://api.foursquare.com/v2/users/{}?client_id={}&client_secret={}&v={}'.format(user_id, CLIENT_ID, CLIENT_SECRET, VERSION) # define URL



# send GET request

results = requests.get(url).json()

user_data = results['response']['user']



# display features associated with user

user_data.keys()
# full information on Andrew Buck

print('First Name: ' + user_data['firstName'])

print('Last Name: ' + user_data['lastName'])

print('Home City: ' + user_data['homeCity'])
# figuring out how many tips have been submitted.

user_data['tips']
# define tips URL

url = 'https://api.foursquare.com/v2/users/{}/tips?client_id={}&client_secret={}&v={}&limit={}'.format(user_id, CLIENT_ID, CLIENT_SECRET, VERSION, limit)



# send GET request and get user's tips

results = requests.get(url).json()

tips = results['response']['tips']['items']



# format column width

pd.set_option('display.max_colwidth', -1)



tips_df = json_normalize(tips)



# filter columns

filtered_columns = ['text', 'agreeCount', 'disagreeCount', 'id']

tips_filtered = tips_df.loc[:, filtered_columns]



# display user's tips

tips_filtered
user_friends = json_normalize(user_data['friends']['groups'][0]['items'])

user_friends
# getting users profile image

user_data
# we can pull the image 

Image(url='https://fastly.4sqi.net/img/user/300x300/KRKQK1XSOZZSATJB.jpg')
# umi restaurant latitude and longitude are as below

latitude=40.744587

longitude=-73.981918
# define the url

url = 'https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION, radius, LIMIT)

url



# get the url in json

results = requests.get(url).json()

'There are {} around Umi restaurant.'.format(len(results['response']['groups'][0]['items']))
# get the relevant part of the JSON

items = results['response']['groups'][0]['items']

items[0]
# create a clean dataframe

dataframe = json_normalize(items) # flatten JSON



# filter columns

filtered_columns = ['venue.name', 'venue.categories'] + [col for col in dataframe.columns if col.startswith('venue.location.')] + ['venue.id']

dataframe_filtered = dataframe.loc[:, filtered_columns]



# filter the category for each row

dataframe_filtered['venue.categories'] = dataframe_filtered.apply(get_category_type, axis=1)



# clean columns

dataframe_filtered.columns = [col.split('.')[-1] for col in dataframe_filtered.columns]



dataframe_filtered.head(10)
# visualize the items on the map

venues_map = folium.Map(location=[latitude, longitude], zoom_start=15) # generate map centred around Umi Restaurant





# add Umi Restaurant as a red circle mark

folium.features.CircleMarker(

    [latitude, longitude],

    radius=10,

    popup='Umi Restaurant',

    fill=True,

    color='red',

    fill_color='red',

    fill_opacity=0.6

    ).add_to(venues_map)





# add popular spots to the map as blue circle markers

for lat, lng, label in zip(dataframe_filtered.lat, dataframe_filtered.lng, dataframe_filtered.categories):

    folium.features.CircleMarker(

        [lat, lng],

        radius=5,

        popup=label,

        fill=True,

        color='blue',

        fill_color='blue',

        fill_opacity=0.6

        ).add_to(venues_map)



# display map

venues_map
# define URL

url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION)



# send GET request and get trending venues

results = requests.get(url).json()

results
# check if there are any venues trending at this time.

if len(results['response']['venues']) == 0:

    trending_venues_df = 'No trending venues are available at the moment!'

    

else:

    trending_venues = results['response']['venues']

    trending_venues_df = json_normalize(trending_venues)



    # filter columns

    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']

    trending_venues_df = trending_venues_df.loc[:, columns_filtered]



    # filter the category for each row

    trending_venues_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)
# display trending venues

trending_venues_df
# lets visualize 

if len(results['response']['venues']) == 0:

    venues_map = 'Cannot generate visual as no trending venues are available at the moment!'



else:

    venues_map = folium.Map(location=[latitude, longitude], zoom_start=15) # generate map centred around Ecco





    # add Ecco as a red circle mark

    folium.features.CircleMarker(

        [latitude, longitude],

        radius=10,

        popup='Umi',

        fill=True,

        color='red',

        fill_color='red',

        fill_opacity=0.6

    ).add_to(venues_map)





    # add the trending venues as blue circle markers

    for lat, lng, label in zip(trending_venues_df['location.lat'], trending_venues_df['location.lng'], trending_venues_df['name']):

        folium.features.CircleMarker(

            [lat, lng],

            radius=5,

            poup=label,

            fill=True,

            color='blue',

            fill_color='blue',

            fill_opacity=0.6

        ).add_to(venues_map)

        

# display map

venues_map