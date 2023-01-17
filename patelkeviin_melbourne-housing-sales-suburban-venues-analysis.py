# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import libraries neccessary





import pandas as pd

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)



import numpy as np



#!pip install matplotlib

from matplotlib import pyplot as plt  # plotting library

%matplotlib inline



import requests

#!pip install folium

import folium

#!pip install geopy

from geopy import Nominatim



print('libraries imported!')
# input variables - Housing budget from the client/user



BUDGET = 0.3    # dummy value

BUDGET = input("Please enter your housing budget (in Millions AU$): ")

BUDGET = float(BUDGET)
# cleaning Melbourne Housing Market dataset





filtered_columns = ['Suburb', 'Address', 'Rooms', 'Price', 'Date', 'Distance', 'Postcode', 'Bedroom2', 'Landsize', 'YearBuilt', 'CouncilArea', 'Lattitude', 'Longtitude']

housing_data = pd.read_csv('/kaggle/input/melbourne-housing-market/Melbourne_housing_FULL.csv', usecols=filtered_columns, parse_dates=True)



# renaming columns

housing_data.rename(columns={'Bedroom2':'Bedrooms', 'Longtitude':'Longitude', 'Price':'Price(in$M)'}, inplace=True)



#changing column types

housing_data.drop([29483], inplace=True)  # Postcode is null for this entry

housing_data.dropna(subset=['Lattitude', 'Longitude'], inplace=True) # Geolocations not available

housing_data = housing_data.astype({'Date': 'datetime64', 'Postcode':'int64'})



# dropping rows where Price is null

housing_data.drop(housing_data[housing_data['Price(in$M)'].isnull()].index, inplace=True)

housing_data = housing_data.reset_index(drop=True)



# changing Price values per 1 Million dollors

housing_data['Price(in$M)'] = housing_data['Price(in$M)'].apply(lambda price: price / 1000000)



housing_data.head()
# analysing average housing prices for each suburb in Melbourne





housing_price_average = housing_data.groupby('Suburb')['Price(in$M)'].mean()



# top 10 most priced suburbs in melbourne

top_housing_price_average = housing_price_average.sort_values(ascending=False).iloc[0:10]



# plotting

fig, ax = plt.subplots(figsize=(3, 3), dpi= 80)

ax.bar(top_housing_price_average.index, top_housing_price_average, label='Price(in$M)')

ax.tick_params('x', rotation=90)

ax.set_xlabel('Suburbs')

ax.set_ylabel('Price in Millions')

plt.show()
# analysing average housing prices for each suburb in Melbourne





top10_costly_suburbs = housing_data[housing_data.Suburb.isin(top_housing_price_average.index)]



top10_costly_suburbs.boxplot(column='Price(in$M)', by='Suburb', figsize=(10, 5))

plt.show()
# analysing average housing prices for each street in each suburb in Melbourne





# extracting street address from address

housing_data['StreetAddress'] = housing_data.Address.str.split(' ').apply(lambda address_list: ' '.join(address_list[1:]))



housing_price_average_street = housing_data.groupby(['Suburb', 'StreetAddress'])['Price(in$M)'].mean()



top10_costliest_suburbs = [

    'Kooyong',

    'Brighton',

    'Canterbury',

    'Malvern',

    'Kew',

    'Middle Park',

    'Balwyn',

    'Albert Park'

]



group = housing_price_average_street.groupby('Suburb')

for suburb in top10_costliest_suburbs:

    g = group.get_group(suburb)

    top5 = g.sort_values(ascending=False).iloc[0:5]

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.bar(top5.index.get_level_values(1), top5, label=suburb)

    ax.tick_params('x', rotation=90)

    ax.set_xlabel('Streets in ' + suburb)

    ax.set_ylabel('Price in Million')

    plt.show()



#for suburb, group in housing_price_average_street.groupby('Suburb'):

#    top5 = group.sort_values(ascending=False).iloc[0:5]

#    fig, ax = plt.subplots(figsize=(5, 4))

#    ax.bar(top5.index.get_level_values(1), top5, label=suburb)

#    ax.tick_params('x', rotation=90)

#    ax.set_xlabel('Streets in ' + suburb)

#    ax.set_ylabel('Price in Million')

#    plt.show()
# encode physical locations to its corresponding geolocations !!Not Working right now!!





#def geocoder(row):

#    locator = Nominatim(user_agent='myGeocoder')

#    location = locator.geocode(row.name[1] + ', ' + row.name[0] + ", Australia")

#    return (location.latitude, location.longitude)

#  !!Not Working right now!!





#housing_price_average_street = housing_price_average_street.to_frame() 

# filtering streets based on client budget

#recommended_streets = housing_price_average_street[housing_price_average_street['Price(in$M)'] <= BUDGET]



#recommended_streets['Latitude'], recommended_streets['Longitude'] = recommended_streets.apply(geocoder, axis=1)



#recommended_streets.head()
# adding latitudes and longitudes for each of these streets





print('Client budget: AU$ {}M'.format(BUDGET))

grouping = {'Price(in$M)': 'mean', 'Lattitude': 'first', 'Longitude': 'first'}

recommended_streets = housing_data.groupby(['Suburb', 'StreetAddress']).agg(grouping)

recommended_streets = recommended_streets[recommended_streets['Price(in$M)'] <= BUDGET]

recommended_streets.head()
print('{} streets were selected based on client budget.'.format(recommended_streets.shape[0]))
# plotting recommended locations on the map of Melbourne with current housing market prices





# Melbourne coordinates

latitude = -37.814

longitude = 144.96332

# create map of Melbourne using latitude and longitude values

map_melbourne = folium.Map(location=[latitude, longitude], zoom_start=10)



# add markers to map

for lat, lng, address in zip(recommended_streets['Lattitude'], recommended_streets['Longitude'], recommended_streets.index):

    address = address[1] + ", " + address[0]

    label = folium.Popup(address, parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius=5,

        popup=label,

        color='blue',

        fill=True,

        fill_color='#3186cc',

        fill_opacity=0.7,

        parse_html=False).add_to(map_melbourne)  

    

map_melbourne
# define Foursquare credentials and API version





CLIENT_ID = 'R5MHPNIHCONOACDO4Q1WWWODRIBTX54TWD05FI0EZL4P4PA3' # your Foursquare ID

CLIENT_SECRET = 'EKHCYAIU4OBDZAWGZITQHPIJ1DTSWCCIKAEZT1NNICQSQSPW' # your Foursquare Secret

VERSION = '20180605' # Foursquare API version



print('Your credentails:')

print('CLIENT_ID: ' + CLIENT_ID)

print('CLIENT_SECRET:' + CLIENT_SECRET)
# obtaining nearby venues to each street selected based on client budget

# this function won't work in Kaggle as accessing web resources are not relaible in Kaggle notebooks. Hence, a new dataset has

# been upladed by me using this same function





def getNearbyVenues(street_names, suburbs, latitudes, longitudes, radius=500):

    LIMIT = 100

    venues_list=[]

    

    print('Street Name, Suburb:')

    for street_name, suburb, lat, lng in zip(street_names, suburbs, latitudes, longitudes):

        print(street_name + ', ' + suburb)

            

        # create the API request URL

        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(

            CLIENT_ID, 

            CLIENT_SECRET, 

            VERSION, 

            lat, 

            lng, 

            radius, 

            LIMIT)

            

        # make the GET request

        results = requests.get(url).json()["response"]['groups'][0]['items']

        

        # return only relevant information for each nearby venue

        venues_list.append([(

            street_name,

            suburb,

            lat, 

            lng, 

            v['venue']['name'], 

            v['venue']['location']['lat'], 

            v['venue']['location']['lng'],  

            v['venue']['categories'][0]['name']) for v in results])



    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])

    nearby_venues.columns = ['Street', 

                             'Suburb', 

                           'Latitude', 

                          'Longitude', 

                              'Venue', 

                     'Venue Latitude', 

                    'Venue Longitude', 

                     'Venue Category']

    

    return (nearby_venues)
# obtaining nearby venues to each street selected based on client budget





# melbourne_venues = getNearbyVenues(street_names=recommended_streets.index.get_level_values(1),

#                                   suburbs=recommended_streets.index.get_level_values(0),

#                                   latitudes=recommended_streets['Lattitude'],

#                                   longitudes=recommended_streets['Longitude']

#                                  )



melbourne_venues = pd.read_csv('/kaggle/input/melbourne-venues/Melbourne_venues.csv')

print(melbourne_venues.shape)

melbourne_venues.head()
# determining unique venues (categories) for each street in a suburb





for gname, group in melbourne_venues.groupby(['Suburb', 'Street']):

    print(gname[1] + ', ' + gname[0])

    print(group['Venue Category'].unique())

    print()
# determining unique venues (categories) overall in Melbourne





melbourne_venues['Venue Category'].unique()
# basic neighborhood amenities that drives up one's choice of residence





basic_amenities = [

    'Station',

    'Stop',

    'Restaurant',

    'Café',

    'Pharmacy',

    'Market',

    'Supermarket',

    'Shop',

    'University',

    'School',

    'Gym',

    'Theater',

    'Laundromat',

    'Lake',

    'Park',

    'Playground', 

]
# analysing each street (in a suburb) against the basic amenities in its proximity





# filtering venues based on wheter they fall into basic amenity or not

def is_amenity(row):

    for amenity in basic_amenities:

        if amenity in row:

            return True

        

    return False





# filtering venues based on wheter they fall into basic amenity or not

amenities = melbourne_venues[melbourne_venues['Venue Category'].apply(is_amenity)]



# Analyze each street

# one hot encoding

amenities = pd.get_dummies(amenities[['Venue Category']], prefix="", prefix_sep="")



# add Street and Suburb columns back to dataframe

amenities['Street'], amenities['Suburb'] = melbourne_venues['Street'], melbourne_venues['Suburb']



# adjust columns

fixed_columns = [amenities.columns[-2]] + [amenities.columns[-1]] + list(amenities.columns[:-2])

amenities = amenities[fixed_columns]



amenities.head()
# Next, let's group rows by street and suburb, and by taking the sum of the frequency of occurrence of each category





amenities_frequency = amenities.groupby(['Suburb', 'Street']).sum()

amenities_frequency.head()
# recommend top 15 streets with the most total number of nearby amenities





recommended_streets = amenities.groupby(['Suburb', 'Street'])[['Afghan Restaurant']].count().sort_values('Afghan Restaurant', ascending=False)

recommended_streets.columns = ['Amenities Count']

recommended_streets = recommended_streets[0:15]



# adding location coordinates data

left = recommended_streets.reset_index()

right = melbourne_venues[['Suburb', 'Street', 'Latitude', 'Longitude']].drop_duplicates(subset=['Suburb', 'Street'])

recommended_streets = pd.merge(left=left, right=right, left_on=['Suburb', 'Street'], right_on=['Suburb', 'Street'])



recommended_streets.head(15)
# plotting recommended locations on the map of Melbourne





# Melbourne coordinates

latitude = -37.814

longitude = 144.96332

# create map of Melbourne using latitude and longitude values

map_melbourne = folium.Map(location=[latitude, longitude], zoom_start=10)



# add markers to map

for lat, lng, street, suburb in zip(recommended_streets['Latitude'], recommended_streets['Longitude'], recommended_streets['Street'], recommended_streets['Suburb']):

    address = street + ", " + suburb

    label = folium.Popup(address, parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius=5,

        popup=label,

        color='blue',

        fill=True,

        fill_color='#3186cc',

        fill_opacity=0.7,

        parse_html=False).add_to(map_melbourne)  

    

map_melbourne