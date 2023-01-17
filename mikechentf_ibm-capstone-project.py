import numpy as np 

import pandas as pd 



!pip install geocoder

import geocoder



import requests



import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="white")



import folium



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
demographics = pd.read_csv('/kaggle/input/sf-demographics-data/SF Demographics Dataset.csv')
demographics.head()
demographics.info() 
demographics = demographics.fillna(0)
demographics['White'] = demographics['White'].astype('float64')

demographics['Other/Two or More Races'] = demographics['Other/Two or More Races'].astype('float64')

demographics['% Latino (of Any Race)'] = demographics['% Latino (of Any Race)'].astype('float64')
demographics.describe()
neighborhoods = demographics['Neighborhood'].to_list()



longitude = []

latitude = []



for neighborhood in neighborhoods:

    

    # initialize the variable to None

    lat_lng_coords = None



    # loop until getting the coordinates

    while(lat_lng_coords is None):

        g = geocoder.arcgis('{}, San Francisco, California'.format(neighborhood))

        lat_lng_coords = g.latlng



    

    # Append the data to the lists

    latitude.append(lat_lng_coords[0])

    longitude.append(lat_lng_coords[1])
location = pd.DataFrame({'Neighborhood': neighborhoods, 'Latitude': latitude, 'Longitude': longitude})
location.head()
# Setting Foursquare credentials

CLIENT_ID = 'DASAS2TJ5QYKKAI2QZEPBF0XACCR5JAX0JL4OKNFPI1SYN0K' # your Foursquare ID

CLIENT_SECRET = 'OXNV1ECFX2G4ZYPKP5BDAYI1OZPA1SYVZDIMCKLDSB05OEPE' # your Foursquare Secret

VERSION = '20180605' # Foursquare API version
"""

def getNearbyVenues(names, latitudes, longitudes, radius=1600, LIMIT=300, categoryId='4d4b7105d754a06374d81259'):

    

    venues_list=[]

    for name, lat, lng in zip(names, latitudes, longitudes):

        print(name)

            

        # create the API request URL

        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}&categoryId={}'.format(

            CLIENT_ID, 

            CLIENT_SECRET, 

            VERSION, 

            lat, 

            lng, 

            radius, 

            LIMIT,

            categoryId)

            

        # make the GET request

        results = requests.get(url).json()["response"]['groups'][0]['items']

        

        # return only relevant information for each nearby venue

        venues_list.append([(

            name, 

            lat, 

            lng, 

            v['venue']['name'], 

            v['venue']['location']['lat'], 

            v['venue']['location']['lng'],  

            v['venue']['categories'][0]['name']) for v in results])



    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])

    nearby_venues.columns = ['Neighborhood', 

                  'Neighborhood Latitude', 

                  'Neighborhood Longitude', 

                  'Venue', 

                  'Venue Latitude', 

                  'Venue Longitude', 

                  'Venue Category']

    

    return(nearby_venues)



venues = getNearbyVenues(names=location['Neighborhood'],

                                   latitudes=location['Latitude'],

                                   longitudes=location['Longitude']

                                  )"""
venues = pd.read_csv('/kaggle/input/sf-venues/SF venues.csv')
venues.groupby('Neighborhood').count()
venue_count = venues.groupby(['Neighborhood', 'Venue Category'])['Venue'].count()

venue_count = venue_count.unstack()

venue_count = venue_count.fillna(0)
venue_count.head()
plt.figure(figsize=(20,10))

ax = sns.barplot(demographics['Neighborhood'], demographics['Total Population'])

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

plt.show()
plt.figure(figsize=(10,10))

sns.boxplot(demographics['Total Population'])

plt.show()
plt.figure(figsize=(20,10))

ax = sns.barplot(demographics['Neighborhood'], demographics['Median Household Income'])

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

plt.show()
plt.figure(figsize=(10,10))

sns.boxplot(demographics['Median Household Income'])

plt.show()
dem_corr = demographics.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(dem_corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(dem_corr, mask=mask, cmap=cmap, vmax=1, vmin=-1,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
import math

def get_info(venue_count, demographics):

    neighs = []

    infos = []

    dem_keys = demographics.iloc[0][['Asian', 'Black/African American', 'White', 'Native American Indian', 'Native Hawaiian/Pacific Islander', 'Other/Two or More Races', '% Latino (of Any Race)']].keys()

    for i in range(len(venue_count)):

        neigh = "<b>" + venue_count.iloc[i].name + "</b>"

        message = ""

        message += neigh

        message = message + '<br>Population: ' + str(demographics['Total Population'][i])

        message += '<br><br>Race (%):<ul> '

        for key in dem_keys:

            message = message + '<li>' + key + ': ' + str(demographics.iloc[i][key]) + '</li>' 

        message += '</ul>'

        message += '<p style="width:200px"><i>Most common restaurant:</i></p><ol>'

        top_keys = venue_count.iloc[i].sort_values(ascending=False).keys()[:5]

        top_values = venue_count.iloc[i].sort_values(ascending=False).values[:5]

        for j in range(5):

            message = message + '<li>' + top_keys[j] + ': ' + str(math.trunc(top_values[j])) + '</li>'

        message += '</ol>'

        neighs.append(neigh)

        infos.append(message)

    return neighs, infos
m = folium.Map(

    location=[37.7749, -122.4194],

    zoom_start=12  

)



restaurants = folium.map.FeatureGroup()



for neighborhood, lat, lng in zip(location['Neighborhood'], location['Latitude'], location['Longitude']):

    restaurants.add_child(

        folium.vector_layers.CircleMarker(

            [lat, lng],

            radius=5, 

            color='yellow',

            fill=True,

            fill_color='blue',

            fill_opacity=0.6

        )

    )



latitudes = list(location['Latitude'])

longitudes = list(location['Longitude'])

neighs, infos = get_info(venue_count, demographics)







for lat, lng, neigh, info in zip(latitudes, longitudes, neighs, infos):

    folium.map.Marker([lat, lng], popup=folium.map.Popup(html=info, parse_html=False, max_width='300px'), tooltip=neigh).add_to(m)    

    

m.add_child(restaurants)
m.save('map.html') 