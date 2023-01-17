import requests #library to handle request

from bs4 import BeautifulSoup

import pandas as pd  # library to process data as dataframes

import numpy as np # library to handle data in a vectorized manner 

from geopy.geocoders import Nominatim     # convert an address into latitude and longitude values  

import folium      # map rendering library

from pandas.io.json import json_normalize  # tranform JSON file into a pandas dataframe

from sklearn.cluster import KMeans   # import k-means from clustering stage



# Matplotlib and associated plotting modules

import matplotlib.cm as cm

import matplotlib.colors as colors
page = requests.get('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_T')

soup = BeautifulSoup(page.content, 'html.parser')
table = soup.find('table', attrs={'class':'wikitable sortable'})

df = pd.read_html(str(table))[0]

df.head()
df = df[df.Borough != 'Not assigned']  # Dropping the rows where Borough is 'Not assigned'

df = df[df.Latitude != 'Not assigned']

df = df[df.Longitude != 'Not assigned']

df = df[df.Neighborhood != 'Not assigned']

df = df.groupby(['Postal Code','Borough'], sort=False).agg(', '.join)  # Combining the neighborhoods with same Postal Code

df.reset_index(inplace=True)   





# Replacing the neighborhoods which are 'Not assigned' with names of Borough

df['Neighborhood'] = np.where(df['Neighborhood'] == 'Not assigned',df['Borough'], df['Neighborhood'])

df.head()
df.shape
df['Latitude'] = pd.to_numeric(df['Latitude'])  # the latitude value is passed into numeric
df['Longitude'] = pd.to_numeric(df['Longitude'])   # the longitude value is passed into numeric
# Get the latitude and longitude coordinates of Calgary

address = "Calgary, Alberta"



geolocator = Nominatim(user_agent="calgary_explorer")

location = geolocator.geocode(address)

latitude = location.latitude

longitude = location.longitude

print('The geograpical coordinate of Calgary city are {}, {}.'.format(latitude, longitude))
# create map of Calgary using latitude and longitude coordinates

map_calgary = folium.Map(location=[latitude, longitude], zoom_start=10)



for lat, lng, borough, neighborhood in zip(

        df['Latitude'], 

        df['Longitude'], 

        df['Borough'], 

        df['Neighborhood']):

    label = '{}, {}'.format(neighborhood, borough)

    label = folium.Popup(label, parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius=5,

        popup=label,

        color='blue',

        fill=True,

        fill_color='#3186cc',

        fill_opacity=0.7,

        parse_html=False).add_to(map_calgary)  



map_calgary

df_calgary = df[df['Borough'].str.contains("Calgary")].reset_index(drop=True)

df_calgary.head()
CLIENT_ID = 'K33SGF45U0LCTPWW2P2DTBSMZPWVSZU4YIS5KHC4FQISDKH4' # your Foursquare ID

CLIENT_SECRET = 'SVEKAXAOHPB1EOM3AP32F1TLL3AEDZLN0TJRW2BET51ELN4L' # your Foursquare Secret

VERSION = '20180605' # Foursquare API version



print('Your credentails:')

print('CLIENT_ID: ' + CLIENT_ID)

print('CLIENT_SECRET:' + CLIENT_SECRET)
neighborhood_name = df_calgary.loc[0, 'Neighborhood']

print('The first neighborhood is ', neighborhood_name)
neighborhood_latitude = df_calgary.loc[0, 'Latitude'] # neighborhood latitude value

neighborhood_longitude = df_calgary.loc[0, 'Longitude'] # neighborhood longitude value



print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 

                                                               neighborhood_latitude, 

                                                               neighborhood_longitude))
LIMIT = 100 # limit of number of venues returned by Foursquare API

radius = 1500 # define radius

url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(

    CLIENT_ID, 

    CLIENT_SECRET, 

    VERSION, 

    neighborhood_latitude, 

    neighborhood_longitude, 

    radius, 

    LIMIT)



# get the result to a json file

results = requests.get(url).json()
def get_category_type(row):

    try:

        categories_list = row['categories']

    except:

        categories_list = row['venue.categories']

        

    if len(categories_list) == 0:

        return None

    else:

        return categories_list[0]['name']
venues = results['response']['groups'][0]['items']

nearby_venues = json_normalize(venues) # flatten JSON



# filter columns

filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']

nearby_venues =nearby_venues.loc[:, filtered_columns]



# filter the category for each row

nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)



# clean columns

nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]



nearby_venues
print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))
def getNearbyVenues(names, latitudes, longitudes, radius=1500):

    venues_list=[]

    

    for name, lat, lng in zip(names, latitudes, longitudes):

        

            

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
calgary_venues = getNearbyVenues(names=df_calgary['Neighborhood'],

                                   latitudes=df_calgary['Latitude'],

                                   longitudes=df_calgary['Longitude']

                                  )
print(calgary_venues.shape)

calgary_venues.head()
calgary_venues.groupby('Neighborhood').count()
print('There are {} uniques categories.'.format(len(calgary_venues['Venue Category'].unique())))
# one hot encoding

calgary_onehot = pd.get_dummies(calgary_venues[['Venue Category']], prefix="", prefix_sep="")



# add neighborhood column back to dataframe

calgary_onehot['Neighborhood'] = calgary_venues['Neighborhood'] 



# move neighborhood column to the first column

fixed_columns = [calgary_onehot.columns[-1]] + list(calgary_onehot.columns[:-1])

calgary_onehot = calgary_onehot[fixed_columns]



calgary_onehot.head()



calgary_grouped = calgary_onehot.groupby('Neighborhood').mean().reset_index()

calgary_grouped.head()
calgary_grouped.shape
def return_most_common_venues(row, num_top_venues):

    row_categories = row.iloc[1:]

    row_categories_sorted = row_categories.sort_values(ascending=False)

    return row_categories_sorted.index.values[0:num_top_venues]



num_top_venues = 10



indicators = ['st', 'nd', 'rd']



# create columns according to number of top venues

columns = ['Neighborhood']

for ind in np.arange(num_top_venues):

    try:

        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))

    except:

        columns.append('{}th Most Common Venue'.format(ind+1))



# create a new dataframe

neighborhoods_venues_sorted = pd.DataFrame(columns=columns)

neighborhoods_venues_sorted['Neighborhood'] = calgary_grouped['Neighborhood']



for ind in np.arange(calgary_grouped.shape[0]):

    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(calgary_grouped.iloc[ind, :], num_top_venues)



neighborhoods_venues_sorted.head()
import matplotlib.pyplot as plt



calgary_grouped_clustering = calgary_grouped.drop('Neighborhood', 1)

cost =[] 

for i in range(1, 11): 

    KM = KMeans(n_clusters = i, max_iter = 500) 

    KM.fit(calgary_grouped_clustering) 

      

    # calculates squared error 

    # for the clustered points 

    cost.append(KM.inertia_)      

  

# plot the cost against K values 

plt.plot(range(1, 11), cost, color ='g', linewidth ='3') 

plt.xlabel("Value of K") 

plt.ylabel("Sqaured Error (Cost)") 

plt.show() # clear the plot 

  

# the point of the elbow is the  

# most optimal value for choosing k
# set number of clusters

kclusters = 4



# run k-means clustering

kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(calgary_grouped_clustering)



# check cluster labels generated for each row in the dataframe

kmeans.labels_[0:10] 
# add clustering labels

neighborhoods_venues_sorted.insert(0,'Cluster Labels', kmeans.labels_)



calgary_merged = df_calgary



# merge calgary_grouped with calgary_data to add latitude/longitude for each neighborhood

calgary_merged = calgary_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')



 # check the last columns!
calgary_merged.head()
# create map

map_clusters = folium.Map(location=[latitude, longitude], zoom_start=10)



# set color scheme for the clusters

x = np.arange(kclusters)

ys = [i + x + (i*x)**2 for i in range(kclusters)]

colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))

rainbow = [colors.rgb2hex(i) for i in colors_array]



# add markers to the map

markers_colors = []

for lat, lon, poi, cluster in zip(

        calgary_merged['Latitude'], 

        calgary_merged['Longitude'], 

        calgary_merged['Neighborhood'], 

        calgary_merged['Cluster Labels']):

    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)

    folium.CircleMarker(

        [lat, lon],

        radius=5,

        popup=label,

        color=rainbow[cluster-1],

        fill=True,

        fill_color=rainbow[cluster-1],

        fill_opacity=0.7).add_to(map_clusters)

       

map_clusters

calgary_merged.loc[calgary_merged['Cluster Labels'] == 0, calgary_merged.columns[[2] + list(range(5, calgary_merged.shape[1]))]]
calgary_merged.loc[calgary_merged['Cluster Labels'] == 1, calgary_merged.columns[[2] + list(range(5, calgary_merged.shape[1]))]]
calgary_merged.loc[calgary_merged['Cluster Labels'] == 2, calgary_merged.columns[[2] + list(range(5, calgary_merged.shape[1]))]]
calgary_merged.loc[calgary_merged['Cluster Labels'] == 3, calgary_merged.columns[[2] + list(range(5, calgary_merged.shape[1]))]]