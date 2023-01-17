# Import libraries and packages

import numpy as np

import pandas as pd



# Machine Learning and Plotting

from sklearn.cluster import KMeans

import matplotlib.cm as cm

import matplotlib.pyplot as plt



# Library to handle JSON files

import json

import requests 

from pandas.io.json import json_normalize



# Maps visualization

import folium



# Address to geographical data

from geopy.geocoders import Nominatim



# Web Scraping and Reading HTML files

from bs4 import BeautifulSoup



# Used with lists

import itertools



# To add delay between quries

import time



# Useful packages that we will use while clustering neighbourhoods

from scipy.spatial.distance import cdist

from sklearn import metrics

from sklearn.metrics import silhouette_score

import matplotlib.colors as colors



# To Ignore Warnings

import warnings

warnings.filterwarnings("ignore")



print("IMPORT STATUS:\t DONE.")
# Get HTML file of Berlin Neighbourhoods

Berlin_neighbourhoods_html = requests.get('https://en.wikipedia.org/wiki/Boroughs_and_neighborhoods_of_Berlin').text

page = BeautifulSoup(Berlin_neighbourhoods_html, 'lxml')



# Check Webpage Title

print('1. Webpage Title: \n----------------------\n{} \n \n'.format(page.title.text))



# Find the first table in the webpage

table = page.find('table')



# Create a DataFrame for Boroughs in Berlin

Berlin_boroughs = pd.DataFrame()



# Assign the retrived table to a list

data_list = pd.read_html(str(table))



# Copy retrived data to the Berlin_boroughs DataFrame

Berlin_boroughs = data_list[0]



# Rename Columns and drop the Map column

Berlin_boroughs.drop(['Map'], axis =1, inplace=True)

Berlin_boroughs.columns = ['Boroughs', 'Population', 'Area', 'Density']



print('2. DataFrame Shape:\n----------------------\n', Berlin_boroughs.shape,'\n \n')

Berlin_boroughs
# In order to search for Borough Coordinates

geolocator = Nominatim(user_agent="Berlin_Maps")



# Create empty lists to store lonitude and latitude values

lat = []

lng = []



# Add Latitude and Longitude values of each Borough to the DataFrame

for borough in Berlin_boroughs.itertuples():

    Berlin_location = geolocator.geocode('{}, Berlin'.format(Berlin_boroughs['Boroughs'][borough.Index]))

    lat.insert(borough.Index, Berlin_location.latitude)

    lng.insert(borough.Index, Berlin_location.longitude)

    

Berlin_boroughs['Latitude'] = lat

Berlin_boroughs['Longitude'] = lng



Berlin_boroughs.head()
# Get all table titles from the page

all_names = page.find_all('dt')

all_tables = page.find_all('table')



# Create a DataFrame for all Neighbourhoods

Berlin_neighbourhoods = pd.DataFrame()



# Create a DataFrame for all Neighbourhoods in Berlin

for boroughs in Berlin_boroughs.itertuples():

    index = boroughs.Index

    # Get the Borough Name, and remove the first word from the string

    Borough_name = str(all_names[index].text).split(' ')[1]



    # Create a temp list of DataFrames and add the Neighbourhood data to it

    temp_list = pd.read_html(str(all_tables[index + 2]))

    temp_list[0]['Borough'] = Borough_name

    Berlin_neighbourhoods = Berlin_neighbourhoods.append(temp_list[0], ignore_index=True)



# Check DataFrame Size 

print('DataFrame Shape:\n----------------------\n{}\n\n'.format(Berlin_neighbourhoods.shape))



# Check the first 10 Neighbourhoods

Berlin_neighbourhoods.head(10)
# First, drop the Map Column

Berlin_neighbourhoods.drop(['Map'], axis =1, inplace=True)



# Rename columns

Berlin_neighbourhoods.columns = ['Neighbourhood', 'Area', 'Population', 'Density', 'Borough']



# Remove brackets in Neighbourhood names

for name in Berlin_neighbourhoods.itertuples():

    index = int(name.Index)

    Berlin_neighbourhoods.at[index, 'Neighbourhood'] = str(Berlin_neighbourhoods.at[index, 'Neighbourhood']).split(' ')[1]

    

# Check the first 5 Neighbourhoods

Berlin_neighbourhoods.head()
# Add Latitude and Longitude to the DataFrame



# In order to search for Borough Coordinates

geolocator = Nominatim(user_agent="Berlin_Data")



# Create empty lists for lat, lng values

lat = []

lng = []





# Add Latitude and Longitude values of each Borough to the DataFrame

for neighbourhood in Berlin_neighbourhoods.itertuples():

    # Set index

    index = int(neighbourhood.Index)

    

    try:

        # Get address and save it, use Borough name as well instead of neighbourhood only

        Berlin_location = geolocator.geocode('{},{}, Berlin'.format(Berlin_neighbourhoods.at[index, 'Neighbourhood'],

                                                                    Berlin_neighbourhoods.at[index, 'Borough']))

    except: 

        print('This generally occurs due to a timeout error from geolocator side, try again.')

        

    

    # Insert new data

    lat.insert(index, Berlin_location.latitude)

    lng.insert(index, Berlin_location.longitude)



# Add New columns with extracted values

Berlin_neighbourhoods['Latitude'] = lat

Berlin_neighbourhoods['Longitude'] = lng



# Examine the data

Berlin_neighbourhoods.head()
# @hidden_cell



CLIENT_ID = 'HIE5ISYEZQEQPNSVGQGRWCJMXA43OC3MBRHICDU01GF1P0EA' # your Foursquare ID

CLIENT_SECRET = 'DK3E0ME2RXUUXAOX54VSSULBVYBUJWUD4BRVAQIJMV2ZIG54' # your Foursquare Secret

VERSION = '20190701' # Foursquare API version
# Returns a DataFrame with Venue details

def getNearbyVenues(names, latitudes, longitudes, radius, LIMIT):

    

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

    nearby_venues.columns = ['Neighbourhood', 

                  'Neighbourhood Latitude', 

                  'Neighbourhood Longitude', 

                  'Venue', 

                  'Venue Latitude', 

                  'Venue Longitude', 

                  'Venue Category']

    

    return(nearby_venues)
# Returns a DataFrame with top venues based on a threshold

def return_top_venues(row, maximum_venues):

    

    # Select all except neighbourhood column

    row_categories = row.iloc[1:]

    row_categories_sorted = row_categories.sort_values(ascending=False)

    

    return row_categories_sorted.index.values[0: maximum_venues]
# Try to fetch Berlin Venues data

try:

    Berlin_venues = getNearbyVenues(Berlin_neighbourhoods['Neighbourhood'],

                                        Berlin_neighbourhoods['Latitude'],

                                        Berlin_neighbourhoods['Longitude'],

                                        radius=2000,

                                        LIMIT=100)

except:

    print('Error fetching data, could be caused by exceeding Forsquare maximum calls/ day.')



# View the DataFrame

Berlin_venues.head()
print('Total venues found:\n----------------------\n\t{}'.format(Berlin_venues.shape[0]))

print('\nTotal unique categories:\n----------------------\n\t{}'.format(Berlin_venues['Venue Category'].unique().shape[0]))
# View unique categories of all venues

Berlin_venues['Venue Category'].unique()
# Get dummy variables from Venue Category

Berlin_onehot = pd.get_dummies(Berlin_venues[['Venue Category']], prefix="", prefix_sep="")



# Add Neighbourhood Column

Berlin_onehot['Neighbourhood'] = Berlin_venues['Neighbourhood']



# Rearrange columns

columns = [Berlin_onehot.columns[-1]] + list(Berlin_onehot.columns[:-1])

Berlin_onehot = Berlin_onehot[columns]



# Column names that I think won't be useful in our analysis, so we drop them

Berlin_onehot.drop(columns=['Stationery Store', 'Salon / Barbershop', 'Drugstore', 'Grocery Store', 

                            'Supermarket', 'IT Services', 'Building', 'Farmers Market', 'Metro Station',

                            'Cemetery', 'Paper / Office Supplies Store', 'Tram Station', 'Bus Stop', 

                            'Post Office', 'Light Rail Station', 'Gas Station', 'Bank', 'Intersection', 

                            'Airport Service', 'Airport Lounge', 'Rental Car Location', 'Duty-free Shop',

                            'Neighborhood', 'Print Shop', 'Train Station',

                            'Tunnel', 'Windmill', 'Credit Union', 'Road', 'Insurance Office'], 

                   inplace=True)



# Create a DataFrame where Category is represented by venue frequency

Berlin_groups = Berlin_onehot.groupby('Neighbourhood').mean().reset_index()



Berlin_groups.head()
# Check if the venue is a resturant or not

cols_list = ['Neighbourhood']



# Loop through all columns/categories

for col in range(1, Berlin_groups.columns.shape[0]):

    

    # Check if the string contains specific words

    if 'Restaurant'in Berlin_groups.columns[col]:

        cols_list.append(Berlin_groups.columns[col])

    if 'BBQ' in Berlin_groups.columns[col]:

         cols_list.append(Berlin_groups.columns[col])

    if 'Pizza' in Berlin_groups.columns[col]:

         cols_list.append(Berlin_groups.columns[col])

    if 'Food' in Berlin_groups.columns[col]:

         cols_list.append(Berlin_groups.columns[col])

    if 'Steakhouse' in Berlin_groups.columns[col]:

         cols_list.append(Berlin_groups.columns[col])

            

            

Berlin_restaurants = Berlin_groups[cols_list]

Berlin_restaurants.head()
# Restaurants

print("Restaurants in Berlin:\n----------------------\n\n",Berlin_restaurants.columns)



print("\nTotal Restaurants:\n----------------------\n", Berlin_restaurants.shape[0])



print("\nTotal Categories:\n----------------------\n", Berlin_restaurants.shape[1])
# Create a Dataframe with top 5 common restaurant types for each neighbourhood

maximum_venues = 5

indicators = ['st', 'nd', 'rd']



# Create Column Names

columns = ['Neighbourhood']

for ind in np.arange(maximum_venues):

    try:

        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))

    except:

        columns.append('{}th Most Common Venue'.format(ind+1))

        

# Create a new DataFrame that will contain top 10 restaurants based on frequency

Berlin_top = pd.DataFrame(columns = columns)

Berlin_top['Neighbourhood'] = Berlin_groups['Neighbourhood']



for index in np.arange(Berlin_groups.shape[0]):

    Berlin_top.iloc[index, 1:] = return_top_venues(Berlin_restaurants.iloc[index, :], maximum_venues)

    

Berlin_top.head(10)
# Get the GeoJSON file for Berlin's Boroughs

!wget --quiet https://raw.githubusercontent.com/funkeinteraktiv/Berlin-Geodaten/master/berlin_bezirke.geojson -O berlin_bezirke.geojson



print('--- GeoJSON files downloaded ---')
# Read the GeoJSON file  

boroughs_JSON = r'berlin_bezirke.geojson'



# Create a Map for Berlin

Berlin_coordinates = geolocator.geocode('Berlin, Germany')

Berlin_map = folium.Map(location = [Berlin_coordinates.latitude, Berlin_coordinates.longitude], 

                        zoom_start=10)



# Create a Choropleth map for each Borough and respective population in 2010

Berlin_map.choropleth(

    geo_data = boroughs_JSON,

    data =  Berlin_boroughs,

    columns = ['Boroughs','Population'],

    key_on = 'feature.properties.name',

    fill_color='YlOrBr',

    fill_opacity = 0.8,

    line_opacity = 0.8,

    legend_name = 'Berlin Boroughs Population')



# In the below part, we create an overlay to add Popups to the map that displays 

# Borough name and Total Population



# Open GeoJSON data and assign it

with open(boroughs_JSON) as f:

    data = json.load(f)



# Add GeoJSON overlay with Html Popups, Just creating a GeoJSON for each borough and adding it to

# the map, and setting it's style to something that's almost transparent. Probably not the best implementation

# but, it's a work-around that I did.



for boroughs in Berlin_boroughs.itertuples():

    # Set current Index and assign borough name

    index = boroughs.Index

    

    # Get the borough data from the data GeoJSON we improted earlier

    borough_data = data['features'][index]['geometry']

    borough_name = data['features'][index]['properties']['name']

    

    # Get the Population

    borough_population = int(Berlin_boroughs.loc[Berlin_boroughs['Boroughs'] == borough_name].Population.values)

    

    # Create a HTML label for PopUp feature

    label = '<h3> {} </h3><p>Poplation: {:,}</p>'.format(borough_name, borough_population)

    

    # Create a GeoJSON with the data we read and set its style to something we can't see

    borough_gj = folium.GeoJson(data= borough_data, style_function=lambda feature: {

        'fillColor': '#000000',

        'color': 'white',

        'weight': 0,

        'fillOpacity': 0

    })

    borough_gj.add_child(folium.Popup(label))

    borough_gj.add_to(Berlin_map)



# View the map

Berlin_map
# Read the GeoJSON file  

boroughs_JSON = r'berlin_bezirke.geojson'



# Create a Map for Berlin

Berlin_coordinates = geolocator.geocode('Berlin, Germany')

Berlin_map = folium.Map(location = [Berlin_coordinates.latitude, Berlin_coordinates.longitude], 

                        zoom_start=10)



# Create a Choropleth map for each Borough and respective population in 2010

Berlin_map.choropleth(

    geo_data = boroughs_JSON,

    data =  Berlin_boroughs,

    columns = ['Boroughs','Density'],

    key_on = 'feature.properties.name',

    fill_color='YlOrBr',

    fill_opacity = 0.8,

    line_opacity = 0.8,

    legend_name = 'Berlin Population Density')



# In the below part, we create an overlay to add Popups to the map that displays 

# Borough name and Total Population



# Open GeoJSON data and assign it

with open(boroughs_JSON) as f:

    data = json.load(f)



# Add GeoJSON overlay with Html Popups, Just creating a GeoJSON for each borough and adding it to

# the map, and setting it's style to something that's almost transparent. Probably not the best implementation

# but, it's a work-around that I did.



for boroughs in Berlin_boroughs.itertuples():

    # Set current Index and assign borough name

    index = boroughs.Index

    

    # Get the borough data from the data GeoJSON we improted earlier

    borough_data = data['features'][index]['geometry']

    borough_name = data['features'][index]['properties']['name']

    

    # Get the Population

    borough_density = int(Berlin_boroughs.loc[Berlin_boroughs['Boroughs'] == borough_name].Density.values)

    

    # Create a HTML label for PopUp feature

    label = '<h3> {} </h3><p>Density: {:,}</p>'.format(borough_name, borough_density)

    

    # Create a GeoJSON with the data we read and set its style to something we can't see

    borough_gj = folium.GeoJson(data= borough_data, style_function=lambda feature: {

        'fillColor': '#000000',

        'color': 'white',

        'weight': 0,

        'fillOpacity': 0

    })

    borough_gj.add_child(folium.Popup(label))

    borough_gj.add_to(Berlin_map)



# View the map

Berlin_map
# Plot different Neighbourhoods on a map

Berlin_map = folium.Map(location = [Berlin_coordinates.latitude, Berlin_coordinates.longitude], zoom_start=10)



# Create a map with circle markers on each neighbourhood

for lat, lng, neighbourhood, borough in zip(Berlin_neighbourhoods['Latitude'], Berlin_neighbourhoods['Longitude'],

                                  Berlin_neighbourhoods['Neighbourhood'], Berlin_neighbourhoods['Borough']):

    label = '<h4>  {}, {}  </h4>'.format(neighbourhood, borough)

    label = folium.Popup(label)

    folium.CircleMarker(

        [lat, lng],

        radius = 6,

        popup=label,

        fill=True,

        fill_color='blue',

        fill_opcaity = 0.9,

        parse_html=False).add_to(Berlin_map)

    

# View the map

Berlin_map
# Create a Bar chart to plot each Borough's population

ax1 = Berlin_boroughs.sort_values(by='Population', ascending=False).plot(kind='bar', 

                                                         x='Boroughs', y='Population', figsize=(15, 3));

# Title and axis labels

ax1.set_title("Borough Population in Berlin (2010)");

ax1.set_xlabel("Borough");

ax1.set_ylabel("Total Population in 2010");





# Create a Bar chart to plot each Borough's Density

ax2 = Berlin_boroughs.sort_values(by='Density', ascending=False).plot(kind='bar', 

                                                         x='Boroughs', y='Density', figsize=(15, 3));

# Title and axis labels

ax2.set_title("Borough Density in Berlin (2010)");

ax2.set_xlabel("Borough");

ax2.set_ylabel("Total Density in 2010");
# Plot Neigbourhood Population as of 2008

ax1 = Berlin_neighbourhoods.sort_values(by='Population', ascending=False).plot(kind='bar', 

                                                         x='Neighbourhood', y='Population', figsize=(20, 5));



# Title and axis labels

ax1.set_title("Neighbourhoods Population in Berlin (2008)");

ax1.set_xlabel("Neighbourhood");

ax1.set_ylabel("Total Population in 2008");



# Plot Neigbourhood Population as of 2008

ax2 = Berlin_neighbourhoods.sort_values(by='Density', ascending=False).plot(kind='bar', 

                                                         x='Neighbourhood', y='Density', figsize=(20, 5));



# Title and axis labels

ax2.set_title("Neighbourhoods Density in Berlin (2008)");

ax2.set_xlabel("Neighbourhood");

ax2.set_ylabel("Total Density in 2008");
# Create an empty list for Neighbourhoods of Interest: 

neighbourhoods_of_interest = []



# Boroughs list that we choose from looking at the map/graphs

boroughs_list = ['Pankow', 'Tempelhof-Schöneberg', 'Mitte', 'Friedrichshain-Kreuzberg']



for boroughs in boroughs_list:

    neighbourhoods_of_interest.append(Berlin_neighbourhoods.loc[Berlin_neighbourhoods['Borough'] == boroughs]['Neighbourhood'])



# Convert into a 1D list

neighbourhoods_of_interest = list(itertools.chain.from_iterable(neighbourhoods_of_interest))



# Examine the data

neighbourhoods_of_interest
# Different competitor categories

competitors_category = ['BBQ Joint', 'Steakhouse', 'American Restaurant', 'New American Restaurant',

                        'Food Truck', 'Kebab Restaurant', 'Doner Restaurant', 'Turkish Restaurant']



# Create a new Competitors DataFrame

competitors = pd.DataFrame(columns = Berlin_venues.columns)



# Add items to the competitors DataFrame

for restaurant in competitors_category:

    competitors = competitors.append(Berlin_venues.loc[Berlin_venues['Venue Category'] == str(restaurant)],

                                     ignore_index=True)



competitors.head(10)
# Plot different Competitors on a map

Berlin_map = folium.Map(location = [Berlin_coordinates.latitude, Berlin_coordinates.longitude], zoom_start=10)



# Create a map with circle markers on each neighbourhood

for neighbourhood, venue, lat, lng, category in zip(competitors['Neighbourhood'], competitors['Venue'],

                                                   competitors['Venue Latitude'], competitors['Venue Longitude'],

                                                   competitors['Venue Category']):

    label = '<h4>  {}, {}  </h4>'.format(category, venue)

    label = folium.Popup(label)

    folium.CircleMarker(

        [lat, lng],

        radius = 8,

        popup=label,

        color='red',

        fill=True,

        fill_color='red',

        fill_opcaity = 1,

        parse_html=False).add_to(Berlin_map)

    

# View the map

Berlin_map
# Initial trial for Clustering Neighbourhoods, random guess for K=10

k = 12



Berlin_clusters = Berlin_restaurants.drop('Neighbourhood', 1)



kmeans = KMeans(n_clusters=k, random_state=42).fit(Berlin_clusters)

kmeans.labels_[0:50]
# Add Cluster labels to the DataFrame

Berlin_top.insert(0, 'Cluster Group', kmeans.labels_)



# Sort values of Berlin_top10 DataFrame by 'Neighbourhood' Column

Berlin_top.sort_values(by='Neighbourhood', ascending=True, inplace=True)



# Sort values of Berlin_neighbourhoods DataFrame by 'Neighbourhood' Column

Berlin_neighbourhoods.sort_values(by='Neighbourhood', ascending=True, inplace=True)



# Now both DataFrames should have matching Neighbourhood 

# Columns, as they were sorted to the same criteria



# Join DataFrames together

Berlin_top = Berlin_neighbourhoods.join(Berlin_top.set_index('Neighbourhood'), on='Neighbourhood')

Berlin_top.reset_index(drop=True, inplace=True)



# Check the DataFrame

Berlin_top.head()
# Optimal K-values using two different methods: Elbow and Silhoutte Method.



# Maximum K value

k_max = range(2, 14)

distortions = []



# Optimum K using elbow method

for k in k_max:

    kmeans = KMeans(n_clusters=k, init= 'k-means++', max_iter=500, random_state=42).fit(Berlin_clusters)

    # Add distorion value to distortions list

    distortions.append(sum(np.min(cdist(Berlin_clusters, 

                                        kmeans.cluster_centers_, 

                                        'euclidean'), axis=1)) / Berlin_clusters.shape[0])

# Plot Distortion Vs. K values

plt.figure(figsize=(15,5))

plt.plot(k_max, distortions, 'rx-')

plt.xlabel('K-Cluster Groups')

plt.ylabel('Distortions')

plt.title('Optimal number of clusters (Elbow Method)')

plt.show()



# Silhoutte Score Computation

s_val = []

for k in k_max:

    clusterer = KMeans(n_clusters=k, init='k-means++', max_iter=500, random_state=42)

    preds = clusterer.fit_predict(Berlin_clusters)

    centers = clusterer.cluster_centers_

    score = silhouette_score (Berlin_clusters, preds, metric='euclidean')

    s_val.append(score)

    

# Plot silhoutte score graph

plt.figure(figsize=(15,5))

plt.plot(k_max, s_val, 'rx-')

plt.xlabel('K-Cluster Groups')

plt.ylabel('Sihloutte Score')

plt.title('Optimal Number of Clusters (Sihloutte Method)')

plt.show()
# Run K-means with optimum K, based on elbow method.

k = 6

kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=500, random_state=42).fit(Berlin_clusters)



# Add Clusters to DataFrame 

Berlin_top['Cluster Group'] = kmeans.labels_



# Visualizate Neighbourhood clusters on a map

Berlin_map = folium.Map(location = [Berlin_coordinates.latitude, Berlin_coordinates.longitude], zoom_start=10)



# Color Scheme for Clusters

x = np.arange(k)

ys = [i + x + (i*x)**2 for i in range(k)]

colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))

rainbow = [colors.rgb2hex(i) for i in colors_array]



for lat, lng, neighbourhood, cluster in zip(Berlin_top['Latitude'], Berlin_top['Longitude'],

                                            Berlin_top['Neighbourhood'], Berlin_top['Cluster Group']):

    label = '<h2> {}</h2> <p><em>Cluster Group: {}</em></p>'.format(neighbourhood, cluster)

    label = folium.Popup(label)

    folium.CircleMarker(

    [lat, lng],

    radius = 8,

    popup=label,

    color=rainbow[int(cluster)],

    fill = True,

    fill_color = rainbow[int(cluster)],

    fill_opacity=1).add_to(Berlin_map)



# View map

Berlin_map
# Sort Berlin_top10 DataFrame

Berlin_top.sort_values(by= 'Cluster Group', ascending=True, inplace=True)

Berlin_top.reset_index(drop=True, inplace=True)



Berlin_top.groupby(by=['Cluster Group'])['Cluster Group'].count()
# Create empty list to store values

val = []



# Add cluster groups of neighbourhoods of interest to the values list

for n in range(0, len(neighbourhoods_of_interest)):

    val.append(int(Berlin_top.loc[Berlin_top['Neighbourhood'] == neighbourhoods_of_interest[n]]['Cluster Group']))



v = np.array(val)

for values in np.unique(v): print('Category Group: {}, Count: {}'.format(values, val.count(values)))

    

print('Most common Cluster Category within neighbourhoods_of_interest: {}'.format(max(val, key=val.count)))
# Cluster Group 0

Cluster0 = Berlin_top.loc[Berlin_top['Cluster Group'] == 0]

Cluster0.reset_index(drop=True)



# Cluster Group 1

Cluster1 = Berlin_top.loc[Berlin_top['Cluster Group'] == 1]

Cluster1.reset_index(drop=True)



# Cluster Group 3

Cluster3 = Berlin_top.loc[Berlin_top['Cluster Group'] == 3]

Cluster3.reset_index(drop=True)



# Cluster Group 4

Cluster4 = Berlin_top.loc[Berlin_top['Cluster Group'] == 4]

Cluster4.reset_index(drop=True)



Cluster1
# 1st Common Venue Cluster 0

ax0 = Cluster0.groupby(['1st Most Common Venue'], as_index=False).count().plot(kind='bar', 

                                                                         x='1st Most Common Venue', y='Cluster Group',

                                                                         figsize = (15,3));

ax0.set_title('1st Most Common Venue Count in Cluster Group 0');

ax0.set_xlabel('1st Most Common Venue');

ax0.set_ylabel('Count of Venues');





# 1st Common Venue Cluster 1

ax1 = Cluster1.groupby(['1st Most Common Venue'], as_index=False).count().plot(kind='bar', 

                                                                         x='1st Most Common Venue', y='Cluster Group',

                                                                         figsize = (15,3));

ax1.set_title('1st Most Common Venue Count in Cluster Group 1');

ax1.set_xlabel('1st Most Common Venue');

ax1.set_ylabel('Count of Venues');



# 1st Common Venue Cluster 3

ax3 = Cluster3.groupby(['1st Most Common Venue'], as_index=False).count().plot(kind='bar', 

                                                                         x='1st Most Common Venue', y='Cluster Group',

                                                                         figsize = (15,3));

ax3.set_title('1st Most Common Venue Count in Cluster Group 3');

ax3.set_xlabel('1st Most Common Venue');

ax3.set_ylabel('Count of Venues');



# 1st Common Venue Cluster 4

ax4 = Cluster4.groupby(['1st Most Common Venue'], as_index=False).count().plot(kind='bar', 

                                                                         x='1st Most Common Venue', y='Cluster Group',

                                                                         figsize = (15,3));

ax4.set_title('1st Most Common Venue Count in Cluster Group 4');

ax4.set_xlabel('1st Most Common Venue');

ax4.set_ylabel('Count of Venues');
# Plot different Competitors on a map

Berlin_map = folium.Map(location = [Berlin_coordinates.latitude, Berlin_coordinates.longitude], zoom_start=10)



# Create a Choropleth map for each Borough and respective Density

Berlin_map.choropleth(

    geo_data = boroughs_JSON,

    data =  Berlin_boroughs,

    columns = ['Boroughs','Density'],

    key_on = 'feature.properties.name',

    fill_color='YlOrBr',

    fill_opacity = 0.4,

    line_opacity = 0.5,

    legend_name = 'Berlin Population Density')



# Create a map with circle markers on each neighbourhood

for neighbourhood, venue, lat, lng, category in zip(competitors['Neighbourhood'], competitors['Venue'],

                                                   competitors['Venue Latitude'], competitors['Venue Longitude'],

                                                   competitors['Venue Category']):

    label = '<h4>  {}, {}  </h4>'.format(category, venue)

    label = folium.Popup(label)

    folium.CircleMarker(

        [lat, lng],

        radius = 8,

        popup=label,

        color='red',

        fill=True,

        fill_color='red',

        fill_opcaity = 1,

        parse_html=False).add_to(Berlin_map)

    

for neighbourhood, lat, lng in zip(Cluster1['Neighbourhood'], Cluster1['Latitude'], Cluster1['Longitude']):

    label = '<h4>  {}  </h4>'.format(neighbourhood)

    label = folium.Popup(label)

    folium.CircleMarker(

        [lat, lng],

        radius = 8,

        popup=label,

        color=rainbow[1],

        fill=True,

        fill_color=rainbow[1],

        fill_opcaity = 1,

        parse_html=False).add_to(Berlin_map)

    

# View the map

Berlin_map