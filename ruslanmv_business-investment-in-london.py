# library for BeautifulSoup, for web scrapping

from bs4 import BeautifulSoup

# library to handle data in a vectorized manner

import numpy as np

# library for data analsysis

import pandas as pd

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

# library to handle JSON files

import json

print('numpy, pandas, ..., imported...')

!pip -q install geopy

print('geopy installed...')

# convert an address into latitude and longitude values

from geopy.geocoders import Nominatim

print('Nominatim imported...')

# library to handle requests

import requests

print('requests imported...')

# tranform JSON file into a pandas dataframe

from pandas.io.json import json_normalize

print('json_normalize imported...')

# Matplotlib and associated plotting modules

import matplotlib.cm as cm

import matplotlib.colors as colors

print('matplotlib imported...')

# import k-means from clustering stage

from sklearn.cluster import KMeans

print('Kmeans imported...')

# install the Geocoder

!pip -q install geocoder

import geocoder

# import time

import time

!pip -q install folium

print('folium installed...')

import folium # map rendering library

print('folium imported...')

from pandas import ExcelWriter

from pandas import ExcelFile

print('...Done')



import warnings

warnings.filterwarnings('ignore')
import os

print(os.listdir("../input/unemployment-in-london"))
unemployment_ratio_df = pd.read_csv("../input/unemployment-in-london/unemployment_ratio.csv") 

unemployment_ratio_df2=unemployment_ratio_df.dropna()

unemployment_ratio_df2.rename({'Unnamed: 1': '2001', 'Unnamed: 2': '2011'}, axis=1, inplace=True)

unemployment_ratio_df2.reset_index(drop=True, inplace=True)

unemployment_ratio_df2['2011'] = unemployment_ratio_df2['2011'].str[:-1].astype(float)

unemployment_ratio_df2['2001'] = unemployment_ratio_df2['2001'].str[:-1].astype(float)

unemployment_ratio_df3=unemployment_ratio_df2.sort_values(by='2011', ascending=True)

unemployment_ratio_df3.plot(x="Unemployment ratio by borough", y=["2001", "2011"], kind="bar")
unemployment_ratio_df3.head()
change_unemployment_ratio_df = pd.read_csv("../input/unemployment-in-london/change_unemployment_ratio.csv") 

change_unemployment_ratio_df=change_unemployment_ratio_df.dropna()

change_unemployment_ratio_df.head()

change_unemployment_ratio_df.rename({'Change in unemployment ratio 2011-13 to 2014-16': 'Borough', 'Unnamed: 1': 'Percentual'}, axis=1, inplace=True)

change_unemployment_ratio_df['Percentual'] = change_unemployment_ratio_df['Percentual'].str[:-1].astype(float)

change_unemployment_ratio_df2=change_unemployment_ratio_df.sort_values(by='Percentual', ascending=True)

change_unemployment_ratio_df2.plot(kind='bar',x='Borough',y='Percentual')
change_unemployment_ratio_df2.head(10)
change_unemployment_ratio_df2["Percentual"].mean()
# library for BeautifulSoup

from bs4 import BeautifulSoup

wikipedia_link = 'https://en.wikipedia.org/wiki/List_of_areas_of_London'

wikipedia_page = requests.get(wikipedia_link)
# Cleans html file

soup = BeautifulSoup(wikipedia_page.content, 'html.parser')

# This extracts the "tbody" within the table where class is "wikitable sortable"

table = soup.find('table', {'class':'wikitable sortable'}).tbody

# Extracts all "tr" (table rows) within the table above

rows = table.find_all('tr')

# Extracts the column headers, removes and replaces possible '\n' with space for the "th" tag

columns = [i.text.replace('\n', '')

           for i in rows[0].find_all('th')]

# Converts columns to pd dataframe

df = pd.DataFrame(columns = columns)

'''

Extracts every row with corresponding columns then appends the values to the create pd dataframe "df". The first row (row[0]) is skipped because it is already the header

'''

for i in range(1, len(rows)):

    tds = rows[i].find_all('td')    

    if len(tds) == 7:

        values = [tds[0].text, tds[1].text, tds[2].text.replace('\n', ''.replace('\xa0','')), tds[3].text, tds[4].text.replace('\n', ''.replace('\xa0','')), tds[5].text.replace('\n', ''.replace('\xa0','')), tds[6].text.replace('\n', ''.replace('\xa0',''))]

    else:

        values = [td.text.replace('\n', '').replace('\xa0','') for td in tds]

        

        df = df.append(pd.Series(values, index = columns), ignore_index = True)

        df
df.head(5)
df.columns = ['Location', 'Borough', 'Post-town', 'Postcode',

       'Dial-code', 'OSgridref']
df.columns
df.head()
# Remove Borough reference numbers with []

df['Borough'] = df['Borough'].map(lambda x: x.rstrip(']').rstrip('0123456789').rstrip('['))
df.head()
df0 = df.drop('Postcode', axis=1).join(df['Postcode'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('Postcode'))
df0.head()
df1 = df0[['Location', 'Borough', 'Postcode', 'Post-town']].reset_index(drop=True)
df1.head()
df2 = df1 # assigns df1 to df2

df21 = df2[df2['Post-town'].str.contains('LONDON')]

df21.shape
df3 = df21[['Location', 'Borough', 'Postcode']].reset_index(drop=True)
df3.head()
df_london = df3 # re-assigns to df_london

# Strips whitespaces before postcode

df_london.Postcode = df_london.Postcode.str.strip()

# New dataframe for South East London postcodes - df_se

df_sw = df_london[df_london['Postcode'].str.startswith(('SW'))].reset_index(drop=True)
df_sw.head(10)
demograph_link = 'https://en.wikipedia.org/wiki/Demography_of_London'

demograph_page = requests.get(demograph_link)

soup1 = BeautifulSoup(demograph_page.content, 'html.parser')

table1 = soup1.find('table', {'class':'wikitable sortable'}).tbody

rows1 = table1.find_all('tr')

columns1 = [i.text.replace('\n', '')

 for i in rows1[0].find_all('th')]

demo_london = pd.DataFrame(columns = columns1)

for j in range(1, len(rows1)):

    tds1 = rows1[j].find_all('td')

    if len(tds1) == 7:

        values1 = [tds1[0].text, tds1[1].text, tds1[2].text.replace('\n', ''.replace('\xa0','')), tds1[3].text, tds1[4].text.replace('\n', ''.replace('\xa0','')), tds1[5].text.replace('\n', ''.replace('\xa0',''))]

    else:

        values1 = [td1.text.replace('\n', '').replace('\xa0','') for td1 in tds1]

        

        demo_london = demo_london.append(pd.Series(values1, index = columns1), ignore_index = True)

demo_london
#converting string to float
demo_london['White'] = demo_london['White'].astype('float')

demo_london_sorted = demo_london.sort_values(by='White', ascending = False)

demo_london_sorted.head(10)
demo_london_sorted["White"].mean()
df_sw_top = df_sw[df_sw['Borough'].isin(['Hammersmith and Fulham','Camden', 'Merton', 'Barnet','Greenwich', 'Westminster'])].reset_index(drop=True)

df_sw_top
df_sw_top.shape
# Geocoder starts here

# Defining a function to use --> get_latlng()'''

def get_latlng(arcgis_geocoder):

    

    # Initialize the Location (lat. and long.) to "None"

    lat_lng_coords = None

    

    # While loop helps to create a continous run until all the location coordinates are geocoded

    while(lat_lng_coords is None):

        g = geocoder.arcgis('{}, London, United Kingdom'.format(arcgis_geocoder))

        lat_lng_coords = g.latlng

    return lat_lng_coords

# Geocoder ends here
sample = get_latlng('SW6')
sample
ga = geocoder.geocodefarm(sample, method = 'reverse')

ga

start = time.time()

postal_codes = df_sw_top['Postcode']    

coordinates = [get_latlng(postal_code) for postal_code in postal_codes.tolist()]

end = time.time()

print("Time of execution: ", end - start, "seconds")
df_sw_loc = df_sw_top

# The obtained coordinates (latitude and longitude) are joined with the dataframe as shown

df_sw_coordinates = pd.DataFrame(coordinates, columns = ['Latitude', 'Longitude'])

df_sw_loc['Latitude'] = df_sw_coordinates['Latitude']

df_sw_loc['Longitude'] = df_sw_coordinates['Longitude']

df_sw_loc.head(5)
df_sw_loc.shape
import json

filename = '../input/dataset/credential.json'

with open(filename) as f:

    data = json.load(f)
CLIENT_ID = data['CLIENT_ID'] #Foursquare )FS) ID

CLIENT_SECRET = data['CLIENT_SECRET'] # FS Secret

VERSION = data['VERSION'] # FS API version


#LIMIT = 30

#print('Your credentails:')

#print('CLIENT_ID: ' + CLIENT_ID)

#print('CLIENT_SECRET:' + CLIENT_SECRET)
# Resets the current index to a new

sw_df = df_sw_loc.reset_index().drop('index', axis = 1)

sw_df.loc[sw_df['Location'] == 'Fulham']
Fulham_lat = sw_df.loc[2, 'Latitude']

Fulham_long = sw_df.loc[2, 'Longitude']

Fulham_loc = sw_df.loc[2, 'Location']

Fulham_postcode = sw_df.loc[2, 'Postcode']

print('The latitude and longitude values of {} with postcode {}, are {}, {}.'.format(Fulham_loc, Fulham_postcode, Fulham_lat, Fulham_long))
# Credentials are provided already for this part

LIMIT = 50 # limit of number of venues returned by Foursquare API

radius = 1500 # define radius

url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(

    CLIENT_ID, 

    CLIENT_SECRET, 

    VERSION, 

    Fulham_lat, 

    Fulham_long, 

    radius, 

    LIMIT)

# displays URL

#url
results = requests.get(url).json()

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
import numpy as np

import pandas as pd
venues = results['response']['groups'][0]['items']

    

nearby_venues = json_normalize(venues) # flatten JSON

# filter columns

filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']

nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row

nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns

nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head(10)
nearby_venues_Fulham_unique = nearby_venues['categories'].value_counts().to_frame(name='Count')

nearby_venues_Fulham_unique.head(5)
def getNearbyVenues(names, latitudes, longitudes, radius=500):

    

    venues_list=[]

    for name, lat, lng in zip(names, latitudes, longitudes):

        print(name)

            

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

                  'Venue Latitude', 'Venue Longitude', 

                  'Venue Category']

    

    return(nearby_venues)
sw_venues = getNearbyVenues(names=sw_df['Location'],

                                   latitudes=sw_df['Latitude'],

                                   longitudes=sw_df['Longitude']

                                  )
sw_venues.shape
sw_venues.head(5)
sw_venues.groupby('Neighborhood').count()
print('There are {} uniques categories.'.format(len(sw_venues['Venue Category'].unique())))
sw_venue_unique_count = sw_venues['Venue Category'].value_counts().to_frame(name='Count')
sw_venue_unique_count.head()
address = 'London, United Kingdom'

geolocator = Nominatim(user_agent="ln_explorer")

location = geolocator.geocode(address)

latitude = location.latitude

longitude = location.longitude

print('The geograpical coordinate of London are {}, {}.'.format(latitude, longitude))
map_london = folium.Map(location = [latitude, longitude], zoom_start = 11)

map_london
# Adding markers to map

for lat, lng, borough, loc in zip(sw_df['Latitude'], 

                                  sw_df['Longitude'],

                                  sw_df['Borough'],

                                  sw_df['Location']):

    label = '{} - {}'.format(loc, borough)

    label = folium.Popup(label, parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius=5,

        popup=label,

        color='blue',

        fill=True,

        fill_color='#3186cc',

        fill_opacity=0.7).add_to(map_london)  

    

display(map_london)
sw_df
# one hot encoding

sw_onehot = pd.get_dummies(sw_venues[['Venue Category']], prefix = "", prefix_sep = "")
# add neighborhood column back to dataframe

sw_onehot['Neighborhood'] = sw_venues['Neighborhood']
# move neighborhood column to the first column

fixed_columns = [sw_onehot.columns[-1]] + list(sw_onehot.columns[:-1])

sw_onehot = sw_onehot[fixed_columns]
#sw_onehot.head()
# To check the Bakery:

#sw_onehot.loc[sw_onehot['Bakery'] != 0]
sw_grouped = sw_onehot.groupby('Neighborhood').mean().reset_index()

num_top_venues = 10 # Top common venues needed

for hood in sw_grouped['Neighborhood']:

    print("----"+hood+"----")

    temp = sw_grouped[sw_grouped['Neighborhood'] == hood].T.reset_index()

    temp.columns = ['venue', 'freq']

    temp = temp.iloc[1:]

    temp['freq'] = temp['freq'].astype(float)

    temp = temp.round({'freq': 2})

    print(temp.sort_values('freq', ascending = False).reset_index(drop = True).head(num_top_venues))

    print('\n')
def return_most_common_venues(row, num_top_venues):

    row_categories = row.iloc[1:]

    row_categories_sorted = row_categories.sort_values(ascending = False)

    

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

neighbourhoods_venues_sorted = pd.DataFrame(columns=columns)

neighbourhoods_venues_sorted['Neighborhood'] = sw_grouped['Neighborhood']

for ind in np.arange(sw_grouped.shape[0]):

    neighbourhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(sw_grouped.iloc[ind, :], num_top_venues)

neighbourhoods_venues_sorted.head(5)
sw_grouped_clustering = sw_grouped.drop('Neighborhood', 1)

# set number of clusters

kclusters = 5

# run k-means clustering

kmeans = KMeans(n_clusters = kclusters, random_state=0).fit(sw_grouped_clustering)

# check cluster labels generated for each row in the dataframe

kmeans.labels_[0:10]
# add clustering labels

neighbourhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

sw_merged = sw_df

# match/merge SE London data with latitude/longitude for each neighborhood

sw_merged_latlong = sw_merged.join(neighbourhoods_venues_sorted.set_index('Neighborhood'), on = 'Location')

sw_merged_latlong.head(5)
sw_clusters=sw_merged_latlong


# create map

map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters

x = np.arange(kclusters)

ys = [i + x + (i*x)**2 for i in range(kclusters)]

colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))

rainbow = [colors.rgb2hex(i) for i in colors_array]
# add markers to the map

markers_colors = []

for lat, lon, poi, cluster in zip(sw_clusters['Latitude'], sw_clusters['Longitude'], sw_clusters['Location'], sw_clusters['Cluster Labels']):

    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)

    folium.CircleMarker(

        [lat, lon],

        radius=20,

        popup=label,

        color=rainbow[cluster-1],

        fill=True,

        fill_color=rainbow[cluster-1],

        fill_opacity=0.7).add_to(map_clusters)
display(map_clusters)
# Cluster 1

cluster1=sw_clusters.loc[sw_clusters['Cluster Labels'] == 0, sw_clusters.columns[[1] + list(range(5, sw_clusters.shape[1]))]]

# Cluster 2

cluster2=sw_clusters.loc[sw_clusters['Cluster Labels'] == 1, sw_clusters.columns[[1] + list(range(5, sw_clusters.shape[1]))]]

# Cluster 3

cluster3=sw_clusters.loc[sw_clusters['Cluster Labels'] == 2, sw_clusters.columns[[1] + list(range(5, sw_clusters.shape[1]))]]
# Cluster 4

cluster4=sw_clusters.loc[sw_clusters['Cluster Labels'] == 3, sw_clusters.columns[[1] + list(range(5, sw_clusters.shape[1]))]]
# Cluster 5

cluster5=sw_clusters.loc[sw_clusters['Cluster Labels'] == 4, sw_clusters.columns[[1] + list(range(5, sw_clusters.shape[1]))]]
cluster1
cluster2
cluster5
cluster4
cluster3