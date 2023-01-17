print('Hello My name is Bashir Abubakar and welcome to this exploration!')
from bs4 import BeautifulSoup
import requests  # library to handle requests
import pandas as pd
import json  # library to handle JSON files
from pandas.io.json import json_normalize  # transform json files to pandas dataframes
from geopy.geocoders import Nominatim  # convert an address into latitude and longitude values

# Matplotlib and associated plotting modules
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# import k-means for clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes
import folium # map rendering library
# chart studio library
!pip install chart_studio
!pip install cufflinks
from chart_studio.plotly import plot, iplot
from plotly.offline import init_notebook_mode, iplot
import plotly.figure_factory as ff
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')
import plotly.graph_objs as go
import chart_studio.plotly as py
import plotly
import chart_studio
chart_studio.tools.set_credentials_file(username='bashman18', api_key='••••••••••')
init_notebook_mode(connected=True)

import numpy as np
import csv

print('All modules imported')
# Lets save the webpage for The Lagos State Data

lagos_state_link = 'https://en.wikipedia.org/wiki/List_of_Lagos_State_local_government_areas_by_population'
source = requests.get(lagos_state_link).text
soup = BeautifulSoup(source,'lxml')

# Let's print it to see what it looks like...

#print(soup.prettify())
table_data = soup.find_all('td')

# Let's view the table

table_data[:10]
type(table_data)
table_text = []

for data in table_data:
    table_text.append(data.text)
relevant_table_data = table_text[4:-3]

# Next let's see the first 3 elements
relevant_table_data[:]
table_dict={'LGA':[], 'POP':[]}
count = 0

for item in relevant_table_data:
    # First let's strip off the \n at the end
    item = item.strip('\n')
    try:
        item = int(item)
    except:
        # if second item after the int, append to POP
        if count > 0:
            # First let's remove the commas
            item = item.replace(',','')
            # Next let's convert to an integer so we can use it for calculations
            item = int(item)
            # Finally, let's append it to the Population list of the dictionary
            table_dict['POP'].append(item)
            count = 0
        else:
         # if first item after the int, append to LGA
            table_dict['LGA'].append(item)
            count +=1
lagos_df = pd.DataFrame(table_dict)

# Let's see the corresponding rows
lagos_df
def latitude_longitude(LGA):
    import time
    """ Method takes a Series object and returns
    a list of Latitude and corresponding Longitude data,
    using the geopy library.
    This method also prints out the coordinate data"""
    
    address = str(LGA)
    
    # We must define a geolocator user agent
    geolocator = Nominatim(user_agent="NG_explorer")
    location = geolocator.geocode(address)
    latitude = location.latitude
    longitude = location.longitude
    print('The geograpical coordinates of {} are lat {} and long {}.'.format(address, latitude, longitude))
    # WARNING: let 2 secs pass after calling each location lat/lon so that the geocode function would not crash as it crashes quite often
    time.sleep(5)  
    return [latitude, longitude]
lagos_df['latitude'] = lagos_df['LGA'].apply(latitude_longitude)
lagos_df.head()
lon_list = []
for i, j in lagos_df.iterrows():
    lon_list.append(j.latitude[1])
    lagos_df.iat[i,2] = j.latitude[0]

lagos_df['longitude'] = lon_list
lagos_df.head()
#next let's convert our dataframe to a csv file by calling the to_csv function
#lagos_df.to_csv(r'dataset.csv')
print(os.listdir("../input"))
# let's read in our data from csv
compiled_df=pd.read_csv('../input/dataset.csv')
compiled_df
# lets drop the column "unnamed"
compiled_df.drop(['Unnamed: 0'], axis = 1,inplace = True) 
compiled_df
CLIENT_ID = 'PJSZT3QA54UA1VYW0VH4AYT30HWXXIJG0LFDRCTITDB3X5CD' # your Foursquare ID
CLIENT_SECRET = 'TETQTTIFLDIS00B4FPS00HAHTJMS4KQILT12DNX05BKFGKUB' # your Foursquare Secret
VERSION = '20200606'
LIMIT = 100
print('Your credentials:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)
compiled_df.head()
#let's returnt the geographical coordinates of Lagos state with geopy
address = 'Lagos, NG'

geolocator = Nominatim(user_agent="ng_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Lagos State are {}, {}.'.format(latitude, longitude))
# next let's create a map showcasing lagos state LGAs
map_lagos_state = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, LGA  in zip(compiled_df['latitude'], compiled_df['longitude'], compiled_df['LGA']):
    label = '{}'.format(LGA)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=6,
        popup=label,
        color='green',
        fill=True,
        fill_color='white',
        fill_opacity=0.7,
        parse_html=False).add_to(map_lagos_state)  
    
map_lagos_state
def getNearbyVenues(names, latitudes, longitudes, radius=10000):
    
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
    nearby_venues.columns = ['LGA', 
                  'LGA_Latitude', 
                  'LGA_Longitude', 
                  'Venue', 
                  'Venue_Latitude', 
                  'Venue_Longitude', 
                  'Venue_Category']
    
    return(nearby_venues)
lagos_state_venues = getNearbyVenues(names= compiled_df['LGA'],
                                     latitudes= compiled_df['latitude'],
                                     longitudes= compiled_df['longitude'])
lagos_state_venues.shape
lagos_state_venues.head(5)
lgs_venue=lagos_state_venues.groupby('LGA').count()
lgs_venue
#let's visualize the data

lgs_venue.iplot(kind='bar',xTitle='LGA',
    yTitle='Venue',
    mode='markers',
    color='crimson',
    y='Venue',
    title='Total number of venues per LGA')
           

print('There are {} unique categories of venues returned for Lagos State.'.format(lagos_state_venues['Venue_Category'].nunique()))
# one hot encoding
lagos_onehot = pd.get_dummies(lagos_state_venues[['Venue_Category']], prefix="", prefix_sep="")

# add LGA column back to dataframe
lagos_onehot['LGA'] = lagos_state_venues['LGA'] 

# move LGA column to the first column
fixed_columns = [lagos_onehot.columns[-1]] + list(lagos_onehot.columns[:-1])
lagos_onehot = lagos_onehot[fixed_columns]

lagos_onehot.head()
lagos_onehot.shape
lagos_grouped = lagos_onehot.groupby('LGA').mean().reset_index()

lagos_grouped
lagos_grouped.shape
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['LGA']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
lga_venues_sorted = pd.DataFrame(columns=columns)
lga_venues_sorted['LGA'] = lagos_grouped['LGA']

for ind in np.arange(lagos_grouped.shape[0]):
    lga_venues_sorted.iloc[ind, 1:] = return_most_common_venues(lagos_grouped.iloc[ind, :], num_top_venues)

lga_venues_sorted.head(10)
# set number of clusters
kclusters = 5

lagos_grouped_clustering = lagos_grouped.drop('LGA', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(lagos_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_
compiled_df.tail()
# insert clustering labels
lga_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

# Let's select LGA, latitude and longitude data columns from lagos_df
lagos_merged = compiled_df[['LGA', 'latitude', 'longitude']]

# merge lagos_merged with lga_venues_sorted to add latitude/longitude for each neighborhood
lagos_merged = lagos_merged.join(lga_venues_sorted.set_index('LGA'), on='LGA')
lagos_merged
import plotly.express as px
fig = px.scatter(lagos_merged, x="LGA", y="Cluster Labels")
fig.show(renderer='notebook_connected')
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=10)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(lagos_merged['latitude'], lagos_merged['longitude'], lagos_merged['LGA'], lagos_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color= rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters
cluster1 = lagos_merged[lagos_merged['Cluster Labels'] == 0]
cluster2 = lagos_merged[lagos_merged['Cluster Labels'] == 1]
cluster3 = lagos_merged[lagos_merged['Cluster Labels'] == 2]
cluster4 = lagos_merged[lagos_merged['Cluster Labels'] == 3]
cluster5 = lagos_merged[lagos_merged['Cluster Labels'] == 4]

for i in range(5):
    x = lagos_merged[lagos_merged['Cluster Labels'] == i]
    print('cluster'+str(i+1) + ' shape is {}'.format(x.shape))
cluster_one = lagos_merged[lagos_merged['Cluster Labels'] == 0]
cluster_one

cluster1_lgas = (cluster1['LGA'])
cluster1_lgas
cluster1_lgas.to_frame() 

cluster_two = lagos_merged[lagos_merged['Cluster Labels'] == 1]
cluster_two
cluster_three = lagos_merged[lagos_merged['Cluster Labels'] == 2]
cluster_three
cluster_four = lagos_merged[lagos_merged['Cluster Labels'] == 3]
cluster_four
cluster_five = lagos_merged[lagos_merged['Cluster Labels'] == 4]
cluster_five
from PIL import Image # Used for converting images into arrays
!pip install wordcloud
# import wordcloud package and its set of stopwords
from wordcloud import WordCloud, STOPWORDS

print ('Wordcloud is installed and imported!')
stopwords = set(STOPWORDS)
# Let's select only the columns in lagos_merged DataFrame that we need
selection = lagos_merged.iloc[:, 4:]

# Let's view the first 5 rows of our selection
selection.head()
# Next lets write code to iterate through each column of our selection DataFrame and append each category word to a string object

col_list = list(selection.columns)

words = ''

for i in col_list:
    for j in list(selection[i]):
        words += j
        words += ', '

print(len(words))
words
# Instantiate a WordCloud object
top_venues_wc = WordCloud(max_font_size=50,
          background_color='white',
          max_words= len(words),
          stopwords=stopwords)

# Generate the WordCloud
top_venues_wc.generate(words)
# Generate the WordCloud
top_venues_wc.generate(words)

# Set the Size
plt.figure(figsize=(10, 15))

# Display the WC
plt.imshow(top_venues_wc, interpolation='bilinear')
plt.axis('off')
plt.show()