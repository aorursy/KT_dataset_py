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
import pandas as pd
import locale
import requests
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import json
from math import sin, cos, sqrt, atan2, radians
from sklearn.cluster import KMeans
import matplotlib.path as mpltPath
locale.setlocale(locale.LC_ALL, '')
!conda install -c conda-forge folium=0.5.0 --yes
import folium
from folium import plugins
# Read the same file from IBM COS
!wget -q -O 'Neighborhood Tabulation Areas.geojson' https://cloud-object-storage-sf-cos-standard-pyhton2.s3.us-south.cloud-object-storage.appdomain.cloud/Neighborhood%20Tabulation%20Areas%20Manhattan.geojson 'Neighborhood Tabulation Areas.geojson'
print('Data downloaded!')
newyork_data_filename = 'Neighborhood Tabulation Areas.geojson'
with open(newyork_data_filename) as json_data:
    newyork_data = json.load(json_data)
# Create New York Area Tabulation Data Polygons
neighborhoods_polygons = {}
for neighborhood in newyork_data['features']:
    name = neighborhood['properties']['ntaname']
    neighborhoods_polygons[name] = neighborhood['geometry']
# Function to define NYC Tab Area by latitude, longitude
def define_tab_area(latitude, longitude):
    point = [[longitude,latitude]]
    for k,v in neighborhoods_polygons.items():                
        polygon_shapes = v['coordinates']      
        if len(polygon_shapes) == 1:
            path = mpltPath.Path(polygon_shapes[0][0])
            if path.contains_points(point):
                return k
        else:
            for p in polygon_shapes:
                path = mpltPath.Path(p[0])
                if path.contains_points(point):
                    return k        
    return 'Not defined'
        
define_tab_area(40.72290,-73.98199)
#Neighborhood Tabulation Areas.geojson file contains only polygon area cordinates for each Neighborhoods
# So we need to define a Centroid point's 'latitude', 'longitude' for each Manhattan's Neighborhoods
# We re-calculated it, made some manual correction because Nominatim service is not quite accurate and stored in NYC_Neiborhood_Lat_Lon_Man.csv in IBM COS
# for i in range(len(newyork_data['features']) -1, -1, -1):
#     if newyork_data['features'][i]["properties"]["boro_name"] != 'Manhattan':
#         del newyork_data['features'][i]      

# from  geopy.geocoders import Nominatim
# import time
# neighborhoods_data = newyork_data['features']
# data = []
# geolocator = Nominatim(user_agent="courseracapstone")
# for n in neighborhoods_data:
#     city_split =n['properties']['ntaname'].split('-')[0]
#     city =n['properties']['ntaname']
#     print(city_split)
#     if city_split == 'SoHo':
#         city_split = 'Soho'
#     if city_split == 'Battery Park City':
#         city_split = 'Wall Street'   
#     if city_split == 'Clinton':
#         city_split = "Hell's Kitchen"   
#     if city_split == 'Central Harlem North':
#         city_split = "Harlem North"
#     if city_split == 'Central Harlem South':
#         city_split = "Harlem South"                        
        
#     loc = geolocator.geocode(city_split + ", New York")    
#     print("latitude is :-" ,loc.latitude,"\nlongtitude is:-" ,loc.longitude)
#     data.append([city,loc.latitude,loc.longitude])  
#     time.sleep(1)

# neighborhoods_geo = pd.DataFrame(data, columns = ['Neighborhood', 'Latitude', 'Longitude'])
# neighborhoods_geo    

# Read pre-calculated NYC_Neiborhood_Lat_Lon_Man.csv Centroids from IBM COS
neighborhoods_geo = pd.read_csv('https://cloud-object-storage-sf-cos-standard-pyhton2.s3.us-south.cloud-object-storage.appdomain.cloud/NYC_Neiborhood_Lat_Lon_Man.csv')
neighborhoods_geo.head()
# We read already extracted file from IBM COS
df= pd.read_csv('https://s3.us-south.cloud-object-storage.appdomain.cloud/cloud-object-storage-sf-cos-standard-pyhton2/listings_NewYork_2019.csv', parse_dates=['last_scraped', 'last_review'])
df.shape
df.head()
# Filter Accomodations 
df_t = df[(df.neighbourhood_group_cleansed == 'Manhattan')
               & (df.number_of_reviews >= 10) 
               & (df.availability_365 >= 10)
               & (df.city=='New York') 
               & (df.state =='NY' )
               & (df.last_scraped > '2019-10-01')
               & (df.last_review > '2019-10-01')
               & (~df.room_type.isin(['Shared room']))
               & (~df.property_type.isin(['Camper/RV', 'Hostel']))
               & (df.minimum_nights < 3 )
              ].copy()
!pip install Babel
import babel
from babel.numbers import format_number, format_decimal, format_percent, parse_decimal
# Select subset of original columns
# Cleaning the data
# Convert Price strings into Float
# Change some Strings Columns into Numeric

df = df_t[['id',  'name', 'last_review', 'listing_url', 'picture_url', 'neighbourhood_group_cleansed', 'neighbourhood_cleansed', 'review_scores_rating',             
             'latitude', 'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms',
              'square_feet', 'price', 'security_deposit', 'cleaning_fee',
             'minimum_nights', 'number_of_reviews_ltm', 'reviews_per_month',
             'number_of_reviews', 'availability_365']].copy()
df['price'].fillna('$0', inplace=True)
df['security_deposit'].fillna('$0', inplace=True)
df['cleaning_fee'].fillna('$0', inplace=True)
df.fillna('0', inplace=True)
df= df.astype({'bathrooms':'int','bedrooms':'int', 'square_feet':'int'})
df['price'] = df['price'].apply(lambda x: x.strip("$"))
df['price'] = df['price'].apply(lambda money:  babel.numbers.parse_decimal(money, locale='en'))
df['security_deposit'] = df['security_deposit'].apply(lambda x: x.strip("$"))
df['security_deposit'] = df['security_deposit'].apply(lambda money:  babel.numbers.parse_decimal(money, locale='en'))
df['cleaning_fee'] = df['cleaning_fee'].apply(lambda x: x.strip("$"))
df['cleaning_fee'] = df['cleaning_fee'].apply(lambda money:  babel.numbers.parse_decimal(money, locale='en'))
# We read this file from IBM COS
df_crime= pd.read_csv('https://s3.us-south.cloud-object-storage.appdomain.cloud/cloud-object-storage-sf-cos-standard-pyhton2/NYPD_Crime_Manhattan_2019.csv')
df_crime.shape
df_crime[['Latitude','Longitude']] = df_crime[['Latitude','Longitude']].apply(lambda x: x.str.replace(',','.').astype(float))
df_crime.head()
# Filter crimes
df_manhattan_crime = df_crime[ (df_crime.BORO_NM =='MANHATTAN') & (df_crime.LAW_CAT_CD.isin(['FELONY', 'MISDEMEANOR']))]
df_manhattan_crime.shape
df_manhattan_crime['tab_area'] =  df_manhattan_crime.apply(lambda row : define_tab_area(row['Latitude'], row['Longitude']), axis = 1) 
df_manhattan_crime = df_manhattan_crime[~(df_manhattan_crime['tab_area'] == 'Not defined')]
df_manhattan_crime.shape
#Function to calculate a distance between two points in km
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = 0.5 - cos((lat2 - lat1) * p)/2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p))/2
    return 12742 * sin(sqrt(a))
# Function to calculate crimes nearby each accomodation
def calculate_crimes(lat1, lon1):
    num_of_crimes = 0
    for rowc in df_manhattan_crime.itertuples(index=False):
        d = distance(lat1, lon1, rowc.Latitude, rowc.Longitude)
        if d <= radius_km:
            num_of_crimes += 1
    return num_of_crimes

#df['crimes'] = df.apply(lambda row: calculate_crimes(row['latitude'], row['longitude']), axis=1)
# We pre-calculate number of crimes in radius of 100 meters from each accommodation
# because it takes about 50 minutes in Python.
# We added calculated number of crimes for each accomodation to our selected Manhattan Airbnb data listing.
# Read it from IBM COS
radius_km = 0.1
df_airnb= pd.read_csv('https://s3.us-south.cloud-object-storage.appdomain.cloud/cloud-object-storage-sf-cos-standard-pyhton2/airnb_nyc_crime_apply.csv')
df_airnb.shape
df_airnb.head()
df_airnb['price_per_person'] =  (df_airnb['price'] + df_airnb['cleaning_fee'])/df_airnb['accommodates']
df_airnb['full_price'] = df_airnb['price'] + df_airnb['cleaning_fee']
df_airnb['tab_area'] =  df_airnb.apply(lambda row : define_tab_area(row['latitude'], row['longitude']), axis = 1) 
df_airnb.head()
sorted(df_airnb['tab_area'].unique())
bp = df_airnb[['tab_area','full_price']].boxplot(column='full_price', by='tab_area',vert=False, fontsize=9, figsize=(12,10) )
bp.get_figure().gca().set_title("Full Price by Neighbourhood")
bp.get_figure().gca().set_xlabel('Full price')
bp.get_figure().suptitle('')
Q1 = df_airnb['full_price'].quantile(0.25)
Q3 = df_airnb['full_price'].quantile(0.75)
IQR = Q3 - Q1

filter = (df_airnb['full_price'] >= Q1 - 1.5 * IQR) & (df_airnb['full_price'] <= Q3 + 1.5 *IQR)
df_airnb_norm = df_airnb.loc[filter]  
df_airnb_norm.shape
bp = df_airnb_norm[['tab_area','full_price']].boxplot(column='full_price', by='tab_area',vert=False, fontsize=9, figsize=(12,10))
bp.get_figure().gca().set_title("Full Price by Neighbourhood")
bp.get_figure().gca().set_xlabel('Full price')
bp.get_figure().suptitle('')
neighbourhood_crime_price_merged = df_airnb_norm.copy()
neighbourhood_crime_price_merged = neighbourhood_crime_price_merged.groupby("tab_area", as_index=False).agg({'crimes':'mean', 'price_per_person':'mean', 'accommodates':'count'})
neighbourhood_crime_price_merged.columns=['tab_area', 'mean_crimes', 'mean_price_per_person', 'accommodates']
neighbourhood_crime_price_merged[['tab_area', 'mean_price_per_person', 'accommodates', 'mean_crimes']].sort_values('mean_price_per_person')
neighbourhood_crime_price_merged3 = neighbourhood_crime_price_merged.copy()
neighbourhood_crime_price_merged3 = neighbourhood_crime_price_merged3.merge(neighborhoods_geo[['Latitude', 'Longitude','Neighborhood']], how = 'left', left_on = 'tab_area', right_on = 'Neighborhood').drop(columns= ['Neighborhood'])
fig, ax = plt.subplots(figsize=(10,8))
x = neighbourhood_crime_price_merged3.index
y = neighbourhood_crime_price_merged3.accommodates
xticks = neighbourhood_crime_price_merged3.tab_area
ax.barh(x, y)
ax.set_yticks(x)
ax.set_yticklabels(xticks)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Accommodations')
ax.set_title('Accommodations Number Total by Neighbourhood on Manhattan in 2019')
plt.tick_params(labelsize=9)
plt.show()
df_manhattan_crime_neighbourhood = df_manhattan_crime.copy()
df_manhattan_crime_neighbourhood = df_manhattan_crime_neighbourhood.groupby("tab_area", as_index=False).agg({'CMPLNT_NUM':'count'})
df_manhattan_crime_neighbourhood.columns=['tab_area', 'crimes_summary']
df_manhattan_crime_neighbourhood.sort_values('crimes_summary')
fig, ax = plt.subplots(figsize=(10, 8))
x = df_manhattan_crime_neighbourhood.index
y = df_manhattan_crime_neighbourhood.crimes_summary
xticks = df_manhattan_crime_neighbourhood.tab_area
ax.barh(x, y)
ax.set_yticks(x)
ax.set_yticklabels(xticks)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Crimes')
ax.set_title('Neighborhoods Total Crimes Records on Manhattan in 2019')
plt.tick_params(labelsize=9)
plt.show()

df_manhattan_crime_neighbourhood2 = df_manhattan_crime_neighbourhood.copy()
df_manhattan_crime_neighbourhood2 = df_manhattan_crime_neighbourhood2.merge(neighborhoods_geo[['Latitude', 'Longitude','Neighborhood']], how = 'left', left_on = 'tab_area', right_on = 'Neighborhood').drop(columns= ['Neighborhood'])
df_manhattan_crime_neighbourhood2.head()
# Neighborhoods prices on Manhattan in 2019
nyc_lat = 40.758896
nyc_lon = -73.985130
nyc_official_neighbourhoods_price_map = folium.Map(location=[nyc_lat,nyc_lon], zoom_start=12)
nyc_official_neighbourhoods_price_map.choropleth(
    geo_data=newyork_data,
    data=neighbourhood_crime_price_merged3,
    columns=['tab_area', 'mean_price_per_person'],
    key_on='properties.ntaname',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Price/Neighbourhoods in New York 2019')

# Add Official Neighbourhoods names
for lat, lng, poi, mean_price_per_person, accommodates, mean_crimes  in zip(neighbourhood_crime_price_merged3['Latitude'],
                                                                         neighbourhood_crime_price_merged3['Longitude'], 
                                                                         neighbourhood_crime_price_merged3['tab_area'],
                                                                         round(neighbourhood_crime_price_merged3['mean_price_per_person'],2), 
                                                                         neighbourhood_crime_price_merged3['accommodates'],
                                                                         round(neighbourhood_crime_price_merged3['mean_crimes'],2)):
    label = folium.Popup(str(poi) +  '|| Mean Price per person: $' + str(mean_price_per_person)  +', Accommodates Number: ' + str(accommodates)  + ', Mean Crimes: ' + str(mean_crimes), parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(nyc_official_neighbourhoods_price_map)  
    
nyc_official_neighbourhoods_price_map
df_airnb_norm_data = df_airnb_norm.iloc[:].copy()
accomodations = folium.map.FeatureGroup()
for lat, lng, poi, rating, full_price, price_per_person, accommodates, crimes, listing_url, picture_url, bathrooms, bedrooms,neighbourhood_cleansed,square_feet in zip(df_airnb_norm_data['latitude'],
                                                                         df_airnb_norm_data['longitude'], 
                                                                         df_airnb_norm_data['name'],
                                                                         df_airnb_norm_data['review_scores_rating'],
                                                                         round(df_airnb_norm_data['full_price'],2), 
                                                                         round(df_airnb_norm_data['price_per_person'],2), 
                                                                         df_airnb_norm_data['accommodates'],
                                                                         round(df_airnb_norm_data['crimes'],2),
                                                                         df_airnb_norm_data['listing_url'],
                                                                         df_airnb_norm_data['picture_url'],
                                                                         df_airnb_norm_data['bathrooms'],
                                                                         df_airnb_norm_data['bedrooms'],
                                                                         df_airnb_norm_data['neighbourhood_cleansed'],
                                                                         df_airnb_norm_data['square_feet']
                                                                        ):
    html = f"""
     <br /> 
     <b>Accommodation: </b>{poi} <br />  
     <b>Host: </b><a href='{listing_url}'>{listing_url}</a>  <br />  
     <b>Neighbourhood: </b> {neighbourhood_cleansed}<br />  
     <b>Rating: </b> {rating}<br />  
     <b>Full price (USD): </b>{full_price}  <br />   
     <b>Price per Person (USD): </b>{price_per_person} <br /> 
     <b>Accommodates: </b>{accommodates}<br /> 
     <b>Bathrooms: </b>{bathrooms}<br /> 
     <b>Bedrooms: </b>{bedrooms}<br /> 
     <b>Square feet: </b>{square_feet}<br /> 
     <b>Crimes in 100 meters: </b>{crimes} <br />      
     <img ALIGN="Right" src="{picture_url}" alt="Host picture" width="300"  height="100">
    """
    #label = folium.Popup(str(poi) +  '|| Full price: $' + str(full_price) + ', Price per person: $' + str(price_per_person)  +', Accommodates: ' + str(accommodates)  + ', Crimes: ' + str(crimes), parse_html=True)
    iframe = folium.IFrame(html=html, width=500, height=350) 
    popup = folium.Popup(iframe, max_width=500)
    accomodations.add_child(
        folium.features.CircleMarker(
            [lat, lng],
            radius=1,
            popup=popup,
            color='yellow',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6
        )
    )     
nyc_official_neighbourhoods_price_map.add_child(accomodations)
nyc_crime_map = folium.Map(location=[nyc_lat,nyc_lon], zoom_start=11)
limit = 1000
df_incidents = df_manhattan_crime.sample(limit) 

# instantiate a mark cluster object for the incidents in the dataframe
incidents = plugins.MarkerCluster().add_to(nyc_crime_map)

# loop through the dataframe and add each data point to the mark cluster
for lat, lng, label, area in zip(df_incidents.Latitude, df_incidents.Longitude, df_incidents.LAW_CAT_CD, df_incidents.tab_area):
    folium.Marker(
        location=[lat, lng],
        icon=None,
        popup=label,
    ).add_to(incidents)

# display map
nyc_crime_map
# Neighborhoods Сrimes Summary on Manhattan in 2019
nyc_lat = 40.758896
nyc_lon = -73.985130
nyc_official_neighbourhoods_crime_map = folium.Map(location=[nyc_lat,nyc_lon], zoom_start=12)
nyc_official_neighbourhoods_crime_map.choropleth(
    geo_data=newyork_data,
    data=df_manhattan_crime_neighbourhood2,
    columns=['tab_area', 'crimes_summary'],
    key_on='properties.ntaname',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Crime/Neighbourhoods in New York 2019')

# Add Official Neighbourhoods names
for lat, lng, poi, crimes_summary  in zip(df_manhattan_crime_neighbourhood2['Latitude'],
                                                                         df_manhattan_crime_neighbourhood2['Longitude'], 
                                                                         df_manhattan_crime_neighbourhood2['tab_area'],
                                                                         df_manhattan_crime_neighbourhood2['crimes_summary']
                                                                        ):
    label = folium.Popup(str(poi) +  '|| Crimes Summary: ' + str(crimes_summary), parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='black',
        fill=True,
        fill_color='#8B0000',
        fill_opacity=0.7,
        parse_html=False).add_to(nyc_official_neighbourhoods_crime_map)  
    

nyc_official_neighbourhoods_crime_map
limit=100
df_airnb_top = df_airnb_norm.sort_values(by=['review_scores_rating',  'full_price','price_per_person',  'crimes'], ascending=[False, True, True, True])[['name', 'tab_area', 'neighbourhood_cleansed','latitude', 'longitude' ,'review_scores_rating',  'property_type', 'room_type', 'accommodates', 'full_price', 'price_per_person', 'crimes', 'listing_url','picture_url','bathrooms', 'bedrooms','square_feet']].head(limit)
df_airnb_top.head()
CLIENT_ID = '2ISEV1BQELTEGUC240PXI1ISJJQLEOOBDUGZMDOPOVSSGAO0' # your Foursquare ID
CLIENT_SECRET = 'QO0FTBIJLB3HJYE4IWYPNJU05ZEBRIAASLPL5II3PMF0JPUN' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version
LIMIT = 50
RADIUS = 1000

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)
def getAccomodatesNearbyVenues(names, latitudes, longitudes, radius, limit):
    venues_list=[]
    nearby_venues = pd.DataFrame()
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
            limit)
        try:
            response = requests.get(url)
            if response.status_code == 200: 
                results = response.json()["response"]['groups'][0]['items']
                venues_list.append([(
                    name, 
                    lat, 
                    lng, 
                    v['venue']['name'], 
                    v['venue']['location']['lat'], 
                    v['venue']['location']['lng'],  
                    v['venue']['categories'][0]['name']) for v in results])

                nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
                nearby_venues.columns = ['name', 
                                         'latitude', 
                                         'longitude', 
                                         'Venue', 
                                         'Venue Latitude', 
                                         'Venue Longitude', 
                                         'Venue Category']
            else:
                print ("Exception during call URL={} , Code={}, Text={}".format(url, response.status_code, response.text))
        except Exception as e:
            print ("Exception: {}".format(e))
            raise e
    
    return(nearby_venues)
df_airnb_top_venues = getAccomodatesNearbyVenues(names=df_airnb_top['name'],
                                   latitudes=df_airnb_top['latitude'],
                                   longitudes=df_airnb_top['longitude'],
                                   radius = RADIUS,
                                   limit = LIMIT
                                  )
df_airnb_top_venues.head()
df_airnb_top_venues.shape
fine_art_cat = ['Art','Arts','Museum', 'Library','Exhibit','Gallery']
eat_place_cat = ['Restaurant','Steakhouse']
shopping_cat = ['Shopping Mall','Market','Boutique']
outdoor_cat = ['Sculpture Garden','Scenic Lookout','Roof Deck','Outdoor Sculpture','Monument / Landmark',
               'Memorial Site','Lighthouse','Historic Site','Harbor / Marina','Fountain','Event Space','Bridge',
               'Waterfront','Church','Building','Garden','Historic Site','Lake','Park',
               'Pier','Rest Area','River','Synagogue','Field']
entertainment_cat = ['Nightclub','Circus','Club', 'Stadium', 'Karaoke Bar', 'Pub','Theater','Opera', 'Concert', 'Zoo']

#Join all categories' values in one
tourists_categories = fine_art_cat + eat_place_cat + shopping_cat + outdoor_cat +entertainment_cat
# We need Venues only from our Custom Categories
def check(category):
    if any(word in category for word in tourists_categories):
        return True
    return False
# Define Venue's Category
def change_categoty(name):
    if any(word in name for word in entertainment_cat):
        return 'Entertainment'
    if any(word in name for word in fine_art_cat):
        return 'Fine Art'
    if any(word in name for word in eat_place_cat):
        return 'Food Place'
    if any(word in name for word in shopping_cat):        
        return 'Shopping'
    if any(word in name for word in outdoor_cat):            
        return 'Sightseeing'
    if any(word in name for word in tansportation_cat):            
        return 'Transportation'    
# Define Top Venues
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)      
    return row_categories_sorted.index.values[0:num_top_venues] 

def return_most_common_venues_stats(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False).astype(float)      
    return row_categories_sorted.values[0:num_top_venues]
df_airnb_top_venues_f = df_airnb_top_venues[df_airnb_top_venues['Venue Category'].apply(lambda x: check(x))].copy()
df_airnb_top_venues_f.head()
df_airnb_top_venues_f.rename(columns = {'Venue Category':'Venue Type'}, inplace = True)

df_airnb_top_venues_f['Venue Category'] = df_airnb_top_venues_f['Venue Type'].apply(lambda x: change_categoty(x))
df_airnb_top_venues_f.head()
# one hot encoding
airnb_onehot = pd.get_dummies(df_airnb_top_venues_f[['Venue Category']], prefix="", prefix_sep="")

# add name column back to dataframe
airnb_onehot['name'] = df_airnb_top_venues_f['name'] 

# move name column to the first column
fixed_columns = [airnb_onehot.columns[-1]] + list(airnb_onehot.columns[:-1])
airnb_onehot = airnb_onehot[fixed_columns]
airnb_grouped = (np.round(airnb_onehot.groupby('name').mean(),2)).reset_index()
airnb_grouped.head()
# Find Top-3 Venues Category
num_top_venues = 3

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['name']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))
        
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue Share'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue Share'.format(ind+1))        
        

# create a new dataframe
airnb_venues_sorted = pd.DataFrame(columns=columns)
airnb_venues_sorted['name'] = airnb_grouped['name']

for ind in np.arange(airnb_grouped.shape[0]):
    strings_stats = np.around(return_most_common_venues_stats(airnb_grouped.iloc[ind, :], num_top_venues),2) 
    strings_category = return_most_common_venues(airnb_grouped.iloc[ind, :], num_top_venues)
    airnb_venues_sorted.iloc[ind, 1:] = np.concatenate([strings_category,strings_stats ])

airnb_venues_sorted.head()
# set number of clusters
kclusters = 3

airnb_grouped_clustering = airnb_grouped.drop('name', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(airnb_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 
# add clustering labels
airnb_venues_clustered = airnb_venues_sorted.copy()
airnb_venues_clustered.head()
airnb_venues_clustered.insert(0, 'Cluster Labels', kmeans.labels_)
airnb_venues_clustered.head()
airnb_merged = df_airnb_top.copy()
#  add latitude/longitude 
airnb_merged  = airnb_merged.join(airnb_venues_clustered.set_index('name'), on='name')
airnb_merged = airnb_merged.dropna()
airnb_merged.head()
df_airnb_norm.groupby(['tab_area','neighbourhood_cleansed']).size().reset_index().rename(columns={0:'Count', 'tab_area': 'NYC Tab Area', 'neighbourhood_cleansed': 'Airbnb Neighbourhood'}).sort_values('NYC Tab Area')
df_airnb_norm.groupby(['tab_area']).size().reset_index().rename(columns={0:'Count', 'tab_area': 'NYC Tab Area'}).sort_values('Count', ascending = False)
df_airnb_norm.groupby(['neighbourhood_cleansed']).size().reset_index().rename(columns={0:'Count', 'neighbourhood_cleansed': 'Airbnb Neighbourhood'}).sort_values('Count', ascending = False)
nyc_lat = 40.758896
nyc_lon = -73.985130
map_clusters = folium.Map(location=[nyc_lat,nyc_lon], zoom_start=12)
accomodations_cl = folium.map.FeatureGroup()

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

for lat, lng, poi, rating, full_price, price_per_person, accommodates, crimes, listing_url, picture_url, bathrooms, bedrooms,neighbourhood_cleansed,square_feet, cluster in zip(airnb_merged['latitude'],
                                                                         airnb_merged['longitude'], 
                                                                         airnb_merged['name'],
                                                                         airnb_merged['review_scores_rating'],
                                                                         round(airnb_merged['full_price'],2), 
                                                                         round(airnb_merged['price_per_person'],2), 
                                                                         airnb_merged['accommodates'],
                                                                         round(airnb_merged['crimes'],2),
                                                                         airnb_merged['listing_url'],
                                                                         airnb_merged['picture_url'],
                                                                         airnb_merged['bathrooms'],
                                                                         airnb_merged['bedrooms'],
                                                                         airnb_merged['neighbourhood_cleansed'],
                                                                         airnb_merged['square_feet'],
                                                                         airnb_merged['Cluster Labels'].astype(int)):
    html_cl = f"""
     <br /> 
     <b>Cluster: </b>{cluster} <br />  
     <b>Accommodation: </b>{poi} <br />  
     <b>Host: </b><a href='{listing_url}'>{listing_url}</a>  <br />  
     <b>Neighbourhood: </b> {neighbourhood_cleansed}<br />  
     <b>Rating: </b> {rating}<br />  
     <b>Full price (USD): </b>{full_price}  <br />   
     <b>Price per Person (USD): </b>{price_per_person} <br /> 
     <b>Accommodates: </b>{accommodates}<br /> 
     <b>Bathrooms: </b>{bathrooms}<br /> 
     <b>Bedrooms: </b>{bedrooms}<br /> 
     <b>Square feet: </b>{square_feet}<br /> 
     <b>Crimes in 100 meters: </b>{crimes} <br />      
     <img ALIGN="Right" src="{picture_url}" alt="Host picture" width="300"  height="100">
    """
    iframe_cl = folium.IFrame(html=html_cl, width=500, height=350) 
    popup = folium.Popup(iframe_cl, max_width=500)
    accomodations_cl.add_child(
        folium.features.CircleMarker(
            [lat, lng],
            radius=5,
            popup=popup,
            color=rainbow[cluster-1],
            fill=True,
            fill_color=rainbow[cluster-1],
            fill_opacity=0.7
        )
    )     
map_clusters.add_child(accomodations_cl)
cluster0 = airnb_merged.loc[airnb_merged['Cluster Labels'] == 0, ~airnb_merged.columns.isin(['listing_url', 'picture_url','latitude', 'longitude', 'square_feet'])]
cluster0.describe()
cluster0.head(10)
cluster1 = airnb_merged.loc[airnb_merged['Cluster Labels'] == 1, ~airnb_merged.columns.isin(['listing_url', 'picture_url','latitude', 'longitude', 'square_feet'])]
cluster1.describe()
cluster1.head(10)
cluster2 = airnb_merged.loc[airnb_merged['Cluster Labels'] == 2, ~airnb_merged.columns.isin(['listing_url', 'picture_url','latitude', 'longitude', 'square_feet'])]
cluster2.describe()
cluster2.head(10)
