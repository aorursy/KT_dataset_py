#I want to make sure I get all the libraries in here that I will need.
import numpy as np 
import pandas as pd 

import requests 

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import types

from sklearn.cluster import KMeans

print('All Systems Go...')
!conda install -c conda-forge geopy --yes 
from geopy.geocoders import Nominatim

!conda install -c conda-forge folium=0.5.0 --yes
import folium
vbhoods = pd.read_csv("../input/hrdfdata/HRDF.csv")
vbhoods.head()

address = '310 Edwin Dr, Virginia Beach, VA 23462'

geolocator = Nominatim(user_agent="vb_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinates of Virginia Beach, VA are {}, {}.'.format(latitude, longitude))
# create map of Va Beach using latitude and longitude values
map_vb = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, neighborhood in zip(vbhoods['Lat'], vbhoods['Long'], vbhoods['Neighborhood']):
    label = '{}'.format(neighborhood)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=3,
        popup=label,
        color='#78609e',
        fill=False,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_vb)  
    
map_vb
CLIENT_ID = 'TVJWUE15PRCVHC0JWOGO0DQMGUJT11D0REZ04AJK1G2JZ3HY' # your Foursquare ID
CLIENT_SECRET = 'DQ2E1HBT4GPJVRIAURN04Q5VVAAVVDWSJNHYDPP4TJG5OKGI' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)
vbhoods.loc[1, 'Neighborhood']
neighborhood_latitude = vbhoods.loc[2, 'Lat'] # neighborhood latitude value
neighborhood_longitude = vbhoods.loc[2, 'Long'] # neighborhood longitude value

neighborhood_name = vbhoods.loc[2, 'Neighborhood'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))
LIMIT = 100 # limit of number of venues returned by Foursquare API
radius = 1000 # define radius

url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)
url  
#Here is where we will go ahead and make the pulls of data.  Lets make sure we are good here
results = requests.get(url).json()
results
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
from pandas.io.json import json_normalize
venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()
print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))
def getNearbyVenues(names, latitudes, longitudes, radius=1000):
    
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
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)
vb_venues = getNearbyVenues(names=vbhoods['Neighborhood'],
                                   latitudes=vbhoods['Lat'],
                                   longitudes=vbhoods['Long']
                                  )

vb_venues.shape
venue_hist=vb_venues.groupby('Neighborhood').count()
venue_hist.drop(['Neighborhood Latitude', 'Neighborhood Longitude', 'Venue Latitude', 'Venue Longitude','Venue Category'], axis=1, inplace=True)
venue_hist.head()
venue_hist['Venue'].plot(kind='barh', figsize=(10,35))

plt.title('Venues Returned by Neighborhood') # add a title to the histogram
plt.ylabel('Neighborhood') # add y-label
plt.xlabel('# of Venues') # add x-label
#plt.savefig('venues-neighborhoods.png')
print('There are {} uniques categories.'.format(len(vb_venues['Venue Category'].unique())))
# one hot encoding
vb_onehot = pd.get_dummies(vb_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
vb_onehot['Neighborhood'] = vb_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [vb_onehot.columns[-1]] + list(vb_onehot.columns[:-1])
vb_onehot = vb_onehot[fixed_columns]

vb_onehot.head()
vb_onehot.shape
vb_grouped = vb_onehot.groupby('Neighborhood').mean().reset_index()
vb_grouped
num_top_venues = 5

for hood in vb_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = vb_grouped[vb_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]
num_top_venues = 5

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
neighborhoods_venues_sorted['Neighborhood'] = vb_grouped['Neighborhood']

for ind in np.arange(vb_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(vb_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()
# Setting up Elbow Analysis
cost =[] 
for i in range(1, 15): 
	KM = KMeans(n_clusters = i, max_iter = 500) 
	KM.fit(vb_grouped_clustering) 
	
	# calculates squared error 
	# for the clustered points 
	cost.append(KM.inertia_)	 

# plot the cost against K values 
plt.plot(range(1, 15), cost, color ='#78609e', linewidth ='3') 
plt.xlabel("Value of K") 
plt.ylabel("Sqaured Error (Cost)") 
plt.show() # clear the plot 

# the point of the elbow is the 
# most optimal value for choosing k 
# set number of clusters
kclusters = 7

vb_grouped_clustering = vb_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(vb_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_
#Hide this code when running a new pass
#vb_merged.drop(['Cluster Labels'], axis=1, inplace=True)
#vb_merged.head()
# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

vb_merged = vbhoods

# merge to add latitude/longitude for each neighborhood
vb_merged = vb_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

vb_merged.head(50) # check the last rows!
vb_merged.sort_values(by="Cluster Labels")
print(vb_merged['Cluster Labels'].value_counts())
vb_null = vb_merged[vb_merged['Cluster Labels'].isnull()]
vb_null.shape
vb_merged.sort_values(by="Cluster Labels")

vb_merged.drop([23,40,41,44,45], inplace=True)
vb_merged.sort_values(by="Cluster Labels")
vb_merged.to_csv('DataforTableauk7.csv')
# create map
#folium.TileLayer('MapQuest Open Aerial').add_to(map_clusters)
map_clusters = folium.Map(location=[latitude, longitude], tiles='stamenterrain', zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(vb_merged['Lat'], vb_merged['Long'], vb_merged['Neighborhood'], vb_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=3,
        popup=label,
        color=rainbow[int(cluster)-1],
        fill=False,
        fill_color=rainbow[int(cluster)-1],
        fill_opacity=0.5).add_to(map_clusters)
       
map_clusters
vb_merged.sort_values(by="Month Over Month", ascending=False)
#We will create dfs for each cluster so that we can study them in detail.
clus0=vb_merged.loc[vb_merged['Cluster Labels']== 0.0]
clus0.head()
clus2=vb_merged.loc[vb_merged['Cluster Labels']== 2.0]
clus2.head()
clus1=vb_merged.loc[vb_merged['Cluster Labels']== 1.0]
clus1
clus3=vb_merged.loc[vb_merged['Cluster Labels']== 3.0]
clus3
clus4=vb_merged.loc[vb_merged['Cluster Labels']== 4.0]
clus4.head()
clus5=vb_merged.loc[vb_merged['Cluster Labels']== 5.0]
clus5.head()

clus6=vb_merged.loc[vb_merged['Cluster Labels']== 6.0]
clus6.head()
print("The Rec Areas have a ZHV of ",clus0["Zillow Home Value"].mean(skipna=True))
print("Eastern Euro Areas have a ZHV of ",clus1["Zillow Home Value"].mean(skipna=True))
print("Outlier has a ZHV of ",clus2["Zillow Home Value"].mean(skipna=True))
print("Strip Mall cluster has a ZHV of ",clus3["Zillow Home Value"].mean(skipna=True))
print("Beach Neighborhoods have a ZHV of ",clus4["Zillow Home Value"].mean(skipna=True))
print("Residential Areas has a ZHV of ",clus5["Zillow Home Value"].mean(skipna=True))
print("The Rec Areas have a Yearly Growth Rate of ",clus0["Year Over Year"].mean(skipna=True))
print("Eastern Euro Areas have a Yearly Growth Rate of ",clus1["Year Over Year"].mean(skipna=True))
print("Outlier has a Yearly Growth Rate of ",clus2["Year Over Year"].mean(skipna=True))
print("Strip Mall cluster has a Yearly Growth Rate of ",clus3["Year Over Year"].mean(skipna=True))
print("Beach Neighborhoods have Yearly Growth Rate of ",clus4["Year Over Year"].mean(skipna=True))
print("Residential Areas has a Yearly Growth Rate of ",clus5["Year Over Year"].mean(skipna=True))
data = {'Cluster Label': [0,1,3,4,5], 'Cluster Alias': ['Recreational Area', 'Eastern European Area', 'Strip Mall Area', 'Beach Areas', 'Residential Areas'], 'Mean Zillow Home Value':[444083,301675,219742,476675,246850]}
mean_ZHV=pd.DataFrame.from_dict(data)
mean_ZHV.head()
#mean_ZHV.drop(['Cluster Label'], axis=1, inplace=True)
mean_ZHV=mean_ZHV.set_index('Cluster Label')
mean_ZHV.head()
mean_ZHV['Mean Zillow Home Value'].plot(kind='bar', figsize=(8,10))

plt.title('Home Values by Cluster') # add a title to the histogram
plt.ylabel('Mean Home Value') # add y-label
plt.xlabel('Cluster') # add x-label
plt.savefig('Home Values By Cluster.png')
#Lets do the chart on the home value growth now.

data = {'Cluster Label': [0,1,3,4,5], 'Cluster Alias': ['Recreational Area', 'Eastern European Area', 'Strip Mall Area', 'Beach Areas', 'Residential Areas'], 'Year Over Year Growth (%)':[.0406,.0293,.0434,.0185,.0291]}
mean_growth=pd.DataFrame.from_dict(data)
mean_growth.head()
mean_growth['Year Over Year Growth (%)']=100*mean_growth['Year Over Year Growth (%)']
mean_growth.head()
mean_growth=mean_growth.set_index('Cluster Label')
mean_growth.head()
mean_growth['Year Over Year Growth (%)'].plot(kind='bar', color='green', figsize=(8,10))

plt.title('Growth Rate % by Cluster') # add a title to the histogram
plt.ylabel('Rate') # add y-label
plt.xlabel('Cluster') # add x-label
plt.savefig('Growth Rate By Cluster.png')
