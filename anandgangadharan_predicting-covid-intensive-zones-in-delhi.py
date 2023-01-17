from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import folium
url = 'https://en.wikipedia.org/wiki/Neighbourhoods_of_Delhi'
html_content = requests.get(url).text

soup = BeautifulSoup(html_content, "html.parser")
delhi_table = soup.find_all("span",attrs={"class":"mw-headline"})
districts = []
i=1;
for v in delhi_table:
    districts.append(v.text)
    i=i+1
    if(i==10):
        break
districts
address = 'Delhi, India'

geolocator = Nominatim(user_agent="del_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Delhi are {}, {}.'.format(latitude, longitude))
df_delhi = pd.read_excel('../input/delhi-neighborhoods-coordinates/Delhi_Neigh_LatLong.xlsx')
df_delhi.head()
df_delhi.shape
map_delhi = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, dist, neighborhood in zip(df_delhi['Latitude'], df_delhi['Longitude'], df_delhi['District'], df_delhi['Neighborhood']):
    label = '{}, {}'.format(neighborhood, dist)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_delhi)  
    
map_delhi
CLIENT_ID = 'Y3H35IKOA5URAE5CNY0CN5UACSA4BIVGWAPYFZ1TYQGOB435'
CLIENT_SECRET = 'F0LOPS1EHYFZPIC3I5OSQ2FMOVN0CHA5VIQ3SRDMLYHZQW1E'
VERSION = '20200511' 
LIMIT = 50
radius = 1000
# Function for getting venues by the neighborhood, latitiute, longitude and radius

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
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)
delhi_venues = getNearbyVenues(names=df_delhi['Neighborhood'],
                                   latitudes=df_delhi['Latitude'],
                                   longitudes=df_delhi['Longitude']
                                  )

delhi_venues.shape
delhi_venues.head()
delhi_venues.groupby('Neighborhood').count()
print('There are {} unique venue categories.'.format(len(delhi_venues['Venue Category'].unique())))
conflicting_value = delhi_venues[delhi_venues['Venue Category']=='Neighborhood']
conflicting_value
map_delhi_venues = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, ven, cat in zip(delhi_venues['Venue Latitude'], delhi_venues['Venue Longitude'], delhi_venues['Venue'], delhi_venues['Venue Category']):
    label = '{}, {}'.format(ven,cat)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='red',
        fill=True,
        fill_color='yellow',
        fill_opacity=0.7,
        parse_html=False).add_to(map_delhi_venues)  
    
map_delhi_venues
url = 'https://www.census2011.co.in/census/state/districtlist/delhi.html'
html_content = requests.get(url).text

soup = BeautifulSoup(html_content, "html.parser")
density_t = soup.find("table")
density_headings = density_t.find_all("th")
den_headings = []
for v in density_headings:
    den_headings.append(v.text)
delhi_table_data = density_t.find_all("tr")
table_data = []
for v in delhi_table_data:
    if(v!=delhi_table_data[0]):
        t_row = {}
        for td,h in zip(v.find_all("td"),den_headings):
            t_row[h] = td.text.replace('\n', '').strip()
        table_data.append(t_row)
den_delhi = pd.DataFrame(table_data)
den_delhi
den_delhi.dropna(inplace=True)
den_delhi = pd.DataFrame(den_delhi[['District','Density']])
den_delhi
new_districts = pd.DataFrame({"District":['South East Delhi','Shahdara'],"Density":[11060,27132]})
den_delhi = den_delhi.append(new_districts)
den_delhi.reset_index(inplace = True, drop = True) 
den_delhi
# one hot encoding
delhi_onehot = pd.get_dummies(delhi_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
delhi_onehot.insert(0, 'Neighborhoods',delhi_venues['Neighborhood'])

delhi_onehot.head()
delhi_grouped = delhi_onehot.groupby('Neighborhoods').mean().reset_index()
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]
num_top_venues = 2

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhoods']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhoods'] = delhi_grouped['Neighborhoods']

for ind in np.arange(delhi_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(delhi_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
delhi_grouped_clustering = delhi_grouped.drop('Neighborhoods', 1)
from sklearn.metrics import silhouette_score

sil = []
kmax = 10

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(3, kmax+1):
    kmeans = KMeans(n_clusters = k).fit(delhi_grouped_clustering)
    labels = kmeans.labels_
    sil.append(silhouette_score(delhi_grouped_clustering, labels, metric = 'euclidean'))
plt.plot(range(3, 11), sil, color ='g', linewidth ='3') 
plt.xlabel("Value of K") 
plt.ylabel("Silhouetee Score") 
plt.show() # clear the plot 
kclusters = 9

kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(delhi_grouped_clustering)

kmeans.labels_

neighborhoods_venues_sorted.drop(['Cluster Labels'], axis=1, inplace=True)

neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

kmeans.labels_[0:10] 
# Creating a dataframe with venues and respective cluster groups

# lets first change the name of the' Neighborhood' column in df_toronto to 'Neighborhoods'
df_delhi.rename(columns={'Neighborhood':'Neighborhoods'},inplace=True)

#Merging
delhi_merged = df_delhi
delhi_merged = delhi_merged.join(neighborhoods_venues_sorted.set_index('Neighborhoods'), on='Neighborhoods')

delhi_merged.head() 
delhi_merged.dropna(inplace=True)
delhi_merged.reset_index(inplace = True, drop = True)
delhi_merged['Cluster Labels'] = pd.to_numeric(delhi_merged['Cluster Labels'], downcast='integer')
delhi_merged.shape
delhi_merged.head()
import matplotlib.cm as cm
import matplotlib.colors as colors
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]
#rainbow = ['yellow','blue','purple','red','turquoise','green']

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(delhi_merged['Latitude'], delhi_merged['Longitude'], delhi_merged['Neighborhoods'], delhi_merged['Cluster Labels']):
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
delhi_merged.loc[delhi_merged['Cluster Labels'] == 1, delhi_merged.columns[[1] + list(range(5, delhi_merged.shape[1]))]]
delhi_merged.loc[delhi_merged['Cluster Labels'] == 3, delhi_merged.columns[[1] + list(range(5, delhi_merged.shape[1]))]]
delhi_merged.loc[delhi_merged['Cluster Labels'] == 7, delhi_merged.columns[[1] + list(range(5, delhi_merged.shape[1]))]]
#Loading my dataset with district polygon coordinates
delhi_geo = r'../input/delhi-district-boundaries/delhi_districts.json' # geojson file

# create a plain world map
delhi_dis_map = folium.Map(location=[latitude, longitude], zoom_start=10)
den_delhi['Density'] = pd.to_numeric(den_delhi['Density'], downcast='integer')
den_delhi.dtypes
threshold_scale = np.linspace(den_delhi['Density'].min(),
                              den_delhi['Density'].max(),
                              6, dtype=int)
threshold_scale = threshold_scale.tolist() # change the numpy array to a list
threshold_scale[-1] = threshold_scale[-1] + 1

delhi_dis_map.choropleth(
    geo_data=delhi_geo,
    data=den_delhi,
    columns=['District', 'Density'],
    key_on='feature.properties.name',
    fill_color='YlOrRd', 
    threshold_scale=threshold_scale,
    fill_opacity=0.55, 
    line_opacity=0.2,
    legend_name='Population Density'
)

# display map
delhi_dis_map
for lat, lon, poi, cluster in zip(delhi_merged['Latitude'], delhi_merged['Longitude'], delhi_merged['Neighborhoods'], delhi_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster],
        fill=True,
        fill_color=rainbow[cluster],
        fill_opacity=0.7).add_to(delhi_dis_map)
       
delhi_dis_map