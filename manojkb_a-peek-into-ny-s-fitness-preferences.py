# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from geopy.geocoders import Nominatim
import requests

#plotting libraries and modules
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import folium #maps and geospatial plots
from folium import plugins
from folium.plugins import MeasureControl
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
!wget -q -O 'newyork_data.json' https://cocl.us/new_york_dataset
print('Data downloaded!')
import json #to consume geolocation file in json format
with open('newyork_data.json') as json_data:
    newyork_data = json.load(json_data)
Neighborhood_data = newyork_data['features']
colnames = ['Borough', 'Neighborhood', 'Latitude', 'Longitude']
Neighborhood_df = pd.DataFrame(columns= colnames)
for data in Neighborhood_data:
    borough = neighborhood_name = data['properties']['borough']
    neighborhood_name = data['properties']['name']
    
    neighborhood_latlon = data['geometry']['coordinates']
    neighborhood_lat = neighborhood_latlon[1]
    neighborhood_lon = neighborhood_latlon[0]
    
    Neighborhood_df = Neighborhood_df.append({'Borough' : borough,
                            'Neighborhood' : neighborhood_name,
                          'Latitude' : neighborhood_lat,
                          'Longitude' : neighborhood_lon}, ignore_index = True)
print(Neighborhood_df.shape)
print('The dataframe has {} Boroughs and {} neighborhoods of New York City'
      .format(len(Neighborhood_df['Borough'].unique()),Neighborhood_df.shape[0]))
address = 'New York City, NY'

geolocator = Nominatim(user_agent='ny_explorer')
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('Geogrpahic location of New York is {}, {}'. format(latitude, longitude))
NY_map = folium.Map(location = [latitude,longitude], zoom_start = 10)
NY_map

for lat, lon, Neighborhood, Borough in zip(Neighborhood_df['Latitude'], Neighborhood_df['Longitude'],
                                          Neighborhood_df['Neighborhood'], Neighborhood_df['Borough']):
    label = '{}, {}'. format(Neighborhood, Borough)
    folium.CircleMarker(
        [lat, lon],
        popup = label,
        radius = 5,
        color = 'blue',
        fill = True,
        fill_color = 'yellow',
        fill_opacity = 0.7,
        parse_html = False).add_to(NY_map)

NY_map
def getNearbyVenues(access_token, version, LIMIT, names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        #print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?oauth_token={}&v={}&ll={},{}&radius={}&limit={}'.format(
            access_token, 
            version, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()['response']['groups'][0]['items']
        
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
token = 'RBPLNFEVFSX5N12KWCQY35ZZNPIDEE0Q5WWWSEFMJLICKMD3'
limit = 100
version = '20180301'
manhattan_data = Neighborhood_df[Neighborhood_df['Borough'] == 'Manhattan'].reset_index(drop=True)
manhattan_venues = getNearbyVenues(token,version,limit,manhattan_data['Neighborhood'],manhattan_data['Latitude'],manhattan_data['Longitude'])
manhattan_venues.shape
'''neighborhood_venues=getNearbyVenues(token,version,limit,Neighborhood_df['Neighborhood'],
                                    Neighborhood_df['Latitude'],Neighborhood_df['Longitude'])
neighborhood_venues.shape'''
manhattan_venues.to_csv('/kaggle/working/manhattan_venues.csv')
Neighborhood_df.loc[0:,'Borough'].unique()
Bronx_data = Neighborhood_df[Neighborhood_df.loc[0:,'Borough']=='Bronx']
Bronx_venues = getNearbyVenues(token,version,limit,Bronx_data['Neighborhood'],Bronx_data['Latitude'],Bronx_data['Longitude'])
Bronx_venues.shape
Queens_data = Neighborhood_df[Neighborhood_df.loc[0:,'Borough']=='Queens']
Queens_venues = getNearbyVenues(token,version,limit,Queens_data['Neighborhood'],Queens_data['Latitude'],Queens_data['Longitude'])
Queens_venues.shape
StatenIsland_data = Neighborhood_df[Neighborhood_df.loc[0:,'Borough']=='Staten Island']
StatenIsland_venues = getNearbyVenues(token,version,limit,StatenIsland_data['Neighborhood'],StatenIsland_data['Latitude'],StatenIsland_data['Longitude'])
StatenIsland_venues.shape
Brooklyn_data = Neighborhood_df[Neighborhood_df.loc[0:,'Borough']=='Brooklyn']
Brooklyn_venues = getNearbyVenues(token,version,limit,Brooklyn_data['Neighborhood'],Brooklyn_data['Latitude'],Brooklyn_data['Longitude'])
Brooklyn_venues.shape
Bronx_venues.to_csv('/kaggle/working/bronx_vnues.csv')
Brooklyn_venues.to_csv('/kaggle/working/brooklyn_venues.csv')
Queens_venues.to_csv('/kaggle/working/queens_venues.csv')
StatenIsland_venues.to_csv('/kaggle/working/statenIsland_venues.csv')
def filter_venues(df,string):
    df_filtered = pd.DataFrame(columns=df.columns)
    df_filtered = df[df['Venue Category']==string]
    return df_filtered
fitness_venues = ['Gym', 'Gym / Fitness Center', 'Climbing Gym', 'Gymnastics Gym', 'Boxing Gym', 'Pilates Studio', 'Martial Arts Dojo', 
                  'Physical Therapist', 'College Gym', 'Weight Loss Center', 'Cycle Studio', 'Yoga Studio', 'Tennis Stadium', 'Sports Club', 
                  'Athletics & Sports', 'Tennis Court', 'Golf Course', 'Volleyball Court', 'Mini Golf', 'Basketball Court', 'Soccer Field', 
                  'Baseball Field', 'Soccer Field', 'Baseball Stadium',  'Golf Course', 'Stadium', 'Squash Court', 'Hockey Field',  
                  'College Basketball Court', 'College Stadium', 'Pool', 'Gym Pool', 'Pool Hall', 'Bike Trail', 'Track','Trail', 
                  'Skate Park', 'Skating Rink', 'Surf Spot', 'Ski Area', 'Roller Rink', 'Park', 'Playground', 'Dance Studio', 'Bowling Alley',
                  'Indoor Play Area', 'Outdoors & Recreation', 'Rock Climbing Spot', 'Other Great Outdoors']
manhattan_fitness_venues = df_sample = pd.DataFrame(columns=manhattan_venues.columns)
for category in fitness_venues:
    manhattan_fitness_venues = manhattan_fitness_venues.append(filter_venues(manhattan_venues,category))

bronx_fitness_venues = df_sample = pd.DataFrame(columns=Bronx_venues.columns)
for category in fitness_venues:
    bronx_fitness_venues = bronx_fitness_venues.append(filter_venues(Bronx_venues,category))

queens_fitness_venues = df_sample = pd.DataFrame(columns=Queens_venues.columns)
for category in fitness_venues:
    queens_fitness_venues = queens_fitness_venues.append(filter_venues(Queens_venues,category))

brooklyn_fitness_venues = df_sample = pd.DataFrame(columns=Brooklyn_venues.columns)
for category in fitness_venues:
    brooklyn_fitness_venues = brooklyn_fitness_venues.append(filter_venues(Brooklyn_venues,category))

statenIsland_fitness_venues = df_sample = pd.DataFrame(columns=StatenIsland_venues.columns)
for category in fitness_venues:
    statenIsland_fitness_venues = statenIsland_fitness_venues.append(filter_venues(StatenIsland_venues,category))
print('Fitness venues: \n\t {} in Manhattan \n\t {} in Brooklyn \n\t {} in Bronx \n\t {} in Staten Island \n\t {} in Queens'.format(manhattan_fitness_venues.shape[0],
                                                                                                                                    brooklyn_fitness_venues.shape[0],
                                                                                                                                    bronx_fitness_venues.shape[0],
                                                                                                                                    statenIsland_fitness_venues.shape[0],
                                                                                                                                    queens_fitness_venues.shape[0]))
# one hot encoding
manhattan_onehot = pd.get_dummies(manhattan_fitness_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
manhattan_onehot['Neighborhood'] = manhattan_fitness_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [manhattan_onehot.columns[-1]] + list(manhattan_onehot.columns[:-1])
manhattan_onehot = manhattan_onehot[fixed_columns]

manhattan_grouped = manhattan_onehot.groupby('Neighborhood').mean().reset_index()
manhattan_grouped.head()
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]
def sort_Venues(df):
    num_top_venues = 3

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
    neighborhoods_venues_sorted['Neighborhood'] = df['Neighborhood']

    for ind in np.arange(manhattan_grouped.shape[0]):
        neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(df.iloc[ind, :], num_top_venues)

    return neighborhoods_venues_sorted
manhattan_preferences = sort_Venues(manhattan_grouped)
manhattan_preferences = manhattan_preferences.groupby('1st Most Common Venue').count()
manhattan_preferences.reset_index(inplace=True)
manhattan_preferences.loc[0:,['1st Most Common Venue','Neighborhood']].sort_values('Neighborhood', ascending=False)
# one hot encoding
brooklyn_onehot = pd.get_dummies(brooklyn_fitness_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
brooklyn_onehot['Neighborhood'] = brooklyn_fitness_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [brooklyn_onehot.columns[-1]] + list(brooklyn_onehot.columns[:-1])
brooklyn_onehot = brooklyn_onehot[fixed_columns]

brooklyn_grouped = brooklyn_onehot.groupby('Neighborhood').mean().reset_index()
brooklyn_grouped.head()
brooklyn_preferences = sort_Venues(brooklyn_grouped)
brooklyn_preferences = brooklyn_preferences.groupby('1st Most Common Venue').count()
brooklyn_preferences.reset_index(inplace=True)
brooklyn_preferences.loc[0:,['1st Most Common Venue','Neighborhood']].sort_values('Neighborhood', ascending=False)
# one hot encoding
bronx_onehot = pd.get_dummies(bronx_fitness_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
bronx_onehot['Neighborhood'] = bronx_fitness_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [bronx_onehot.columns[-1]] + list(bronx_onehot.columns[:-1])
bronx_onehot = bronx_onehot[fixed_columns]

bronx_grouped = bronx_onehot.groupby('Neighborhood').mean().reset_index()
bronx_grouped.head()
bronx_preferences = sort_Venues(bronx_grouped)
bronx_preferences = bronx_preferences.groupby('1st Most Common Venue').count()
bronx_preferences.reset_index(inplace=True)
bronx_preferences.loc[0:,['1st Most Common Venue','Neighborhood']].sort_values('Neighborhood', ascending=False)
# one hot encoding
statenIsland_onehot = pd.get_dummies(statenIsland_fitness_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
statenIsland_onehot['Neighborhood'] = statenIsland_fitness_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [statenIsland_onehot.columns[-1]] + list(statenIsland_onehot.columns[:-1])
statenIsland_onehot = statenIsland_onehot[fixed_columns]

statenIsland_grouped = statenIsland_onehot.groupby('Neighborhood').mean().reset_index()
statenIsland_grouped.head()
statenIsland_preferences = sort_Venues(statenIsland_grouped)
statenIsland_preferences = statenIsland_preferences.groupby('1st Most Common Venue').count()
statenIsland_preferences.reset_index(inplace=True)
statenIsland_preferences.loc[0:,['1st Most Common Venue','Neighborhood']].sort_values('Neighborhood', ascending=False)
# one hot encoding
queens_onehot = pd.get_dummies(queens_fitness_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
queens_onehot['Neighborhood'] = queens_fitness_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [queens_onehot.columns[-1]] + list(queens_onehot.columns[:-1])
queens_onehot = queens_onehot[fixed_columns]

queens_grouped = queens_onehot.groupby('Neighborhood').mean().reset_index()
queens_grouped.head()
queens_preferences = sort_Venues(queens_grouped)
queens_preferences = queens_preferences.groupby('1st Most Common Venue').count()
queens_preferences.reset_index(inplace=True)
queens_preferences.loc[0:,['1st Most Common Venue','Neighborhood']].sort_values('Neighborhood', ascending=False)
newyork_venues = [manhattan_fitness_venues,
                  bronx_fitness_venues,
                  brooklyn_fitness_venues,
                  statenIsland_fitness_venues,
                  queens_fitness_venues]
newYork_fitness_venues = pd.concat(newyork_venues,keys=['Manhattan','Bronx','Brooklyn','statenIsland','Queens'])
newYork_fitness_venues.reset_index(inplace=True)
newYork_fitness_venues.columns
newYork_fitness_venues.drop('level_1', axis=1, inplace=True)
newYork_fitness_venues.rename(columns={'level_0':'Borough'}, inplace=True)
newYork_fitness_venues
newYork_parks=newYork_fitness_venues[newYork_fitness_venues.loc[0:,'Venue Category']=='Park']
newYork_parks.shape
# create map of Manhattan using latitude and longitude values
newYork_parks['color'] = newYork_parks.loc[0:,'Borough'].apply(lambda Borough:'Orange' if Borough=='Manhattan'
                                                       else 'dark blue' if Borough=='Bronx'
                                                       else 'red' if Borough=='statenIsland'
                                                       else 'green' if Borough=='Queens'
                                                       else 'purple')
newYork_parks.head(2)
parks_count=newYork_parks.groupby(['Borough']).count()
parks_count.reset_index(inplace=True)
parks_count=parks_count.loc[0:,['Borough','Venue']]
parks_count.set_index('Borough', inplace=True)


ax=parks_count.sort_values('Venue',ascending=True).plot(kind='bar', figsize=(10,6),color='#98FB98')
ax.set_title("Parks in  New York's Borough", fontsize=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xlabel('')
ax.set_yticks([])

for p in ax.patches[0:]:
    h = p.get_height()
    x = p.get_x()+p.get_width()/2.
    #print(p,h,x)
    if h != 0:
        ax.annotate("%i" % p.get_height(), xy=(x,h), xytext=(0,4), rotation=0, 
                  textcoords="offset points", ha="center", va="bottom",size = 14)
        
ax.legend("")

plt.show()
ny_park_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

# add markers to map
for lat, lng, Neighborhood,venue,color in zip(newYork_parks['Venue Latitude'], newYork_parks['Venue Longitude'], newYork_parks['Neighborhood'],newYork_parks['Venue'], newYork_parks['color']):
    label='{}, {}'.format(venue,Neighborhood)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='grey',
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        parse_html=False).add_to(ny_park_map)  
    
ny_park_map
gym_cat = ['Gym', 'Gym / Fitness Center', 'Climbing Gym', 'Gymnastics Gym', 'Boxing Gym', 'Pilates Studio', 
       'Martial Arts Dojo', 'Physical Therapist', 'College Gym', 'Weight Loss Center', 'Cycle Studio']

sports_cat=['Tennis Stadium', 'Sports Club', 'Athletics & Sports', 'Tennis Court', 'Golf Course', 'Volleyball Court', 'Mini Golf', 
            'Basketball Court','Soccer Field', 'Baseball Field','Soccer Field', 'Baseball Stadium',  'Golf Course', 'Stadium', 
            'Squash Court', 'Hockey Field', 'College Basketball Court', 'College Stadium']

swim_cat=['Pool', 'Gym Pool', 'Pool Hall']

TT_cat=['Bike Trail', 'Track','Trail']

Others=['Skate Park', 'Skating Rink', 'Surf Spot', 'Ski Area', 'Roller Rink','Dance Studio', 'Bowling Alley','Indoor Play Area',
       'Outdoors & Recreation', 'Rock Climbing Spot', 'Other Great Outdoors']
for i,category in enumerate(newYork_fitness_venues['Venue Category']):   
    #print(i,category)
    if category in gym_cat:
            newYork_fitness_venues.loc[i,'Category']='gym/fitness center'
            #break
    elif category in ['Yoga Studio']:
        newYork_fitness_venues.loc[i,'Category']='Yoga Studios'
    elif category in sports_cat:
           # print(category)
            newYork_fitness_venues.loc[i,'Category']='Sports'
    elif category in swim_cat:
            newYork_fitness_venues.loc[i,'Category']='Pool'
    elif category in TT_cat:
        newYork_fitness_venues.loc[i,'Category']='Track/Trail'
    elif category in Others:
        newYork_fitness_venues.loc[i,'Category']='Other venues'
    else:
        newYork_fitness_venues.loc[i,'Category']='Parks/Playgrounds'
newYork_fitness_venues_wo_parks=newYork_fitness_venues[newYork_fitness_venues['Category'] != 'Parks/Playgrounds']
ax = newYork_fitness_venues_wo_parks['Category'].value_counts().plot(kind='bar', figsize=(10,5))
ax.set_title('Fitness Venues in New York (excluding parks)', fontsize=14)
ax.set_ylim([0,500])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.show()
#venues_per_borough=venues_per_borough.reset_index()
venues_per_borough=newYork_fitness_venues_wo_parks.pivot_table(index='Borough', columns='Category', values='Venue',aggfunc='count')

del(venues_per_borough.index.name)
columns=['gym/fitness center','Yoga Studios','Sports','Pool','Track/Trail','Other venues']

venues_per_borough=venues_per_borough[columns]

ordr=['Manhattan','Brooklyn','Queens','statenIsland','Bronx']
ax=venues_per_borough.sort_values('gym/fitness center', ascending=False).plot(kind='bar', figsize=(20,8),width=0.8, stacked=False)
ax.set_title('Fitness Venues spread across Boroughs', fontsize=16)
ax.set_yticks([])

for p in ax.patches[0:]:
    h = p.get_height()
    x = p.get_x()+p.get_width()/2.
    #print(p,h,x)
    if h != 0:
        ax.annotate("%i" % p.get_height(), xy=(x,h), xytext=(0,4), rotation=0, 
                  textcoords="offset points", ha="center", va="bottom",size = 14)
    else:
        ax.annotate("0", xy=(x,h), xytext=(0,4), rotation=0, 
                  textcoords="offset points", ha="center", va="bottom",size = 14)

ax.legend(fontsize=14)
spines=['top', 'left','right']
for spine in spines:
    ax.spines[spine].set_visible(False)
#newYork_fitness_venues_wo_parks
newYork_fitness_venues_wo_parks['color'] = newYork_fitness_venues_wo_parks.loc[0:,'Category'].apply(lambda Category:'Orange' if Category=='gym/fitness center'
                                                        else 'blue' if Category=='Yoga Studios'
                                                        else 'red' if Category=='Pool'
                                                        else 'green' if Category=='Track/Trail'
                                                        else 'purple' if Category=='Sports'
                                                        else 'yellow')
ny_fit_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

venues=plugins.MarkerCluster().add_to(ny_fit_map)
# add markers to map
for lat, lng, Neighborhood,venue,color in zip(newYork_fitness_venues_wo_parks['Venue Latitude'], newYork_fitness_venues_wo_parks['Venue Longitude'], 
                                              newYork_fitness_venues_wo_parks['Neighborhood'],newYork_fitness_venues_wo_parks['Venue'], newYork_fitness_venues_wo_parks['color']):
    label='{}, {}'.format(venue,Neighborhood)
    label = folium.Popup(label, parse_html=True)
    folium.Marker(
        [lat, lng],
        #radius=5,
        popup=label,
        #color='grey',
        #fill=True,
        #fill_color='yellow',
        #fill_opacity=0.7,
        parse_html=False).add_to(venues)  
    
ny_fit_map.add_child(MeasureControl())
ny_fit_map
manhattan_resto=manhattan_venues[manhattan_venues.loc[0:,'Venue Category'].str.contains('restaurant', case=False)]
manhattan_resto.shape
manhattan_fitness_per_neighborhood=manhattan_fitness_venues['Neighborhood'].value_counts().to_frame()
manhattan_resto_per_neighborhood=manhattan_resto['Neighborhood'].value_counts().to_frame()
manhattan_fitness_per_neighborhood.reset_index(inplace=True)
manhattan_fitness_per_neighborhood.rename(columns={'index':'Neighborhood', 'Neighborhood':'venue_count'}, inplace=True)
manhattan_fitness_per_neighborhood.head()
manhattan_resto_per_neighborhood.reset_index(inplace=True)
manhattan_resto_per_neighborhood.rename(columns={'index':'Neighborhood', 'Neighborhood':'venue_count'}, inplace=True)
manhattan_resto_per_neighborhood.head()
manhattan_ratio=pd.DataFrame(columns=['Neighborhood','fitnessVenue/restaurant'])
n=[]

for n1,val1 in zip(manhattan_resto_per_neighborhood['Neighborhood'],manhattan_resto_per_neighborhood['venue_count']):
    for n2, val2 in zip(manhattan_fitness_per_neighborhood['Neighborhood'],manhattan_fitness_per_neighborhood['venue_count']):
        if n1==n2:
            val=val2/val1
    n.append([n1,round(val,2)])
for i in range(1,len(n)):
    manhattan_ratio.loc[i,'Neighborhood']=n[i][0]
    manhattan_ratio.loc[i,'fitnessVenue/restaurant']=n[i][1]
ax = manhattan_ratio.set_index('Neighborhood').sort_values('fitnessVenue/restaurant',ascending=False).plot(kind='bar',figsize=(15,8),color='orange')
ax.set_title('Manhattan - Ratio of Fitness Venues per Restaurant')
ax.legend('')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.show()
bronx_resto=Bronx_venues[Bronx_venues.loc[0:,'Venue Category'].str.contains('restaurant', case=False)]

bronx_fitness_per_neighborhood=bronx_fitness_venues['Neighborhood'].value_counts().to_frame()
bronx_resto_per_neighborhood=bronx_resto['Neighborhood'].value_counts().to_frame()
bronx_fitness_per_neighborhood.reset_index(inplace=True)
bronx_fitness_per_neighborhood.rename(columns={'index':'Neighborhood', 'Neighborhood':'venue_count'}, inplace=True)

bronx_resto_per_neighborhood.reset_index(inplace=True)
bronx_resto_per_neighborhood.rename(columns={'index':'Neighborhood', 'Neighborhood':'venue_count'}, inplace=True)

bronx_ratio=pd.DataFrame(columns=['Neighborhood','fitnessVenue/restaurant'])
n=[]
for n1,val1 in zip(bronx_resto_per_neighborhood['Neighborhood'],bronx_resto_per_neighborhood['venue_count']):
    for n2, val2 in zip(bronx_fitness_per_neighborhood['Neighborhood'],bronx_fitness_per_neighborhood['venue_count']):
        if n1==n2:
            val=val2/val1
    n.append([n1,round(val,2)])
for i in range(1,len(n)):
    bronx_ratio.loc[i,'Neighborhood']=n[i][0]
    bronx_ratio.loc[i,'fitnessVenue/restaurant']=n[i][1]
ax = bronx_ratio.set_index('Neighborhood').sort_values('fitnessVenue/restaurant',ascending=False).plot(kind='bar',figsize=(18,8),color='blue')
ax.set_title('The Bronx - Ratio of Fitness Venues per Restaurant')
ax.legend('')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.show()
brooklyn_resto=Brooklyn_venues[Brooklyn_venues.loc[0:,'Venue Category'].str.contains('restaurant', case=False)]

brooklyn_fitness_per_neighborhood=brooklyn_fitness_venues['Neighborhood'].value_counts().to_frame()
brooklyn_resto_per_neighborhood=brooklyn_resto['Neighborhood'].value_counts().to_frame()
brooklyn_fitness_per_neighborhood.reset_index(inplace=True)
brooklyn_fitness_per_neighborhood.rename(columns={'index':'Neighborhood', 'Neighborhood':'venue_count'}, inplace=True)

brooklyn_resto_per_neighborhood.reset_index(inplace=True)
brooklyn_resto_per_neighborhood.rename(columns={'index':'Neighborhood', 'Neighborhood':'venue_count'}, inplace=True)

brooklyn_ratio=pd.DataFrame(columns=['Neighborhood','fitnessVenue/restaurant'])
n=[]
for n1,val1 in zip(brooklyn_resto_per_neighborhood['Neighborhood'],brooklyn_resto_per_neighborhood['venue_count']):
    for n2, val2 in zip(brooklyn_fitness_per_neighborhood['Neighborhood'],brooklyn_fitness_per_neighborhood['venue_count']):
        if n1==n2:
            val=val2/val1
    n.append([n1,round(val,2)])
for i in range(1,len(n)):
    brooklyn_ratio.loc[i,'Neighborhood']=n[i][0]
    brooklyn_ratio.loc[i,'fitnessVenue/restaurant']=n[i][1]
ax = brooklyn_ratio.set_index('Neighborhood').sort_values('fitnessVenue/restaurant',ascending=False).plot(kind='bar',figsize=(18,8),color='purple')
ax.set_title('Brooklyn - Ratio of Fitness Venues per Restaurant')
ax.legend('')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.show()
queens_resto=Queens_venues[Queens_venues.loc[0:,'Venue Category'].str.contains('restaurant', case=False)]

queens_fitness_per_neighborhood=queens_fitness_venues['Neighborhood'].value_counts().to_frame()
queens_resto_per_neighborhood=queens_resto['Neighborhood'].value_counts().to_frame()
queens_fitness_per_neighborhood.reset_index(inplace=True)
queens_fitness_per_neighborhood.rename(columns={'index':'Neighborhood', 'Neighborhood':'venue_count'}, inplace=True)

queens_resto_per_neighborhood.reset_index(inplace=True)
queens_resto_per_neighborhood.rename(columns={'index':'Neighborhood', 'Neighborhood':'venue_count'}, inplace=True)

queens_ratio=pd.DataFrame(columns=['Neighborhood','fitnessVenue/restaurant'])
n=[]
for n1,val1 in zip(queens_resto_per_neighborhood['Neighborhood'],queens_resto_per_neighborhood['venue_count']):
    for n2, val2 in zip(queens_fitness_per_neighborhood['Neighborhood'],queens_fitness_per_neighborhood['venue_count']):
        if n1==n2:
            val=val2/val1
    n.append([n1,round(val,2)])
for i in range(1,len(n)):
    queens_ratio.loc[i,'Neighborhood']=n[i][0]
    queens_ratio.loc[i,'fitnessVenue/restaurant']=n[i][1]
ax = queens_ratio.set_index('Neighborhood').sort_values('fitnessVenue/restaurant',ascending=False).plot(kind='bar',figsize=(18,8),color='green')
ax.set_title('Queens - Ratio of Fitness Venues per Restaurant')
ax.legend('')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.show()
statenIsland_resto=StatenIsland_venues[StatenIsland_venues.loc[0:,'Venue Category'].str.contains('restaurant', case=False)]

statenIsland_fitness_per_neighborhood=statenIsland_fitness_venues['Neighborhood'].value_counts().to_frame()
statenIsland_resto_per_neighborhood=statenIsland_resto['Neighborhood'].value_counts().to_frame()
statenIsland_fitness_per_neighborhood.reset_index(inplace=True)
statenIsland_fitness_per_neighborhood.rename(columns={'index':'Neighborhood', 'Neighborhood':'venue_count'}, inplace=True)

statenIsland_resto_per_neighborhood.reset_index(inplace=True)
statenIsland_resto_per_neighborhood.rename(columns={'index':'Neighborhood', 'Neighborhood':'venue_count'}, inplace=True)

statenIsland_ratio=pd.DataFrame(columns=['Neighborhood','fitnessVenue/restaurant'])
n=[]
for n1,val1 in zip(statenIsland_resto_per_neighborhood['Neighborhood'],statenIsland_resto_per_neighborhood['venue_count']):
    for n2, val2 in zip(statenIsland_fitness_per_neighborhood['Neighborhood'],statenIsland_fitness_per_neighborhood['venue_count']):
        if n1==n2:
            val=val2/val1
    n.append([n1,round(val,2)])
for i in range(1,len(n)):
    statenIsland_ratio.loc[i,'Neighborhood']=n[i][0]
    statenIsland_ratio.loc[i,'fitnessVenue/restaurant']=n[i][1]
ax = statenIsland_ratio.set_index('Neighborhood').sort_values('fitnessVenue/restaurant',ascending=False).plot(kind='bar',figsize=(18,8),color='red')
ax.set_title('Staten Island - Ratio of Fitness Venues per Restaurant')
ax.legend('')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.show()
# one hot encoding
newyork_onehot = pd.get_dummies(newYork_fitness_venues_wo_parks[['Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
newyork_onehot['Neighborhood'] = newYork_fitness_venues_wo_parks['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [newyork_onehot.columns[-1]] + list(newyork_onehot.columns[:-1])
newyork_onehot = newyork_onehot[fixed_columns]
newyork_grouped = newyork_onehot.groupby('Neighborhood').mean().reset_index()
#newyork_grouped
num_top_venues = 3

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
newyork_fitness_venues_sorted = pd.DataFrame(columns=columns)
newyork_fitness_venues_sorted['Neighborhood'] = newyork_grouped['Neighborhood']

for ind in np.arange(newyork_grouped.shape[0]):
    newyork_fitness_venues_sorted.iloc[ind, 1:] = return_most_common_venues(newyork_grouped.iloc[ind, :], num_top_venues)

#newyork_fitness_venues_sorted.head()
from sklearn.cluster import KMeans
# set number of clusters
kclusters = 4

newyork_grouped_clustering = newyork_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(newyork_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 
# add clustering labels
newyork_fitness_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

newyork_merged = newYork_fitness_venues_wo_parks

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
newyork_merged = newyork_merged.join(newyork_fitness_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

#newyork_merged.head()
import matplotlib.cm as cm
import matplotlib.colors as colors
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(newyork_merged['Venue Latitude'], newyork_merged['Venue Longitude'], newyork_merged['Neighborhood'], newyork_merged['Cluster Labels']):
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
newyork_merged.loc[newyork_merged['Cluster Labels'] == 0, newyork_merged.columns[[1] + list(range(5, newyork_merged.shape[1]))]]
newyork_merged.loc[newyork_merged['Cluster Labels'] == 1, newyork_merged.columns[[1] + list(range(5, newyork_merged.shape[1]))]]
newyork_merged.loc[newyork_merged['Cluster Labels'] == 2, newyork_merged.columns[[1] + list(range(5, newyork_merged.shape[1]))]]
newyork_merged.loc[newyork_merged['Cluster Labels'] == 3, newyork_merged.columns[[1] + list(range(5, newyork_merged.shape[1]))]]