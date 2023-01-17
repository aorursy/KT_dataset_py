#import standard libraries 

import pandas as pd

from pandas.io.json import json_normalize  

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np 

import seaborn as sns 

import math

import json

from math import radians, cos, sin, asin, sqrt



#import maps - remove this if you need to install gmaps 

#!pip install gmaps 

#import gmaps 



#import datetime libraries 

import datetime 

from datetime import date 

from datetime import time 

from datetime import timedelta
#read json file from google takeout

df = pd.read_json('Location History.json')





#create a dictionary to parse information from json file 

df_dict = {}



for i in range(10000, 30000): #note the range describes the number of json objects we wish to have in our dataset 

    

    

    time = int(df.iloc[i][0]['timestampMs'])

    daytime = datetime.fromtimestamp(time/1000.0)

    

    lat = int(df.iloc[i][0]['latitudeE7'])/10000000

    long = int(df.iloc[i][0]['longitudeE7'])/10000000

    

    print(daytime)

    df_dict[daytime] = [long, lat]

    

#transpose dataframe

df = pd.DataFrame.from_dict(df_dict, orient='columns').transpose()



#reset index 

df.reset_index(inplace=True)



#rename columns 

df.columns = ['Datetime', 'Long', 'Lat']



#set datatime column as a datetime formate 

df['Datetime'] = pd.to_datetime(df['Datetime'], format="%d/%m/%Y, %H:%M:%S")



#create a geo_tuple as a new column 

df['geo_tuple'] = list(zip(df['Lat'], df['Long']))



#convert to csv 

df.to_csv('location_history.csv')
#define coverage 



def pings_per_hour(df):

    """This function calculates the number of hours of the day that have a ping associated with them.

       It also returns the coverage rate as percentage of hours with at least 1 ping and the average number of pings per hous"""

    

    df['hour'] = df['Datetime'].dt.hour

    total_hrs = len(df['hour'].value_counts())

    coverage = total_hrs/24 

    pings_per_hour = df.shape[0]/total_hrs 

    return(total_hrs, coverage, pings_per_hour)



#sleeping geo-location 



def sleep_location(df):

    

    """This finds the most common location between the hours of 3am and 8am.

       If this has no value, most common location for the day is chosen"""

    

    date = str((df.iloc[0]['Datetime']).date())

    night_start = date + ' 03:00:00'

    night_end = date  + ' 08:00:00'

    datetime.datetime.strptime(night_start, '%Y-%m-%d %H:%M:%S') 

    datetime.datetime.strptime(night_end, '%Y-%m-%d %H:%M:%S') 

    

    sleeping_df = df[(df['Datetime'] > night_start) & (df['Datetime'] < night_end)]

    

    sleeping_geo_tuple = sleeping_df['geo_tuple'].mode()

    most_common_geo_tuple = df['geo_tuple'].mode()

    

    if not any(sleeping_geo_tuple):

        sleeping_geo_tuple = most_common_geo_tuple



        

    else: 

        sleeping_geo_tuple = sleeping_geo_tuple 

        



    

    return(sleeping_geo_tuple)



#google maps image showing locations for the day 



def gmaps_day(df, lat, long, api): 

    """Creates google map for dataframe. Lat and Long represent sleeping geo-locations and are centre point of map. 

       This requires a google api."""

    

    gmaps.configure(api_key=api)

    

    start_coordinates = (lat, long)

    fig = gmaps.figure(center=start_coordinates, zoom_level=12.5)

    

    marker_locations = list(zip(df['Lat'], df['Long']))



    markers = gmaps.marker_layer(marker_locations)



    fig.add_layer(markers)

    fig

    return(fig)



#exclusion zone - function that states if a geo-location is within a certain radius of another geo-location 



def exclusion_zone(row, radius, lat, long): 

    """returns home if within radius of lat and long

       returns away if outside given radius of lat and long. 

       employs haversine formula"""

    

    radius = radius 

    sleeping_geo_location = (lat, long)

    lat1 = sleeping_geo_location[0]

    lon1 = sleeping_geo_location[1]

    lat2 = row['Lat']

    lon2 = row['Long']

    

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    

    # haversine formula 

    dlon = lon2 - lon1 

    dlat = lat2 - lat1 

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2

    c = 2 * asin(sqrt(a)) 

    r = 6371 # Radius of earth in kilometers. Use 3956 for miles

    a = c*r



    if a <= radius:

        return('Home')

    else:

        return('Away')

    

#maximum distance from home function 



def max_distance_from_home(row, lat, long): 

    """returns max distance from home in km"""



    sleeping_geo_location = (lat, long)

    lat1 = sleeping_geo_location[0]

    lon1 = sleeping_geo_location[1]

    lat2 = row['Lat']

    lon2 = row['Long']

    

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    

    # haversine formula 

    dlon = lon2 - lon1 

    dlat = lat2 - lat1 

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2

    c = 2 * asin(sqrt(a)) 

    r = 6371 # Radius of earth in kilometers. Use 3956 for miles

    a = c*r



    return(a)



#area enclosed within co-ordinates 



def PolyArea(df):

    

    x=df['Long']

    y=df['Lat']

    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

#import data as csv files 

pre_covid = pd.read_csv('../input/covid-data2/pre_covid_paris2.csv')

post_covid = pd.read_csv('../input/covid-data2/paris_post_covid.csv')





#drop redundant columns 

pre_covid.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

post_covid.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1', 'Unnamed: 0.1.1.1.1'], axis=1, inplace=True)



#reformat datetime

pre_covid['Datetime'] = pd.to_datetime(pre_covid['Datetime'], format="%Y-%m-%d %H:%M:%S")

post_covid['Datetime'] = pd.to_datetime(post_covid['Datetime'], format="%Y-%m-%d %H:%M:%S")
pre_covid
post_covid
pre_covid.shape
post_covid.shape
pings_per_hour(pre_covid)

pings_per_hour(post_covid)
pre_covid['hour'] = pre_covid['Datetime'].dt.hour

pre_covid_coverage = pd.DataFrame(pre_covid['hour'].value_counts().sort_index(ascending=True)).reset_index()

pre_covid_coverage.columns = ['24 hour', 'pings']
post_covid['hour'] = post_covid['Datetime'].dt.hour

post_covid_coverage = pd.DataFrame(post_covid['hour'].value_counts().sort_index(ascending=True)).reset_index()

post_covid_coverage.columns = [' 24 hour', 'pings']
pre_covid_coverage
post_covid_coverage
sleep_location(pre_covid)
sleep_location(post_covid)
gmaps_day(pre_covid, 48.84954515968211, 2.3391025994419903)
gmaps_day(post_covid, 48.856599912662276, 2.3522000741992013)
#pre_covid

gmaps.configure(api_key='api') #insert own api here 

    

start_coordinates = (48.84954515968211, 2.3391025994419903) #start co-ordinates are for where the map should be centred



fig = gmaps.figure(center=start_coordinates, zoom_level=14)

    

# Adding a marker layer for home co-ordinates 

home = [(48.84954515968211, 2.3391025994419903)]

marker_layer = gmaps.marker_layer(home)

fig.add_layer(marker_layer)





# Adding a heatmap layer to the map

heatmap_layer = gmaps.heatmap_layer(pre_covid[['Lat', 'Long']], point_radius=10, dissipating=False)

fig.add_layer(heatmap_layer)



fig
#post-covid

gmaps.configure(api_key='api') #insert own api here 

    

start_coordinates = (48.856599912662276, 2.3522000741992013) #start co-ordinates are for where the map should be centred



fig = gmaps.figure(center=start_coordinates, zoom_level=14)

    

# Adding a marker layer for home co-ordinates 

home = [(48.856599912662276, 2.3522000741992013)]

marker_layer = gmaps.marker_layer(home)

fig.add_layer(marker_layer)





# Adding a heatmap layer to the map

heatmap_layer = gmaps.heatmap_layer(post_covid[['Lat', 'Long']], point_radius=10, dissipating=False)

fig.add_layer(heatmap_layer)



fig
pre_covid['Home_Away'] = pre_covid.apply(lambda row: exclusion_zone(row, 0.5, 48.84954515968211, 2.3391025994419903), axis=1)

post_covid['Home_Away'] = post_covid.apply(lambda row: exclusion_zone(row, 0.5, 48.856599912662276, 2.3522000741992013), axis=1)
pre_covid['Home_Away'].value_counts()
post_covid['Home_Away'].value_counts()
pre_covid['Home_Away'].value_counts().plot(kind='bar', title='Pre-covid')
post_covid['Home_Away'].value_counts().plot(kind='bar', title='Post-covid')
pre_covid['Distance_from_home'] = pre_covid.apply(lambda row: max_distance_from_home(row, 48.84954515968211, 2.3391025994419903), axis=1)

post_covid['Distance_from_home'] = post_covid.apply(lambda row: max_distance_from_home(row, 48.856599912662276, 2.3522000741992013), axis=1)
pre_covid['Distance_from_home'].describe()
post_covid['Distance_from_home'].describe()
sns.distplot(pre_covid['Distance_from_home'], kde=False).set_title('Pre-covid distance histogram');

sns.distplot(post_covid['Distance_from_home'], kde=False).set_title('Post-covid distance histogram');
#reformat the datetime to another column just detailing hour 

pre_covid['hour'] = pre_covid['Datetime'].dt.hour

post_covid['hour'] = post_covid['Datetime'].dt.hour
sns.scatterplot(pre_covid['Distance_from_home'], pre_covid['hour']).set_title('Pre-covid: Distance from home according to hour');
sns.scatterplot(post_covid['Distance_from_home'], post_covid['hour']).set_title('Post-covid: Distance from home according to hour');
#combine the two dataframes 



combined_df = pd.concat([pre_covid, post_covid])



#add another column according to whether the data is from pre_covid or post_covid times... 



def pre_or_post_covid(row): 

    

    if str(row['Datetime'].date()) == '2013-12-18': 

        return('pre_covid')

    elif str(row['Datetime'].date()) == '2013-12-11':

        return('post_covid')

    

combined_df['pre_or_post_covid'] = combined_df.apply(pre_or_post_covid, axis=1)
#check that this has done the task correctly - value counts are correct 

combined_df['pre_or_post_covid'].value_counts()
sns.scatterplot(x = 'Distance_from_home', y= 'hour', data=combined_df, hue='pre_or_post_covid').set_title('Distance from home by hour')
PolyArea(pre_covid)
PolyArea(post_covid)
#set time at zero 

time_at_home = datetime.timedelta(seconds=0)



#for loop to iterate through the dataframe 

for i in range(0, pre_covid.shape[0]): 

    

    row = i 

    next_row = i + 1 

    

    #if I am at the last row - skip 

    if row == (pre_covid.shape[0]-1): 

        pass 

    

    else:

        

        #if I am at home 

        if 'Home' in pre_covid.iloc[row]['Home_Away']: 

        

            #and my next row says that I am still at home by the next ping 

            if 'Home' in pre_covid.iloc[next_row]['Home_Away']: 

                

                #find the difference in times 

                duration = pre_covid.iloc[next_row]['Datetime'] - pre_covid.iloc[row]['Datetime']

                

                #add this time to the time_at_home timer

                time_at_home += duration

            

            else: 

                pass 

        
print('pre-covid time at home: ', time_at_home)
time_at_home = datetime.timedelta(seconds=0)



for i in range(0, post_covid.shape[0]): 

    

    row = i 

    next_row = i + 1 

    

    if row == (post_covid.shape[0]-1): 

        pass 

    

    else:

        

        if 'Home' in post_covid.iloc[row]['Home_Away']: 

        

            if 'Home' in post_covid.iloc[next_row]['Home_Away']: 

            

                duration = post_covid.iloc[next_row]['Datetime'] - post_covid.iloc[row]['Datetime']

            

                time_at_home += duration

            

            else: 

                pass 

        
print('post-covid time at home: ', time_at_home)
post_covid.groupby('Home_Away')['geo_tuple'].value_counts()
pre_covid.groupby('Home_Away')['geo_tuple'].value_counts()
sleep_location(post_covid)
def time_per_place_dataframe(df): 

    

    #create a dataframe to count pings per location for away 

    away_m = df['Home_Away'] == 'Away'

    away_df = df[away_m]

    away_df= pd.DataFrame(away_df['geo_tuple'].value_counts().reset_index())

    away_df.columns = ['geo_tuple', 'pings']

    return(away_df)

    

   



    



pings_df = time_per_place_dataframe(post_covid)
pings_df.head()
library_m = post_covid['geo_tuple'] == '(48.84413542128853, 2.3545557504317665)'



library_df = post_covid[library_m]



library_df
def duration_per_place(df): 

    """Returns the top 10 locations and the time spent at each location"""

    

    away_m = df['Home_Away'] == 'Away'

    away_df = df[away_m]

    

    pings_df= pd.DataFrame(away_df['geo_tuple'].value_counts().reset_index())

    pings_df.columns = ['geo_tuple', 'pings']



    durations = {}



    for i in range(0, 10):

    

        geo_tuple = pings_df.iloc[i]['geo_tuple']

    

        location_m = away_df['geo_tuple'] == geo_tuple 

        location_df = away_df[location_m]

    

        time = location_df.iloc[-1]['Datetime'] - location_df.iloc[0]['Datetime']

    

        durations[geo_tuple] = time

        

    return(durations)
post_covid_durations = duration_per_place(post_covid)

post_covid_durations_df = pd.DataFrame.from_dict(post_covid_durations, orient='index').reset_index()

post_covid_durations_df.columns = ['geo_tuple', 'time']
post_covid_durations_df
pre_covid_durations = duration_per_place(pre_covid)

pre_covid_durations_df = pd.DataFrame.from_dict(pre_covid_durations, orient='index').reset_index()

pre_covid_durations_df.columns = ['geo_tuple', 'time']
pre_covid_durations_df
#this function adds a new column to the dataframe, returning yes or no if I was within 0.2Km of the library

def library_zone(row, radius, lat, long): 

    """returns home if within radius of lat and long

       returns away if outside given radius of lat and long. 

       employs haversine formula"""

    

    radius = radius 

    sleeping_geo_location = (lat, long)

    lat1 = sleeping_geo_location[0]

    lon1 = sleeping_geo_location[1]

    lat2 = row['Lat']

    lon2 = row['Long']

    

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    

    # haversine formula 

    dlon = lon2 - lon1 

    dlat = lat2 - lat1 

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2

    c = 2 * asin(sqrt(a)) 

    r = 6371 # Radius of earth in kilometers. Use 3956 for miles

    a = c*r



    if a <= radius:

        return('library')

    else:

        return('away')
post_covid['library_YN'] = post_covid.apply(lambda row: library_zone(row, 0.2, 48.84413542128853, 2.3545557504317665), axis=1)
post_covid['library_YN'].value_counts()
time_at_library = datetime.timedelta(seconds=0)



for i in range(0, post_covid.shape[0]): 

    

    row = i 

    next_row = i + 1 

    

    if row == (post_covid.shape[0]-1): 

        pass 

    

    else:

        

        if post_covid.iloc[row]['geo_tuple'] == '(48.84413542128853, 2.3545557504317665)': 

        

            if post_covid.iloc[next_row]['geo_tuple'] == '(48.84413542128853, 2.3545557504317665)': 

                

                duration = post_covid.iloc[next_row]['Datetime'] - post_covid.iloc[row]['Datetime']

            

                time_at_library += duration

            

            else: 

                pass 

            



            
time_at_library