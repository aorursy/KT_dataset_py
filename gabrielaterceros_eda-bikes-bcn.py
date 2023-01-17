import requests
import json
import pandas as pd
import numpy as np
import dateutil.parser
import datetime
import math
import psycopg2
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
import folium
import geopy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import sys
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df_stations = pd.read_csv('../input/train_file.csv')
# EDA
#Let's what we have..

df_stations.head(10)
#Okay, some statistics for our quantitative data
df_stations.describe()
df_stations.shape
#FEATURE ENGENEERING
#We are goint to add some information to help us to undersatnd better the data
#Format the datetime
#Add day of week to see if there is any difference between weekday and weekend

def dateFormats(data):
    ## convert the timestamp to a python date object and get the hour
    data['timestamp'] = list(map(lambda v : dateutil.parser.isoparse(str(v)), data['timestamp']))
    
    ## add dedicated columns for day, hour and weekday 
    data['day'] = list(map(lambda v : v.day, data['timestamp']))
    data['hour'] = list(map(lambda v : v.hour, data['timestamp']))
    data['minute'] = list(map(lambda v : v.minute, data['timestamp']))
    data['dayofweek'] = data['timestamp'].dt.weekday
    
    #Let's group the minutes each 5 to have a better distrubution
    data['minute_c5'] = list(map(lambda v : round(v/5)*5 , data['minute']))
  
    return data


df_stations= dateFormats(df_stations)
#Getting the postal code, I'm using geopy
#Lets get the coordinates apart, because it last some time...

geoLocs = pd.DataFrame(df_stations[['latitude','longitude']].copy())
geoLocs.sort_values('latitude', inplace=True) 
geoLocs.drop_duplicates(keep='first' ,inplace=True, ignore_index= True) 

def get_zipcode(df, geolocator, lat_field, lon_field):
    try:
        location = geolocator.reverse((df[lat_field], df[lon_field]))
        return location.raw['address']['postcode']
    except :
        return [-1][-1]
    

geolocator = geopy.Nominatim(user_agent='my-application')


zipcodes = geoLocs.apply(get_zipcode, axis=1, geolocator=geolocator, lat_field='latitude', lon_field='longitude')

#Joining the postalcode to the dataset
geoLocs['postcode'] = zipcodes[0:]
df_stations = pd.merge(df_stations , geoLocs, on=['latitude','longitude'], how='inner')

print(df_stations)
print(df_stations.head(10))
#Let's check the quantitative varibles
f = pd.melt(df_stations, value_vars=df_stations[['empty_slots','free_bikes','day','hour','minute','dayofweek']])
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")
#Qualitative variables time, let's see how do they behave with the free bikes
for c in df_stations[['day','hour','dayofweek','postcode']]:
    df_stations[c] = df_stations[c].astype('category')
    if df_stations[c].isnull().any():
        df_stations[c] = df_stations[c].cat.add_categories(['MISSING'])
        df_stations[c] = df_stations[c].fillna('MISSING')

def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)
f = pd.melt(df_stations, id_vars=['free_bikes'], value_vars=df_stations[['day','hour','dayofweek','postcode']])
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, height=10)
g = g.map(boxplot, "value", "free_bikes")
#Changing the ID variable from categorical to numeric

try:
    df_stations['id'] = df_stations['id'].astype('category')
    if(df_stations['id_dwh']):
        pass
except:    
    df_stations['id_dwh'] = df_stations['id'].cat.codes
#preparing training and testing data

#Creating the dependents varibles

#Setting dependient variable:  free_bike, change to  0,1 (0: no free bike, 1:At least one bike)  
#Setting dependient variable:  empty_slot, change to  0,1 (0: no empty slot, 1:At least one empty slot) 

df_stations['is_free_bike'] = [1 if x>0 else 0 for x in df_stations.free_bikes]
df_stations['is_free_slot'] = [1 if x>0 else 0 for x in df_stations.empty_slots]

df_bikes_X = df_stations.drop(['name','postcode','day','timestamp','id','minute','free_bikes','empty_slots', 'is_free_bike', 'is_free_slot'], axis = 1)
df_bikes_y = df_stations['is_free_bike']


df_slot_X = df_stations.drop(['name','postcode','day','timestamp','id','minute','free_bikes','empty_slots','is_free_bike', 'is_free_slot'], axis = 1)
df_slot_y = df_stations['is_free_slot']
#Building the regression model to free bikes
#Building the regression model to free slots

#Trainig the model for the slots
X_train_bk, X_test_bk, y_train_bk, y_test_bk = model_selection.train_test_split(df_bikes_X, df_bikes_y ,test_size=0.2)
pred_bikes = []

model_bk = LinearRegression()
model_bk.fit(X_train_bk, y_train_bk)
pred_bikes = model_bk.predict(X_test_bk)
cnf_matrix_bikes = confusion_matrix(y_test_bk,pred_bikes.round())

#confusion matrix to see the how efficent is the model
print(X_train_bk.head)

def predicting_bikes(X):
    model_bk = LinearRegression()
    model_bk.fit(X_train_bk, y_train_bk)
    y = model_bk.predict(X) 
    return y

#Trainig the model for the slots
X_train_sl, X_test_sl, y_train_sl, y_test_sl = model_selection.train_test_split(df_slot_X, df_slot_y ,test_size=0.2)
pred_slot = []
model_sl = LinearRegression()
model_sl.fit(X_train_sl, y_train_sl)
pred_slot = model_sl.predict(X_test_sl)
cnf_matrix_slot =confusion_matrix(y_test_sl,pred_slot.round())
print(cnf_matrix_slot)

def predicting_slot(X):
    model_sl = LinearRegression()
    model_sl.fit(X_train_sl, y_train_sl)
    y = model_sl.predict(X)
    return y
#Now we want to know if there are free bikes and free slots given 2 points, date and time
#Let's see how our regression model predict the availability of the bikes
#I've added a field day_of_week to help the model to do a better prediction
#day_of_week: {0:'monday', 1:'thuesday', 2:'Wensday' , 3:'thursday', 4:'friday', 5:'saturday', 6:'sunday'}
def calculate_avalilability_pred(hour, minute, time_of_use_min, day_of_week, point_A, point_B):
    #Let's search the ID of the station A
    id_A = df_stations[df_stations['name'] == point_A] 
    id_dwh_A = df_stations[df_stations['id'] == id_A['id'].iloc[0]]
    
    #Let's round the minutes, for a better prediction we are grouping the data each 5 minutes
    minute = round(minute/5)*5
    
    #Preparing the data to predict the bikes
    X_bike = pd.DataFrame({'latitude':[id_dwh_A.latitude.max()],'longitude':[id_dwh_A.longitude.max()], 'id_dwh':[id_dwh_A.id_dwh.max()],'hour':[hour],'minute':[minute], 'dayofweek':[day_of_week]})    
    
    y_bike = predicting_bikes(X_bike)
    
    #Let's search the ID of the station B
    id_B = df_stations[df_stations['name'] == point_B]  
    id_dwh_B = df_stations[df_stations['id'] == id_B['id'].iloc[0]]    
    
    #Calculating the ETA (estimate time of arrival) to predict the slots
    min_B = minute+time_of_use_min
    if min_B >59:
        hour+=1
        min_B= 60-min_B
        
    X_slot = pd.DataFrame({'latitude':[id_dwh_B.latitude.max()],'longitude':[id_dwh_B.longitude.max()], 'id_dwh':[id_dwh_B.id_dwh.max()],'hour':[hour],'minute':[min_B], 'dayofweek':[day_of_week]})
    y_slot = predicting_slot(X_slot)
    
    print('We will find a free bike?: ' + 'Yes' if (y_bike>=1) else 'No' )

    print('We will find a free slot?: '+ 'Yes' if (y_slot>=1) else 'No' )
    
calculate_avalilability_pred(10,51,15,2,'C/ DOCTOR AIGUADER, 2', 'C/ SARDENYA, 292')
#Building searching functions to find the stations witn a distance x closer from a target (point_B)

#Grouping the geoloc info and the name
geoLocs = df_stations.groupby(['latitude','longitude','name'], as_index= False)['id'].count()

#Calculating the distance between the point_A and point_B
#We are going to use this distance such a parameter to find the others station that are arround the same distance
def get_distance(point_A, point_B):
    id_A = df_stations[df_stations['name'] == point_A] 
    latA = id_A['latitude'].iloc[0]
    lonA = id_A['longitude'].iloc[0]
    
    id_B = df_stations[df_stations['name'] == point_B] 
    latB = id_B['latitude'].iloc[0]
    lonB = id_B['longitude'].iloc[0]
    
    #dist_A = abs(latA) - abs(latB)
    #dist_B = abs(lonA) - abs(lonB)
    dist = distance(latA,latB,lonA,lonB)
    
    return dist #(abs(dist_A), abs(dist_B))

#Having the distance and the target, let's find the station arround.
def get_geoLoc_target(point_B, dist):
    id_B = df_stations[df_stations['name'] == point_B] 
    latB = id_B['latitude'].iloc[0]
    lonB = id_B['longitude'].iloc[0]
    
    geoLocs['distance']= list(map(lambda v: distance(v[0],latB,v[1],lonB), zip(geoLocs['latitude'], geoLocs['longitude'])))

    res = geoLocs[geoLocs['distance']<= dist]
    
    return res

def distance(x2,x1,y1,y2):
    dist = math.sqrt( (abs(x2) - abs(x1))**2 + (abs(y2) - abs(y1))**2 )
    return dist

def get_stations_arround(point_A, point_B, hour):
    dist = get_distance(point_A, point_B)
    res = get_geoLoc_target(point_B, dist)
    
    #lets list all the station that have free bikes in a time period (hr)
    df= df_stations[df_stations['hour']== 9]
    joined = pd.merge(df, res[['name', 'distance']], on=['name'], how='left')
    return joined



#Listing the station 
#This is the destination
res_st = get_stations_arround('C/ GOMBAU, 24', 'C/ CASANOVAS, 67', 8)
res_st = res_st.groupby(['name','latitude','longitude', 'distance'])['free_bikes'].mean().reset_index()

# This are the 10 closer station from our destination with free bikes
# We can see that the first values is the origin (distance 0)
print(res_st.sort_values(['distance']).head(10))

# This are the 10 farest station fron our destination with free bikes
print(res_st.sort_values(['distance'], ascending=False).head(10))


#Building the map to visualize the stations
#showing the closest 5
#res = res_st.sort_values(['distance']).head(55)

#showing the farest 5
res = res_st.sort_values(['distance'], ascending=False).head(60)


df = res_st[res_st['distance'] == 0]

world_map = folium.Map(location=[df.iloc[0]['latitude'], df.iloc[0]['longitude']], tiles="cartodbpositron", zoom_start=13,max_zoom=20,min_zoom=2)
for i in range(0,len(res)):
    folium.Circle(
        location=[res.iloc[i]['latitude'], res.iloc[i]['longitude']],
        tooltip = "<h5 style='text-align:center;font-weight: bold'>"+res.iloc[i]['name']+"</h5>"+
                    "<hr style='margin:10px;'>"+
                    "<ul style='color: #444;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+
        "<li>Free bikes: "+str(round(res.iloc[i]['free_bikes']))+"</li>"+
        "</ul>"
        ,
        radius=(int((np.log(res.iloc[i]['free_bikes']+1.00001)))+0.2)*20,
        #radius=(res.iloc[i]['free_bikes']),
        color='#ff6600',
        fill_color='#ff8533',
        fill=True).add_to(world_map)
    

folium.Circle(
location=[df.iloc[0]['latitude'], df.iloc[0]['longitude']],
tooltip = "<h5 style='text-align:center;font-weight: bold'>"+df.iloc[0]['name']+"</h5>"+
                    "<hr style='margin:10px;'>",
radius=(int((np.log(res.iloc[i]['free_bikes']+1.00001)))+0.2)*25,
color='blue',
fill_color='blue',
fill=True).add_to(world_map)

world_map
