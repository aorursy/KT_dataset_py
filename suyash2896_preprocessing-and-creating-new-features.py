# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from math import sin, cos, sqrt, atan2, radians
from sklearn.cluster import DBSCAN,KMeans
import random as rd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
# Read first 2500000 rows of train file
df=pd.read_csv('../input/train.csv',nrows=5000000)
df=df[df['dropoff_latitude'].notnull()]
#del df['Unnamed: 0']
df.head()
# Read test file for rest of the preprocessing
dftest=pd.read_csv('../input/test.csv')
dftest.head()
# Split pickup_datetime into date and time
df['Date']=df.pickup_datetime.str.slice(start=0,stop=10)
df['Time']=df.pickup_datetime.str.slice(start=11,stop=19)
df.head()

df['Date']=pd.to_datetime(arg=df.Date)
df['month']=pd.DatetimeIndex(df.Date).month
df['year']=pd.DatetimeIndex(df.Date).year
df['weekday']=pd.DatetimeIndex(df.Date).weekday
df['hour']=pd.DatetimeIndex(df.key).hour
df['week_of_month']=pd.DatetimeIndex(df.key).day%7
df.head()
# For test file
# Split pickup_datetime into date and time
dftest['Date']=dftest.pickup_datetime.str.slice(start=0,stop=10)
dftest['Time']=dftest.pickup_datetime.str.slice(start=11,stop=19)
dftest.head()

dftest['Date']=pd.to_datetime(arg=dftest.Date)
dftest['month']=pd.DatetimeIndex(dftest.Date).month
dftest['year']=pd.DatetimeIndex(dftest.Date).year
dftest['weekday']=pd.DatetimeIndex(dftest.Date).weekday
dftest['hour']=pd.DatetimeIndex(dftest.key).hour
dftest['week_of_month']=pd.DatetimeIndex(dftest.key).day%7
dftest.head()
def preprocess_hour(val):
    if val>=17:
        return 1
    elif val>7 and val<17:
        return 2
    elif val==0:
        return 3
    else:
        return 4

df['processed_hour_num_of_records']=df['hour'].apply(preprocess_hour)
dftest['processed_hour_num_of_records']=dftest['hour'].apply(preprocess_hour)
dftest.head()
di=df[['hour','fare_amount']].groupby([ "hour"]).median().to_dict()
def process_hour_fare(val):
    return di['fare_amount'][val]

df['processed_hour_fare']=df['hour'].apply(process_hour_fare)
dftest['processed_hour_fare']=dftest['hour'].apply(process_hour_fare)
dftest.head()
df.head()
def distance(coord1lat, coord1lon, coord2lat, coord2lon):
    R = 6373.0
    lat1 = radians(float(coord1lat))
    lon1 = radians(float(coord1lon))
    lat2 = radians(float(coord2lat))
    lon2 = radians(float(coord2lon))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance

def manhattan(coord1lat, coord1lon, coord2lat, coord2lon):
    d1=distance(coord1lat, coord2lon, coord2lat, coord2lon)
    d2=distance(coord1lat, coord1lon, coord1lat, coord2lon)
    
    return d1+d2
df['Distance']=df[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']].apply(lambda x: distance(x[0], x[1], x[2], x[3]), axis=1)
df['Manhattan_Distance']=df[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']].apply(lambda x: manhattan(x[0], x[1], x[2], x[3]), axis=1)
dftest['Distance']=dftest[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']].apply(lambda x: distance(x[0], x[1], x[2], x[3]), axis=1)
dftest['Manhattan_Distance']=dftest[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']].apply(lambda x: manhattan(x[0], x[1], x[2], x[3]), axis=1)

max(dftest['Manhattan_Distance'])
df.head()
# Clustering using DBSCAN
#clus=DBSCAN(eps=0.0002353, min_samples=4,algorithm='ball_tree',metric='haversine')
clus=KMeans(n_clusters=25)
df['Pickup_clusters']=clus.fit_predict(df[['pickup_latitude','pickup_longitude']].values)
dftest['Pickup_clusters']=clus.predict(dftest[['pickup_latitude','pickup_longitude']].values)
# df.plot.scatter(y='pickup_latitude',x='pickup_longitude')
# plt.show()
dftest.loc[dftest['Distance']==0,:]
#clus=DBSCAN(eps=0.0002353, min_samples=4,algorithm='ball_tree',metric='haversine')
clus=KMeans(n_clusters=25)
df['Dropoff_clusters']=clus.fit_predict(df[['dropoff_latitude','dropoff_longitude']].values)
dftest['Dropoff_clusters']=clus.predict(dftest[['dropoff_latitude','dropoff_longitude']].values)
df.head()
di=df[['Pickup_clusters','fare_amount']].groupby(["Pickup_clusters"]).median().to_dict()
def process_pickup_fare(val):
    return di['fare_amount'][val]

df['processed_pickup_fare']=df['Pickup_clusters'].apply(process_pickup_fare)
dftest['processed_pickup_fare']=dftest['Pickup_clusters'].apply(process_pickup_fare)
di=df[['Dropoff_clusters','fare_amount']].groupby(["Dropoff_clusters"]).median().to_dict()
def process_dropoff_fare(val):
    return di['fare_amount'][val]

df['processed_dropoff_fare']=df['Dropoff_clusters'].apply(process_dropoff_fare)
dftest['processed_dropoff_fare']=dftest['Dropoff_clusters'].apply(process_dropoff_fare)

di=df[['Pickup_clusters','Dropoff_clusters','fare_amount']].groupby(['Pickup_clusters',"Dropoff_clusters"]).median().to_dict()
def process_route_fare(val):
    return di['fare_amount'][val]

df['processed_route_fare']=df[['Pickup_clusters','Dropoff_clusters']].apply(lambda x:process_route_fare((x.Pickup_clusters,x.Dropoff_clusters)),axis=1)
dftest['processed_route_fare']=dftest[['Pickup_clusters','Dropoff_clusters']].apply(lambda x:process_route_fare((x.Pickup_clusters,x.Dropoff_clusters)),axis=1)
df.head()
di=df[['Pickup_clusters','Dropoff_clusters','fare_amount']].groupby(['Pickup_clusters',"Dropoff_clusters"]).count().to_dict()
def process_route_popularity(val):
    return di['fare_amount'][val]

df['processed_route_rank']=df[['Pickup_clusters','Dropoff_clusters']].apply(lambda x:process_route_popularity((x.Pickup_clusters,x.Dropoff_clusters)),axis=1)
dftest['processed_route_rank']=dftest[['Pickup_clusters','Dropoff_clusters']].apply(lambda x:process_route_popularity((x.Pickup_clusters,x.Dropoff_clusters)),axis=1)
df.head()
di=df[['Pickup_clusters','Dropoff_clusters','passenger_count','fare_amount']].groupby(['Pickup_clusters',"Dropoff_clusters",'passenger_count']).median().to_dict()
def process_route_people_fare(val):
    return di['fare_amount'][val]

df['processed_route_people_fare']=df[['Pickup_clusters','Dropoff_clusters','passenger_count']].apply(lambda x:process_route_people_fare((x.Pickup_clusters,x.Dropoff_clusters,x.passenger_count)),axis=1)
dftest['processed_route_people_fare']=dftest[['Pickup_clusters','Dropoff_clusters','passenger_count']].apply(lambda x:process_route_people_fare((x.Pickup_clusters,x.Dropoff_clusters,x.passenger_count)),axis=1)
dftest.head()
df.processed_hour_num_of_records
di=df[['Pickup_clusters','Dropoff_clusters','processed_hour_num_of_records','passenger_count','fare_amount']].groupby(['Pickup_clusters',"Dropoff_clusters",'processed_hour_num_of_records','passenger_count']).median().to_dict()
def process_route_hour_people_fare(val):
    return di['fare_amount'][val]

df['processed_route_hour_people_fare']=df[['Pickup_clusters','Dropoff_clusters','processed_hour_num_of_records','passenger_count']].apply(lambda x:process_route_hour_people_fare((x.Pickup_clusters,x.Dropoff_clusters,x.processed_hour_num_of_records,x.passenger_count)),axis=1)
dftest['processed_route_hour_people_fare']=dftest[['Pickup_clusters','Dropoff_clusters','processed_hour_num_of_records','passenger_count']].apply(lambda x:process_route_hour_people_fare((x.Pickup_clusters,x.Dropoff_clusters,x.processed_hour_num_of_records,x.passenger_count)),axis=1)
df.head()
# # Test - Train Split
# train=rd.sample(range(len(df)),round(0.8*len(df)))
# test=list(set(range(len(df)))-set(train))
# dftrain=df.iloc[train,:]
# dftest=df.iloc[test,:]
#df.to_csv('final_train2.csv')
#dftest.to_csv('final_test2.csv')
#print (len(df),len(dftest))
cols=['passenger_count', 'month', 'year', 'weekday', 'hour',
       'week_of_month', 'processed_hour_num_of_records', 'processed_hour_fare',
       'Distance', 'Manhattan_Distance', 'Pickup_clusters', 'Dropoff_clusters',
       'processed_pickup_fare', 'processed_dropoff_fare',
       'processed_route_fare', 'processed_route_rank',
       'processed_route_people_fare', 'processed_route_hour_people_fare']
labelcol='fare_amount'