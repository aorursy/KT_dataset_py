import time
start_time = time.time()
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pandas import Series
from sklearn.ensemble import RandomForestRegressor

from subprocess import check_output
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train.isnull().sum().sum()
train.columns
test.columns
month = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S" ).month
day_of_week = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S" ).weekday()
day = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S" ).day
hour = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S" ).hour
minute = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S" ).minute

seasons = [0,0,1,1,1,2]
season = lambda x: seasons[(datetime.strptime(x, "%Y-%m-%d %H:%M:%S" ).month-1)]
train['month'] = train['pickup_datetime'].map(month)
train['day'] = train['pickup_datetime'].map(day)
train['day_of_week'] = train['pickup_datetime'].map(day_of_week)
train['hour'] = train['pickup_datetime'].map(hour)
train['minute'] = train['pickup_datetime'].map(minute)
train['season'] = train['pickup_datetime'].map(season)
train['store_and_fwd_flag']=train['store_and_fwd_flag'].map( {'N': 0, 'Y': 1} ).astype(int)
test['month'] = test['pickup_datetime'].map(month)
test['day_of_week'] = test['pickup_datetime'].map(day_of_week)
test['day'] = test['pickup_datetime'].map(day)
test['hour'] = test['pickup_datetime'].map(hour)
test['minute'] = test['pickup_datetime'].map(minute)
test['season'] = test['pickup_datetime'].map(season)

test['store_and_fwd_flag']=test['store_and_fwd_flag'].map( {'N': 0, 'Y': 1} ).astype(int)

plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
Series(train['month']).value_counts().sort_index().plot(kind = 'bar')
plt.xlabel("Month")
plt.ylabel("Count")
plt.title('Which month has the most rides? (train data)')
plt.subplot(1, 2, 2)
Series(test['month']).value_counts().sort_index().plot(kind = 'bar')
plt.xlabel("Month")
plt.ylabel("Count")
plt.title('Which month has the most rides? (test data)')
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
Series(train['day']).value_counts().sort_index().plot(kind = 'bar')
plt.xlabel("Day")
plt.ylabel("Count")
plt.title('Which day has the most rides?(train data)')
plt.subplot(1, 2, 2)
Series(test['day']).value_counts().sort_index().plot(kind = 'bar')
plt.xlabel("Day")
plt.ylabel("Count")
plt.title('Which day has the most rides?(test data)')
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
Series(train['day_of_week']).value_counts().sort_index().plot(kind = 'bar')
plt.xlabel("Day of week")
plt.ylabel("Count")
plt.title('Which day of week has the most rides? \n (0 = Monday)-(train data)')
plt.subplot(1, 2, 2)
Series(test['day_of_week']).value_counts().sort_index().plot(kind = 'bar')
plt.xlabel("Day of week")
plt.ylabel("Count")
plt.title('Which day of week has the most rides? \n (0 = Monday)-(test data)')
from geopy.distance import great_circle
def distance(p_lat,p_long,d_lat,d_long):
    pickup = (p_lat, p_long)
    dropoff = (d_lat, d_long)
    distance_all=great_circle(pickup, dropoff).km
    return distance_all
#for train data
d_train = [distance(train['pickup_latitude'].values[i], train['pickup_longitude'].values[i],
              train['dropoff_latitude'].values[i],train['dropoff_longitude'].values[i]) for i in range(len(train['pickup_latitude']))]
train['distance_est']=d_train
train["d1"]=train['distance_est']<=5
train["d2"]=train['distance_est'].between(5, 10, inclusive=False)
train["d3"]=train['distance_est'].between(10, 20, inclusive=False)
train["d4"]=train['distance_est'].between(20, 30, inclusive=False)
train["d5"]=train['distance_est'].between(30, 100, inclusive=False)
train["d6"]=train['distance_est']>100
#for test data
d_test = [distance(test['pickup_latitude'].values[i], test['pickup_longitude'].values[i],
              test['dropoff_latitude'].values[i],test['dropoff_longitude'].values[i]) for i in range(len(test['pickup_latitude']))]
test['distance_est']=d_test
test["d1"]=test['distance_est']<=5
test["d2"]=test['distance_est'].between(5, 10, inclusive=False)
test["d3"]=test['distance_est'].between(10, 20, inclusive=False)
test["d4"]=test['distance_est'].between(20, 30, inclusive=False)
test["d5"]=test['distance_est'].between(30, 100, inclusive=False)
test["d6"]=test['distance_est']>100
#dimension reduction of pickup and dropoff (not sure about this step)
from sklearn.decomposition import TruncatedSVD
tsvd = TruncatedSVD(n_components=1, random_state=1)
#for train data
frames_p_train=[train['pickup_latitude'],train['pickup_longitude']]
p_train=pd.concat(frames_p_train,axis=1)

frames_d_train=[train['dropoff_latitude'],train['dropoff_longitude']]
d_train=pd.concat(frames_d_train,axis=1)

train['tsvd_p']=tsvd.fit_transform(p_train)
train['tsvd_d']=tsvd.fit_transform(d_train)
#for test data
frames_p_test=[test['pickup_latitude'],test['pickup_longitude']]
p_test=pd.concat(frames_p_test,axis=1)

frames_d_test=[test['dropoff_latitude'],test['dropoff_longitude']]
d_test=pd.concat(frames_d_test,axis=1)

test['tsvd_p']=tsvd.fit_transform(p_test)
test['tsvd_d']=tsvd.fit_transform(d_test)
import math

def calculate_initial_compass_bearing(pointA, pointB):
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing
#for train data
train['direction']=[calculate_initial_compass_bearing((train['pickup_latitude'].loc[i],train['pickup_longitude'].loc[i]), 
                                  (train['dropoff_latitude'].loc[i],train['dropoff_longitude'].loc[i])) for i in range(len(train['pickup_latitude']))]
#for test data
test['direction']=[calculate_initial_compass_bearing((test['pickup_latitude'].loc[i],test['pickup_longitude'].loc[i]), 
                                  (test['dropoff_latitude'].loc[i],test['dropoff_longitude'].loc[i])) for i in range(len(test['pickup_latitude']))]
sns.heatmap(train.corr())
test_id=test["id"]
test = test.drop(['id', 'pickup_datetime'], axis=1)
X = train.drop(['id','trip_duration', 'pickup_datetime','dropoff_datetime'], axis=1)
Y = train['trip_duration']
RF = RandomForestRegressor(verbose=True)
RF.fit(X, Y)
features_list = X.columns.values
feature_importance = RF.feature_importances_
sorted_idx = np.argsort(feature_importance)
 
plt.figure(figsize=(5,10))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importances')
plt.draw()
plt.show()
Y_pred = RF.predict(test)
sub = pd.DataFrame()
sub['id'] = test_id
sub['trip_duration'] = Y_pred
sub.to_csv('RF.csv', index=False)