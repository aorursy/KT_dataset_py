import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from geopy.distance import vincenty
#get the distance 'as the crow flies" between 2 points
#we could try getting the route distance and maybe even a bit of traffic information, but that can be done later.
#read the csv file
train = pd.read_csv('../input/train.csv')
train.info()
train.head(4)
train.drop('store_and_fwd_flag',axis=1,inplace = True)
train.drop('dropoff_datetime',axis=1,inplace=True)
def distance(start_long,start_lat,stop_long,stop_lat):
    start = (start_long,start_lat)
    stop = (stop_long,stop_lat)
    return vincenty(start,stop).miles
#this is going to take a while
train['distance'] = train.apply(lambda row: distance(row['pickup_longitude'], row['pickup_latitude'], row['dropoff_longitude'], row['dropoff_latitude']), axis=1)
#check that it worked
train.head(4)
distance = list(train['distance'])
distance.sort()
plt.scatter(range(len(distance)),distance)
plt.show()
train = train.loc[train['distance'] < 15.0]
train = train.loc[train['distance'] > 0.2]
duration = list(train['trip_duration'])
duration.sort()
plt.scatter(range(len(duration)),duration)
ax = plt.gca()
ax.set_yscale('log')
plt.show()
train = train.loc[train['trip_duration'] > 300]
outlier = train.loc[train['trip_duration'] > 10000]
plt.scatter(outlier['trip_duration'],outlier['distance'])
plt.xlabel('trip_duration')
plt.ylabel('distance')
plt.show()
fig = plt.figure()
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
ax[0].scatter(outlier['pickup_longitude'].values, outlier['pickup_latitude'].values,
              color='green', s=1, label='pickup', alpha=0.1)
ax[1].scatter(outlier['dropoff_longitude'].values, outlier['dropoff_latitude'].values,
              color='red', s=1, label='dropoff', alpha=0.1)
ax[0].set_ylabel('latitude')
ax[0].set_xlabel('longitude')
ax[1].set_xlabel('longitude')
plt.ylim([40.6,40.9])
plt.xlim([-74.1,-73.8])
plt.show()
train = train.loc[train['trip_duration'] < 10000]
plt.scatter(train['distance'],train['trip_duration'])
axis = plt.gca()
axis.set_xlabel('distance')
axis.set_ylabel('trip_duration')
plt.show()
train = train.loc[~((train['distance'] < 4.0) & (train['trip_duration'] > 6000))]
train = train.loc[~((train['distance'] < 8.0) & (train['trip_duration'] > 8000))]
plt.scatter(train['distance'],train['trip_duration'])
axis = plt.gca()
axis.set_xlabel('distance')
axis.set_ylabel('trip_duration')
plt.show()
train['trip_duration_hours'] = train['trip_duration']/3600.0
train['average_speed'] = train['distance']/train['trip_duration_hours']
plt.scatter(train['distance'],train['average_speed'])
axis = plt.gca()
axis.set_xlabel('distance')
axis.set_ylabel('average_speed')
plt.show()
train = train.loc[train['average_speed'] < 55.0]
plt.scatter(train['trip_duration_hours'],train['average_speed'])
axis = plt.gca()
axis.set_xlabel('trip_duration_hour')
axis.set_ylabel('average_speed')
plt.show()
train.info()
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
train['pickup_hour'] = train['pickup_datetime'].dt.hour
train['pickup_weekday'] = train['pickup_datetime'].dt.weekday
train['pickup_month'] = train['pickup_datetime'].dt.month
train.head(4)
prediction_var = list(train.columns)
remove = ['id','pickup_datetime','trip_duration','distance','trip_duration_hours','average_speed']
for i in prediction_var[:]:
    if i in remove:
        prediction_var.remove(i)

prediction_var
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
dummy_train , dummy_test = train_test_split(train,test_size = 0.2)
#let's start with the KNeighbors first
knn = KNeighborsRegressor(n_jobs=-1)
knn.fit(dummy_train[prediction_var],dummy_train['trip_duration'])
knn.predict(dummy_test[prediction_var])
knn.score(dummy_test[prediction_var],dummy_test['trip_duration'])
rf = RandomForestRegressor(n_jobs=-1)
rf.fit(dummy_train[prediction_var],dummy_train['trip_duration'])
rf.predict(dummy_test[prediction_var])
rf.score(dummy_test[prediction_var],dummy_test['trip_duration'])
from sklearn.model_selection import GridSearchCV
def gridsearch(model,param_grid,data_x,data_y):
    clf = GridSearchCV(model,param_grid,cv=10,n_jobs=-1)
    clf.fit(data_x,data_y)
    print(clf.best_params_)
    print(clf.best_score_)
    return clf.best_estimator_
knn_param_grid = [{'n_neighbors':[5,10,15,20],'algorithm':['brute'],'weights':['uniform','distance']},
                  {'n_neighbors':[5,10,15,20],'algorithm':['kd_tree'],'weights':['uniform','distance'],'leaf_size':[15,30,45,60]},
                  {'n_neighbors':[5,10,15,20],'algorithm':['ball_tree'],'weights':['uniform','distance'],'leaf_size':[15,30,45,60]}]
rf_param_grid = {'n_estimators':[10,50,100,200],'min_samples_leaf':[10,25,50]}
rf = RandomForestRegressor(min_samples_leaf=10,n_estimators=50,n_jobs=-1)
rf.fit(dummy_train[prediction_var],dummy_train['trip_duration'])
rf.predict(dummy_test[prediction_var])
rf.score(dummy_test[prediction_var],dummy_test['trip_duration'])