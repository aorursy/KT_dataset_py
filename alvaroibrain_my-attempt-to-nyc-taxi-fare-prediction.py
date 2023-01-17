import pandas as pd
import os
import numpy as np
import random

from matplotlib import pyplot as plt

from math import radians, cos, sin, asin, sqrt

#Nicer style :)
plt.style.use('seaborn')

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.listdir('../input')
#Load dataset
train = pd.read_csv('../input/train.csv', nrows=800_000)
# Drop rows which contain NaN values
train = train.dropna()
# Drop wrong fare rows. Minimum amount is 2.50 (http://www.nyc.gov/html/tlc/html/passenger/taxicab_rate.shtml)
train = train[train['fare_amount'] > 2] 
# Must have  between 0, 5 (+1 child)
# https://www1.nyc.gov/nyc-resources/faq/484/how-many-passengers-are-allowed-in-a-taxi
train = train[(train['passenger_count'] > 0)  & (train['passenger_count'] < 6)]
train.head()
#work in a copy
train2 = train.copy()
"""
    Auxiliar function to calculate distance between two coordinates.
    You could also install harvesine python package.
    Returns distance in kilometers.
    
    Found on StackOverflow
"""
def haversine(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Probed radius of earth in kilometers.
    return c * r
#Convert "objects" to "datetime"
train2['key'] = pd.to_datetime(train['key'])
train2['pickup_datetime'] = pd.to_datetime(train['key'])
train2.dtypes
#Free memory
del(train)
ax = train2.plot.scatter(x ='pickup_longitude', y = 'pickup_latitude', s=10)
train2.plot.scatter(x ='dropoff_longitude', y = 'dropoff_latitude', color='Red', s=9, ax=ax)
train2.describe()
#Limit to the New York area.
train2 = train2[(train2['pickup_latitude'] < 41.1) & (train2['pickup_latitude'] > 40)]
train2 = train2[(train2['dropoff_latitude'] < 41.1) & (train2['dropoff_latitude'] > 40)]
train2 = train2[(train2['pickup_longitude'] < -50) & (train2['pickup_longitude'] > - 80)]
train2 = train2[(train2['dropoff_longitude'] < -50) & (train2['dropoff_longitude'] > - 80)]
ax = train2.plot.scatter(x ='pickup_longitude', y = 'pickup_latitude', s=9)
train2.plot.scatter(x ='dropoff_longitude', y = 'dropoff_latitude', color='Red', s=9, ax=ax)

ny_img_map = plt.imread('https://i.imgur.com/zrZwViw.png')


plt.rcParams["figure.figsize"] = (11,10)

ax = train2.plot.scatter(x ='pickup_longitude', y = 'pickup_latitude', s=0.7, zorder=1)
train2.plot.scatter(x ='dropoff_longitude', y = 'dropoff_latitude', color='Red', s=0.2, zorder=1, ax=ax)
ax.imshow(ny_img_map, zorder=0, extent=[-75.354559, -72.856968, 40.121653, 41.087085])

# I wrote this  function based on https://www.kaggle.com/breemen/nyc-taxi-fare-data-exploration kernel.
# Previously I tried doing it with shapefiles but it took arround 3.5sec to analyze each point. This method
# is faster
def is_in_water(map_mask, left_top, right_bottom, coord):
    width = map_mask.shape[1]
    height = map_mask.shape[0]
    array_sea_color = np.array([0., 0. , 0.], dtype='float32')
    
    #Delta distance in limits
    dist_x = (right_bottom[0] - left_top[0]) 
    dist_y = (left_top[1] - right_bottom[1])

    #Map delta distance to pixels
    pix_x = ((coord[0] - left_top[0]) / dist_x) * width
    pix_y = ((coord[1] - right_bottom[1]) / dist_y) * height

    if pix_x < 0 or pix_y < 0 or pix_x > width or pix_y > height:
        return True #coord outside bounds
    
    #Get the color of the pixel
    color = map_mask[ height- int(pix_y), int(pix_x)]
    
    #Is in sea?  (compare color)
    return np.array_equal(color, array_sea_color)
#Custom mask made quickly with photoshop. Black pixels will be considered sea,
ny_img_mask = plt.imread('https://i.imgur.com/ov0cDqP.png')
#Bounds of the area
left_top = (-74.8, 41.1)
right_bottom = (-72.8, 40.092)
#Drop all the points in the water.
train2['is_water'] = train2.apply(lambda row: \
                                   is_in_water(ny_img_mask, left_top, right_bottom, \
                                               (row.pickup_longitude, row.pickup_latitude)), axis=1)
train2 =train2[train2['is_water'] == False]
train2['is_water'] = train2.apply(lambda row: \
                                   is_in_water(ny_img_mask, left_top, right_bottom, \
                                               (row.dropoff_longitude, row.dropoff_latitude)), axis=1)
train2 =train2[train2['is_water'] == False]
ny_img_map = plt.imread('https://i.imgur.com/IqJFL8l.png')
#ny_img_map = plt.imread('https://i.imgur.com/ov0cDqP.png') #Show mask in figure instead of the map.


plt.rcParams["figure.figsize"] = (18,15)

ax = train2.plot.scatter(x ='pickup_longitude', y = 'pickup_latitude', s=1, zorder=1)
train2.plot.scatter(x ='dropoff_longitude', y = 'dropoff_latitude', color='Red', s=0.01, zorder=1, ax=ax)
ax.imshow(ny_img_map, zorder=0, extent=[-74.8, -72.8, 40.092, 41.1])
#Remove old column
train2 = train2.drop('is_water', axis=1)
#Visualize the city
fig, ax = plt.subplots(2,1, figsize=(16,16))
train2[(train2['pickup_longitude'] > -74.2) & (train2['pickup_longitude'] < -73.85) &
       (train2['pickup_latitude'] < 40.9) & (train2['pickup_latitude'] > 40.6)]\
        .plot.scatter(x ='pickup_longitude', y = 'pickup_latitude', s=0.1, ax=ax[0])

train2[(train2['dropoff_longitude'] > -74.2) & (train2['dropoff_longitude'] < -73.85) &
       (train2['dropoff_latitude'] < 40.9) & (train2['dropoff_latitude'] > 40.6)]\
        .plot.scatter(x ='dropoff_longitude', y = 'dropoff_latitude', s=0.1, ax=ax[1], c='red')
#Add derived column with the distance traveled
train2['distance'] = train2.apply(lambda row: haversine(row.pickup_longitude, row.pickup_latitude, 
                        row.dropoff_longitude, row.dropoff_latitude), axis=1)

#If some distance traveled is less than 100 meters I will take that row as invalid.
train2 = train2[train2['distance'] > 0.1]
#Long trips can have agreed fares, so they are outliers. I will take trips shorter than 50km
train2 = train2[train2['distance'] <= 50]
#Some stats of the dataset
train2.describe()
fig, ax = plt.subplots(1,2, figsize=(16,8))
train2[train2['distance'] <= 6.6][['fare_amount', 'distance']].plot.scatter('distance', 'fare_amount', s=3,
                                                                           ax=ax[0])
train2[train2['distance'] > 6.6][['fare_amount', 'distance']].plot.scatter('distance', 'fare_amount', s=3, ax=ax[1])
#Both together
train2[['fare_amount', 'distance']].plot.scatter('distance', 'fare_amount', s=3)
def airport_trip(pick_lon, pick_lat, drop_lon, drop_lat):     
    #Airport coords
    la_guardia_coords = (-73.872192, 40.774210)
    jfk_coords = (-73.786932, 40.645753)
    newark_coords = (-74.182634, 40.692509)
    
    #La guardia airport
    if haversine(pick_lon,pick_lat, la_guardia_coords[0], la_guardia_coords[1]) < 3 or \
        haversine(drop_lon,drop_lat, la_guardia_coords[0], la_guardia_coords[1]) < 3:
            return 1
    #JFK airport
    if haversine(pick_lon,pick_lat, jfk_coords[0], jfk_coords[1]) < 3 or \
        haversine(drop_lon,drop_lat, jfk_coords[0], jfk_coords[1]) < 3:
            return 2
    #Newark airport
    if haversine(pick_lon,pick_lat, newark_coords[0], newark_coords[1]) < 3 or \
        haversine(drop_lon,drop_lat, newark_coords[0], newark_coords[1]) < 3:
            return 3
    #Trip is no from/to airport
    return 0
train2['airport'] = train2.apply(lambda row: airport_trip(row.pickup_longitude, row.pickup_latitude, row.dropoff_longitude, row.dropoff_latitude), axis=1)
print("LAG: " +  str(len(train2[train2['airport'] == 1])))
print('JFK: ' + str(len(train2[train2['airport'] == 2])))
print('NEW: ' + str(len(train2[train2['airport'] == 3])))
print('NONE: '+ str(len(train2[train2['airport'] == 0])))
# Trips outside of the city may have a agreed fare. We should remove them.
# http://www.nyc.gov/html/tlc/html/passenger/taxicab_rate.shtml

#Let's define a point in the "center" of all trips. It represents the center of NYC. (It's Manhattan btw).
nyc_center_coords = (-73.972500, 40.769804) 

#Add a column to the DF. to see how far from the center are the dropoff points.
train2['distance_center'] = train2.apply(lambda row: 
                                         haversine(nyc_center_coords[0], nyc_center_coords[1],
                                                   row.dropoff_longitude, row.dropoff_latitude), axis=1)
#train2 = train2[train2['distance_center'] < 50]
fig, ax = plt.subplots(1,3, figsize=(16,8))
train2[train2['airport'] == 1][['fare_amount', 'distance']].plot.scatter('distance', 'fare_amount', s=3, ax=ax[0])
train2[train2['airport'] == 2][['fare_amount', 'distance']].plot.scatter('distance', 'fare_amount', s=3, ax=ax[1])
train2[train2['airport'] == 3][['fare_amount', 'distance']].plot.scatter('distance', 'fare_amount', s=3, ax=ax[2])
train2[train2['airport'] == 0][['fare_amount', 'distance']].plot.scatter('distance', 'fare_amount', s=3)
train2[(train2['airport'] == 0) & (train2['fare_amount'] > 56.75)& (train2['fare_amount'] < 57)][['fare_amount', 'distance']].plot.scatter('distance', 'fare_amount', s=10)
tmp = train2[(train2['airport'] == 0) & (train2['fare_amount'] > 56.75)& (train2['fare_amount'] < 57)]
#Where are the dropoff points?
ny_img_map = plt.imread('https://i.imgur.com/IqJFL8l.png')

plt.rcParams["figure.figsize"] = (18,15)

ax = tmp.plot.scatter(x ='pickup_longitude', y = 'pickup_latitude', s=10, zorder=1)
tmp.plot.scatter(x ='dropoff_longitude', y = 'dropoff_latitude', color='Red', s=20, zorder=1, ax=ax)
ax.imshow(ny_img_map, zorder=0, extent=[-74.8, -72.8, 40.092, 41.1])
#def haversine(lon1, lat1, lon2, lat2):
def is_to_center(pick_lon, pick_lat, drop_lon, drop_lat):
    if haversine(nyc_center_coords[0], nyc_center_coords[1], drop_lon, drop_lat) < 2 and \
    haversine(nyc_center_coords[0], nyc_center_coords[1], pick_lon, pick_lat)  > 9:
        return 1
    else:
        return 0
train2['to_center'] = train2.apply(lambda row: is_to_center(row.pickup_longitude, row.pickup_latitude, row.dropoff_longitude, row.dropoff_latitude), axis=1)
#Free memory
del(tmp)
from sklearn.cluster import KMeans
pickups_arr = np.array(train2[['pickup_longitude', 'pickup_latitude']])
dropoffs_arr = np.array(train2[['dropoff_longitude', 'dropoff_latitude']])
kmeans_picks = KMeans(n_clusters=30, random_state=0).fit(pickups_arr)
kmeans_drops = KMeans(n_clusters=30, random_state=0).fit(dropoffs_arr)
#Visualize the centers in the map
fig, ax = plt.subplots(1,1, figsize=(16,8))
ax.scatter(x=kmeans_picks.cluster_centers_[:,0], y=kmeans_picks.cluster_centers_[:,1])
ax.scatter(x=kmeans_drops.cluster_centers_[:,0], y=kmeans_drops.cluster_centers_[:,1], c='red')
ax.imshow(ny_img_map, zorder=0, extent=[-74.8, -72.8, 40.092, 41.1])
del(pickups_arr, dropoffs_arr) # Free memory
def predict_clusters(dataframe, kmeans_picks, kmeans_drops):
    #arr_picks = kmeans_picks.predict(np.array(dataframe[['pickup_longitude', 'pickup_latitude']]))
    #arr_drops = kmeans_drops.predict(np.array(dataframe[['dropoff_longitude', 'dropoff_latitude']]))
    copy = dataframe.copy()
    copy['cluster_pick'] = kmeans_picks.predict(np.array(copy[['pickup_longitude', 'pickup_latitude']]))
    copy['cluster_drop'] = kmeans_drops.predict(np.array(copy[['dropoff_longitude', 'dropoff_latitude']]))
    return copy
train2 = predict_clusters(train2, kmeans_picks, kmeans_drops)
train2

#Hour of the trip
train2['hour'] = train2.apply(lambda row: row.pickup_datetime.hour, axis=1)
#Day of week
train2['dow'] = train2.apply(lambda row: row.pickup_datetime.weekday(), axis=1)
#fare_per_km
train2['fare_per_km'] = train2['fare_amount'] / train2['distance']
#Study the average fare by hour in each week day
df = train2.groupby(['dow', 'hour'])['fare_amount'].mean()
fig, ax = plt.subplots(1,2, figsize=(16,8))

ax[0].set_title('avg. Fare amount on each weekday')
plt.title('avg. Fare amount on each weekday')
for i in range(7):
    df[i].plot(ax=ax[0], label="DAY {}".format(i))
ax[0].legend()    

df = train2.groupby(['dow', 'hour'])['fare_per_km'].mean()

plt.title('avg. Fare per km on each weekday')
for i in range(7):
    df[i].plot(ax=ax[1], label="DAY {}".format(i))
    
ax[1].legend()
del(df)
train2['work_day'] = 0
train2['peak_hour'] = 0
train2.loc[(train2['dow'] >= 0) & (train2['dow'] < 5), ['work_day']] = 1
train2.loc[(train2['hour'] > 5) & (train2['hour'] < 20), ['peak_hour']] = 1



train2.describe()
fig, ax = plt.subplots(3,1, figsize=(16,8))
train2[(train2['distance'] <= 1) & (train2['fare_per_km'] < 500)][['fare_per_km', 'distance']].plot.scatter('distance', 'fare_per_km', s=3, ax=ax[0])

train2[train2['distance'] <= 1][['fare_per_km', 'distance']].plot.scatter('distance', 'fare_per_km', s=3, ax=ax[1])

train2[['fare_per_km', 'distance']].plot.scatter('distance', 'fare_per_km', s=3, ax=ax[2])
train2 = train2[train2['fare_per_km'] < 150]
train2.describe()
fig, ax = plt.subplots(3,1, figsize=(16,8))
train2[(train2['distance'] <= 1) & (train2['fare_per_km'] < 500)][['fare_per_km', 'distance']].plot.scatter('distance', 'fare_per_km', s=3, ax=ax[0])

train2[train2['distance'] <= 1][['fare_per_km', 'distance']].plot.scatter('distance', 'fare_per_km', s=3, ax=ax[1])

train2[['fare_per_km', 'distance']].plot.scatter('distance', 'fare_per_km', s=3, ax=ax[2])






from sklearn import linear_model
model = linear_model.LinearRegression()
#Train only with 6000 samples
model.fit(np.array(train2.head(6000).distance).reshape(-1,1), np.array(train2.head(6000).fare_amount).reshape(-1,1))

#Make the line
pred = model.predict(np.arange(40).reshape(-1,1))
fig,ax = plt.subplots(1,1, figsize=(10,10))
train2[train2.distance < 35][['fare_amount', 'distance']].plot.scatter('distance', 'fare_amount', s=3, ax=ax)
ax.plot(pred, color='red')

train3 = train2.copy()
train3['pred'] = train3.apply(lambda row: model.predict(np.array([row.distance]).reshape(1,-1))[0][0],  axis=1)
# Score the prediction
((train3.fare_amount - train3.pred) ** 2).mean() ** .5
corr = train2.copy()
corr = corr.drop(['key', 'pickup_datetime'], axis=1)
import matplotlib.ticker as ticker
plt.style.use('default')
fig, ax = plt.subplots(1, 1, figsize=(15,15))
cax = ax.matshow(corr.corr(), interpolation='nearest', origin='upper')

ax.set_xticklabels([''] + corr.columns.tolist(), rotation=45)
ax.set_yticklabels([''] + corr.columns.tolist())

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.style.use('seaborn')

fig.colorbar(cax)
#
corr.corr().head(1)
train3 = train2.copy()
import xgboost as xgb
import math
#Split 20% of the dataset for evaluating the model
test_size = math.floor(train3.count()[0]*0.2)

data_train = train3.head(train3.count()[0] - test_size)
data_train_fare = data_train['fare_amount']

data_test = train3.tail(test_size)
data_test_fare = data_test['fare_amount']

data_train= data_train.drop(['key', 'pickup_datetime', 'fare_amount', 'fare_per_km', 'hour', 'dow', 'passenger_count'], axis=1)
data_test= data_test.drop(['key', 'pickup_datetime', 'fare_amount', 'fare_per_km', 'hour', 'dow', 'passenger_count'], axis=1)

#Build and train model
model = xgb.train(params={'objective':'reg:linear','eval_metric':'rmse', 'max_depth':'6', 'n_estimators':'300'},
                    dtrain=xgb.DMatrix(data_train, label=data_train_fare),
                    num_boost_round=170, 
                    early_stopping_rounds=10,evals=[(xgb.DMatrix(data_test, label=data_test_fare),'test')])




#Load test dataset
test_eval_base = pd.read_csv('../input/train.csv', nrows=1_600_000)[800_000:]
test_eval = test_eval_base.copy()

#Convert "objects" to "datetime"
test_eval['key'] = pd.to_datetime(test_eval['key'])
test_eval['pickup_datetime'] = pd.to_datetime(test_eval['key'])


test_eval['distance'] = test_eval.apply(lambda row: haversine(row.pickup_longitude, row.pickup_latitude, 
                        row.dropoff_longitude, row.dropoff_latitude), axis=1)

#Airport
test_eval['airport'] = test_eval.apply(lambda row: airport_trip(row.pickup_longitude, row.pickup_latitude, row.dropoff_longitude, row.dropoff_latitude), axis=1)

test_eval['distance_center'] = test_eval.apply(lambda row: 
                                         haversine(nyc_center_coords[0], nyc_center_coords[1],
                                                   row.dropoff_longitude, row.dropoff_latitude), axis=1)

#test_data = predict_clusters(test_data, kmeans_picks, kmeans_drops)

#Hour of the trip
test_eval['hour'] = test_eval.apply(lambda row: row.pickup_datetime.hour, axis=1)
#Day of week
test_eval['dow'] = test_eval.apply(lambda row: row.pickup_datetime.weekday(), axis=1)
#Is a trip to NYC center?
test_eval['to_center'] = test_eval.apply(lambda row: is_to_center(row.pickup_longitude, row.pickup_latitude, row.dropoff_longitude, row.dropoff_latitude), axis=1)
#Drop passenger count
test_eval = test_eval.drop(['passenger_count', 'fare_amount'], axis=1)

test_eval['work_day'] = 0
test_eval['peak_hour'] = 0
test_eval.loc[(test_eval['dow'] >= 0) & (test_eval['dow'] < 5), ['work_day']] = 1
test_eval.loc[(test_eval['hour'] > 5) & (test_eval['hour'] < 20), ['peak_hour']] = 1
test_eval.loc[(test_eval['dow'] >= 0) & (test_eval['dow'] < 5), ['work_day']] = 1
test_eval.loc[(test_eval['hour'] > 5) & (test_eval['hour'] < 20), ['peak_hour']] = 1

test_eval= test_eval.drop(['key', 'pickup_datetime', 'dow', 'hour'], axis=1)
results = model.predict(xgb.DMatrix(test_eval))
#Evaluate model (RMSE)
test_eval_base['pred'] = results
((test_eval_base.fare_amount - test_eval_base.pred) ** 2).mean() ** .5
#Free memory
del(test_eval, test_eval_base)
import keras as k
nn = k.Sequential()
nn.add(k.layers.Dense(10, input_shape=(10,)))
nn.add(k.layers.Dense(20, activation='tanh'))
nn.add(k.layers.Dense(100, activation='selu'))
nn.add(k.layers.Dense(80, activation='tanh'))
nn.add(k.layers.Dense(10, activation='sigmoid'))
nn.add(k.layers.Dense(10, activation='sigmoid'))
nn.add(k.layers.Dense(1, activation='relu'))
nn.compile(k.optimizers.Adam(), loss='mae')
test_size = math.floor(train3.count()[0]*0.2)

data_train = train3.head(train3.count()[0] - test_size)
data_train_fare = data_train['fare_amount']

data_test = train3.tail(test_size)
data_test_fare = data_test['fare_amount']

data_train= data_train.drop(['key', 'pickup_datetime', 'fare_amount', 'fare_per_km'], axis=1)
data_test= data_test.drop(['key', 'pickup_datetime', 'fare_amount', 'fare_per_km'], axis=1)
d_train = np.array(data_train)
d_train_l = np.array(data_train_fare)

d_test = np.array(data_test)
d_test_l = np.array(data_test_fare)
nn.fit(x=d_train, y=d_train_l, batch_size=32, validation_split=0.2, epochs=100,     
       callbacks=[k.callbacks.EarlyStopping(monitor='val_loss')])

test_data = pd.read_csv('../input/test.csv')
#Convert "objects" to "datetime"
test_data['key'] = pd.to_datetime(test_data['key'])
test_data['pickup_datetime'] = pd.to_datetime(test_data['key'])


test_data['distance'] = test_data.apply(lambda row: haversine(row.pickup_longitude, row.pickup_latitude, 
                        row.dropoff_longitude, row.dropoff_latitude), axis=1)

#Airport
test_data['airport'] = test_data.apply(lambda row: airport_trip(row.pickup_longitude, row.pickup_latitude, row.dropoff_longitude, row.dropoff_latitude), axis=1)

test_data['distance_center'] = test_data.apply(lambda row: 
                                         haversine(nyc_center_coords[0], nyc_center_coords[1],
                                                   row.dropoff_longitude, row.dropoff_latitude), axis=1)

#test_data = predict_clusters(test_data, kmeans_picks, kmeans_drops)

#Hour of the trip
test_data['hour'] = test_data.apply(lambda row: row.pickup_datetime.hour, axis=1)
#Day of week
test_data['dow'] = test_data.apply(lambda row: row.pickup_datetime.weekday(), axis=1)
#Is a trip to NYC center?
test_data['to_center'] = test_data.apply(lambda row: is_to_center(row.pickup_longitude, row.pickup_latitude, row.dropoff_longitude, row.dropoff_latitude), axis=1)

test_data['work_day'] = 0
test_data['peak_hour'] = 0
test_data.loc[(test_data['dow'] >= 0) & (test_data['dow'] < 5), ['work_day']] = 1
test_data.loc[(test_data['hour'] > 5) & (test_data['hour'] < 20), ['peak_hour']] = 1

#Drop passenger count
test_data = test_data.drop(['passenger_count', 'hour', 'dow'], axis=1)
data_test= test_data.drop(['key', 'pickup_datetime'], axis=1)
results = model.predict(xgb.DMatrix(data_test))
results
#An error uploading the .csv if not done this way
sample_sub = pd.read_csv('../input/sample_submission.csv')
sample_sub['fare_amount'] = results
sample_sub.to_csv('air_submission.csv', index=False)
