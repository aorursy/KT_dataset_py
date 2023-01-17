# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import timedelta

import datetime as dt

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 10]

import seaborn as sns

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.cluster import MiniBatchKMeans

t0 = dt.datetime.now()

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')
train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)

test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)

train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date

test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date

train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)

train['store_and_fwd_flag'] = 1 * (train.store_and_fwd_flag.values == 'Y')

test['store_and_fwd_flag'] = 1 * (test.store_and_fwd_flag.values == 'Y')

train['check_trip_duration'] = (train['dropoff_datetime'] - train['pickup_datetime']).map(lambda x: x.total_seconds())

duration_difference = train[np.abs(train['check_trip_duration'].values  - train['trip_duration'].values) > 1]

print('Trip_duration and datetimes are ok.') if len(duration_difference[['pickup_datetime', 'dropoff_datetime', 'trip_duration', 'check_trip_duration']]) == 0 else print('Ooops.')
# Let's compute pickup hour for each ride

train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])

train['pickup_hour'] = train.pickup_datetime.dt.hour

train['day_week'] = train.pickup_datetime.dt.weekday

# Get pick up hour for test data as well

test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])

test['pickup_hour'] = test.pickup_datetime.dt.hour

test['day_week'] = test.pickup_datetime.dt.weekday
train['log_trip_duration'] = np.log(train['trip_duration'].values + 1)

sns.distplot(train['log_trip_duration'].values, bins=100)

plt.xlabel('log(trip_duration)')

plt.ylabel('number of train records')

plt.show()
#from pandas.plotting import parallel_coordinates

#dct = {'training': train.groupby('pickup_date').count()[['id']], 

#       'testing': train.groupby('pickup_date').count()[['id']] }



#df = pd.DataFrame.from_dict(dct)



#parallel_coordinates(df, 'training')



#df = pd.DataFrame( {train.groupby('pickup_date').count()[['id']] columns=['a', 'b', 'c', 'd']})



#df.plot.area();

#fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)



plt.plot((train.groupby('pickup_date').count()[['id']]), label='train', color = 'g')

plt.plot((test.groupby('pickup_date').count()[['id']]), label='train', color = 'r')



plt.title('Train and test period complete overlap.')

plt.legend(loc=0)

plt.ylabel('number of records')

plt.show()
def bearing_array(lat1, lng1, lat2, lng2):

    AVG_EARTH_RADIUS = 6371  # in km

    lng_delta_rad = np.radians(lng2 - lng1)

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    y = np.sin(lng_delta_rad) * np.cos(lat2)

    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)

    return np.degrees(np.arctan2(y, x))



train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values, train['pickup_longitude'].values, 

                                          train['dropoff_latitude'].values, train['dropoff_longitude'].values)



test.loc[:, 'direction'] = bearing_array(test['pickup_latitude'].values, test['pickup_longitude'].values, 

                                         test['dropoff_latitude'].values, test['dropoff_longitude'].values)
allLat  = np.array(list(train['pickup_latitude'])  + list(train['dropoff_latitude']))

allLong = np.array(list(train['pickup_longitude']) + list(train['dropoff_longitude']))



longLimits = [np.percentile(allLong, 0.3), np.percentile(allLong, 99.7)]

latLimits  = [np.percentile(allLat , 0.3), np.percentile(allLat , 99.7)]

durLimits  = [np.percentile(train['trip_duration'], 0.4), np.percentile(train['trip_duration'], 99.7)]



train = train[(train['pickup_latitude']   >= latLimits[0] ) & (train['pickup_latitude']   <= latLimits[1]) ]

train = train[(train['dropoff_latitude']  >= latLimits[0] ) & (train['dropoff_latitude']  <= latLimits[1]) ]

train = train[(train['pickup_longitude']  >= longLimits[0]) & (train['pickup_longitude']  <= longLimits[1])]

train = train[(train['dropoff_longitude'] >= longLimits[0]) & (train['dropoff_longitude'] <= longLimits[1])]

train = train[(train['trip_duration']     >= durLimits[0] ) & (train['trip_duration']     <= durLimits[1]) ]

train = train.reset_index(drop=True)



allLat  = np.array(list(train['pickup_latitude'])  + list(train['dropoff_latitude']))

allLong = np.array(list(train['pickup_longitude']) + list(train['dropoff_longitude']))



# convert fields to sensible units

medianLat  = np.percentile(allLat,50)

medianLong = np.percentile(allLong,50)



latMultiplier  = 111.32

longMultiplier = np.cos(medianLat*(np.pi/180.0)) * 111.32



train['duration [min]'] = train['trip_duration']/60.0

train['src lat [km]']   = latMultiplier  * (train['pickup_latitude']   - medianLat)

train['src long [km]']  = longMultiplier * (train['pickup_longitude']  - medianLong)

train['dst lat [km]']   = latMultiplier  * (train['dropoff_latitude']  - medianLat)

train['dst long [km]']  = longMultiplier * (train['dropoff_longitude'] - medianLong)



allLat  = np.array(list(train['src lat [km]'])  + list(train['dst lat [km]']))

allLong = np.array(list(train['src long [km]']) + list(train['dst long [km]']))
# show the log density of pickup and dropoff locations

imageSize = (700,700)

longRange = [-5,19]

latRange = [-13,11]



allLatInds  = imageSize[0] - (imageSize[0] * (allLat  - latRange[0])  / (latRange[1]  - latRange[0]) ).astype(int)

allLongInds =                (imageSize[1] * (allLong - longRange[0]) / (longRange[1] - longRange[0])).astype(int)



locationDensityImage = np.zeros(imageSize)

for latInd, longInd in zip(allLatInds,allLongInds):

    locationDensityImage[latInd,longInd] += 1



fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(12,12))

ax.imshow(np.log(locationDensityImage+1),cmap='magma')

ax.set_axis_off()
from sklearn import decomposition

from scipy import stats

from sklearn import cluster





tripAttributes = np.array(train.loc[:,['src lat [km]','src long [km]','dst lat [km]','dst long [km]','duration [min]', 'direction']])

meanTripAttr = tripAttributes.mean(axis=0)

stdTripAttr  = tripAttributes.std(axis=0)

tripAttributes = stats.zscore(tripAttributes, axis=0)



# choose number of clusters





numClusters = 60

TripKmeansModel = cluster.MiniBatchKMeans(n_clusters=numClusters, batch_size=120000, n_init=100)

clusterInds = TripKmeansModel.fit_predict(tripAttributes)



clusterTotalCounts, _ = np.histogram(clusterInds, bins=numClusters)

sortedClusterInds = np.flipud(np.argsort(clusterTotalCounts))



plt.figure(figsize=(12,4)); plt.title('Cluster Histogram of all trip')

plt.bar(range(1,numClusters+1),clusterTotalCounts[sortedClusterInds])

plt.ylabel('Frequency [counts]'); plt.xlabel('Cluster index (sorted by cluster frequency)')

plt.xlim(0,numClusters+1)
#%% show the templeate trips on the map 



def ConvertToImageCoords(latCoord, longCoord, latRange, longRange, imageSize):

    latInds  = imageSize[0] - (imageSize[0] * (latCoord  - latRange[0])  / (latRange[1]  - latRange[0]) ).astype(int)

    longInds =                (imageSize[1] * (longCoord - longRange[0]) / (longRange[1] - longRange[0])).astype(int)



    return latInds, longInds



templateTrips = TripKmeansModel.cluster_centers_ * np.tile(stdTripAttr,(numClusters,1)) + np.tile(meanTripAttr,(numClusters,1))



srcCoords = templateTrips[:,:2]

dstCoords = templateTrips[:,2:4]



srcImCoords = ConvertToImageCoords(srcCoords[:,0],srcCoords[:,1], latRange, longRange, imageSize)

dstImCoords = ConvertToImageCoords(dstCoords[:,0],dstCoords[:,1], latRange, longRange, imageSize)



plt.figure(figsize=(12,12))

plt.imshow(np.log(locationDensityImage+1),cmap='magma'); plt.grid('off')

plt.scatter(srcImCoords[1],srcImCoords[0],c='y',s=200,alpha=0.9)

plt.scatter(dstImCoords[1],dstImCoords[0],c='g',s=200,alpha=0.9)



for i in range(len(srcImCoords[0])):

    plt.arrow(srcImCoords[1][i],srcImCoords[0][i], dstImCoords[1][i]-srcImCoords[1][i], dstImCoords[0][i]-srcImCoords[0][i], 

              edgecolor='c', facecolor='c', width=2.4,alpha=0.6,head_width=10.0,head_length=10.0,length_includes_head=True)
#color = sns.color_palette()



grouped_df = train.groupby('pickup_hour')['trip_duration'].aggregate(np.mean).reset_index()

plt.figure(figsize=(12,8))

sns.pointplot(grouped_df.pickup_hour.values, grouped_df.trip_duration.values, alpha=0.8, 

              color='k' )

plt.ylabel('Average trip duration', fontsize=12)

plt.xlabel('Pickup Hour', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
sns.set(style="ticks")

sns.set_context("poster")

sns.boxplot(x="day_week", y="trip_duration", hue="vendor_id", data=train

             )

plt.ylim(0, 6000)

sns.despine(offset=10, trim=True)

train.trip_duration.max()
from sklearn import model_selection, preprocessing

import xgboost as xgb



for f in train.columns:

    if train[f].dtype=='object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train[f].values)) 

        train[f] = lbl.transform(list(train[f].values))

train_y = train.trip_duration.values

train_X = train.drop(["id", "dropoff_datetime", "pickup_datetime", "trip_duration"], axis=1)

from sklearn.ensemble import RandomForestRegressor

from sklearn.grid_search import GridSearchCV

import matplotlib.pyplot as plt



rf_clf = RandomForestRegressor(max_depth=15,n_estimators=100, min_samples_leaf=75,

                                  min_samples_split=100, random_state=10)



# Train the model

rf_clf.fit(train_X, train_y)