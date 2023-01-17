# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import zipfile as zf
train_zip = zf.ZipFile('/kaggle/input/nyc-taxi-trip-duration/train.zip')

train_set = pd.read_csv(train_zip.open('train.csv')) #training set



test_zip = zf.ZipFile('/kaggle/input/nyc-taxi-trip-duration/test.zip')

test_set = pd.read_csv(test_zip.open('test.csv')) #testing set



sample_submission_zip = zf.ZipFile('/kaggle/input/nyc-taxi-trip-duration/sample_submission.zip')

sample_submission = pd.read_csv(sample_submission_zip.open('sample_submission.csv')) #format of result wanted
train_set.head()

train_set.describe()
train_set.head()

test_set.describe()
train_set.info()

test_set.info()
train_set.shape
test_set.shape
train_set.isna().sum()
test_set.isna().sum()
print('Is there duplicates ? {} '.format((len(train_set['id']) != len(set(train_set['id'])))))
plt.figure()

plt.title("Boxplot")

train_set.boxplot()
ax = train_set['vendor_id'].value_counts(normalize=True).plot(kind='barh')

ax.set_ylabel('Vendor id')

ax.set_xlabel('Percentage')

ax.set_title("Repartition of vendor id", fontdict={'fontsize': 18});
ax2 = train_set['store_and_fwd_flag'].value_counts(normalize=True).plot(kind='bar')

ax2.set_ylabel('Flag')

ax2.set_xlabel('Percentage')

ax2.set_title("Repartition of store and forward flag", fontdict={'fontsize': 18});
plt.figure()

plt.scatter(train_set['pickup_longitude'],train_set['pickup_latitude'], c='red', marker=".", s=8)

plt.title('Pickup coordinates')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.show()
plt.figure()

plt.scatter(train_set['dropoff_longitude'], train_set['dropoff_latitude'], c='blue', marker=".", s=8)

plt.title('Pickup coordinates')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.show()
from datetime import datetime as dt



train_set['pickup_datetime'] = pd.to_datetime(train_set['pickup_datetime'])

train_set['dropoff_datetime'] = pd.to_datetime(train_set['dropoff_datetime'])
print('Incorrect matches between trip_duration column and calculated trip duration : {}'.format((train_set['trip_duration']!=train_set['dropoff_datetime'].sub(train_set['pickup_datetime'], axis=0)/np.timedelta64(1, 's')).sum()))

print('Numbers of passengers from {} to {} '.format(train_set['passenger_count'].min(), train_set['passenger_count'].max()))

print('Trip duration in seconds: {} seconds to {} hours'.format(train_set['trip_duration'].min(), train_set['trip_duration'].max()/(3600)))
e = train_set.loc[(train_set['passenger_count'] == 0) | (train_set['trip_duration'] < 60) | (train_set['trip_duration'] > 3600*3)]



train_set = train_set.loc[(train_set['passenger_count'] > 0)]

train_set = train_set.loc[(train_set['trip_duration'] > 60)]

train_set = train_set.loc[(train_set['trip_duration'] < 3600*2)]

train_set = train_set.loc[(train_set.pickup_longitude > -90)]

train_set = train_set.loc[(train_set.pickup_latitude < 50)]



train_set["pickup_date"] = train_set["pickup_datetime"].map(lambda x: x.date())

train_set["pickup_time"] = train_set["pickup_datetime"].map(lambda x: x.time())

train_set["pickup_weekday"] = train_set["pickup_datetime"].map(lambda x: x.isoweekday())
train_set.drop(['dropoff_datetime'], axis=1, inplace=True) #this column doesn't exist in test_set
test_set['pickup_datetime'] = pd.to_datetime(test_set['pickup_datetime'])

test_set["pickup_date"] = test_set["pickup_datetime"].map(lambda x: x.date())

test_set["pickup_time"] = test_set["pickup_datetime"].map(lambda x: x.time())

test_set["pickup_weekday"] = test_set["pickup_datetime"].map(lambda x: x.isoweekday())
'''train_set["dropoff_date"] = train_set["dropoff_datetime"].map(lambda x: x.date())

train_set["dropoff_time"] = train_set["dropoff_datetime"].map(lambda x: x.time())

train_set["dropoff_weekday"] = train_set["dropoff_datetime"].map(lambda x: x.isoweekday())'''
train_set.info()
train_set.drop(['pickup_datetime'], axis=1, inplace=True)
'''import folium



ny_map = folium.Map(location=[40.738, -73.98],

                        zoom_start=10,

                        tiles="CartoDB dark_matter")'''
'''locations = train_set.loc[:, ["pickup_latitude",

                              "pickup_longitude",

                              "vendor_id"]]



for index, row in locations.iterrows():

    if(row["vendor_id"] == 1):

        color = "#0A8A9F"

    else:

        color = "#E37222"

    folium.CircleMarker(location=(row["pickup_latitude"],

                                  row["pickup_longitude"]),

                        radius=5,

                        color=color,

                        fill=True).add_to(ny_map)'''

def haversine_np(lon1, lat1, lon2, lat2):

    """

    Calculate the great circle distance between two points

    on the earth (specified in decimal degrees)



    All args must be of equal length.    



    """

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])



    dlon = lon2 - lon1

    dlat = lat2 - lat1



    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2



    c = 2 * np.arcsin(np.sqrt(a))

    km = 6367 * c

    return km
train_set['distance'] = haversine_np( train_set['pickup_latitude'].values,

    train_set['pickup_longitude'].values, 

    train_set['dropoff_latitude'].values,

    train_set['dropoff_longitude'].values

    )
train_set.info()
test_set['distance'] = haversine_np( test_set['pickup_latitude'].values,

    test_set['pickup_longitude'].values, 

    test_set['dropoff_latitude'].values,

    test_set['dropoff_longitude'].values

    )
test_set.info()
plt.figure()

plt.title('Distance outliers')

train_set.boxplot(column='distance', return_type='axes');
train_set = train_set.loc[(train_set.distance < 200)]
'''from haversine import haversine

trip_distance = []



for i in train_set['id']:

    row = train_set.loc[(train_set['id'] == i)]

    pickup_coord = (row['pickup_latitude'], row['pickup_longitude'])

    dropoff_coord = (row['dropoff_latitude'], row['dropoff_longitude'])

    distance = haversine(pickup_coord, dropoff_coord)

    trip_distance.append(distance)

trip_distance[0:10]'''
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_squared_error as MSE
NUM_FEATURES = [ column for column in train_set.columns if (train_set[column].dtype == np.float64 or train_set[column].dtype == np.int64) ]

CAT_FEATURES = [ column for column in train_set.columns if (train_set[column].dtype == np.bool or train_set[column].dtype == np.object) ]

TARGET = 'trip_duration'





NUM_FEATURES.remove('trip_duration')

CAT_FEATURES.remove('id')

def preprocess(X, CAT_FEATURES):

    for c in CAT_FEATURES:

        X[c] = X[c].astype('category').cat.codes

X = train_set.loc[:,NUM_FEATURES + CAT_FEATURES] #features

y = train_set[TARGET] #target



preprocess(X, CAT_FEATURES)



X.shape, y.shape
X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=42)

X_train.shape, y_train.shape, X_test.shape, y_test.shape
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_hat = rf.predict(X_test)

y_hat.shape
MSE(y_hat, y_test)
np.sqrt(MSE(y_hat, y_test))
r2_score(y_hat, y_test)
X.columns
test_set.columns
test_set.drop(['pickup_datetime'], axis=1, inplace=True)
NUM_FEATURES_TEST = [ column for column in test_set.columns if (test_set[column].dtype == np.float64 or test_set[column].dtype == np.int64) ]

CAT_FEATURES_TEST = [ column for column in test_set.columns if (test_set[column].dtype == np.bool or test_set[column].dtype == np.object) ]



CAT_FEATURES_TEST.remove('id')



X_test_set = test_set.loc[:,NUM_FEATURES_TEST + CAT_FEATURES_TEST]

preprocess(X_test_set, CAT_FEATURES_TEST)

y_test_set = rf.predict(X_test_set)
test_set['trip_duration'] = y_test_set

test_set.head()

result = ['id', 'trip_duration']

result_set = test_set[result]



result_set
submission = pd.DataFrame( {'id': test_set['id'], 'trip_duration':  test_set['trip_duration']}, columns = ['id', 'trip_duration']) 

submission.to_csv('submission.csv', index=False)