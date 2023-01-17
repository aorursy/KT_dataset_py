import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import zipfile as zf # zip files processing

import os

import matplotlib.pyplot as plt



# Useful to calculate distance from coordonates

from geopy import distance

from geopy import Point



from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error as MSE



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Get test, train and sample submission data

test_data = pd.read_csv('../input/nyc-taxi-trip-duration/test.zip', index_col=0)

train_data = pd.read_csv('../input/nyc-taxi-trip-duration/train.zip', index_col=0)

kaggle_submission = pd.read_csv('../input/nyc-taxi-trip-duration/sample_submission.zip')
# Train data visualization

train_data.head()
# Show train data info

train_data.info()
# Duplicated rows

train_data.duplicated().sum()
# Delete duplicated rows

train_data = train_data.drop_duplicates()
# Number of not a number values for each column

train_data.isna().sum()
plt.boxplot([train_data.trip_duration])
# Remove trips that take more than 5000 seconds

train_data = train_data[(train_data.trip_duration < 5000)]
# Remove trips without passengers

train_data = train_data[(train_data.passenger_count > 0)]
def add_date_features(data) :

    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])

    

    data['pickup_year'] = data['pickup_datetime'].dt.year

    data['pickup_month'] = data['pickup_datetime'].dt.month

    data['pickup_day'] = data['pickup_datetime'].dt.day

    data['pickup_weekday'] = data['pickup_datetime'].dt.weekday

    data['pickup_hour'] = data['pickup_datetime'].dt.hour

    data['pickup_minutes'] = data['pickup_hour'] * 60 + data['pickup_datetime'].dt.minute

    data['pickup_seconds'] = data['pickup_minutes'] * 60 + data['pickup_datetime'].dt.second

    

add_date_features(train_data)

add_date_features(test_data)
train_data.head()
train_data = pd.concat([train_data, pd.get_dummies(train_data['store_and_fwd_flag'])], axis=1)

test_data = pd.concat([test_data, pd.get_dummies(test_data['store_and_fwd_flag'])], axis=1)



train_data = pd.concat([train_data, pd.get_dummies(train_data['vendor_id'])], axis=1)

test_data = pd.concat([test_data, pd.get_dummies(test_data['vendor_id'])], axis=1)
# Useful function for getting a distance from coordinates

def calculate_distance(lat1, lng1, lat2, lng2):

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    AVG_EARTH_RADIUS = 6371 # kilometers

    lat = lat2 - lat1

    lng = lng2 - lng1

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))

    return h



# Add distance field

train_data['distance'] = calculate_distance(train_data['pickup_latitude'].values, train_data['pickup_longitude'].values, train_data['dropoff_latitude'].values, train_data['dropoff_longitude'].values)



test_data['distance'] = calculate_distance(test_data['pickup_latitude'].values, test_data['pickup_longitude'].values, test_data['dropoff_latitude'].values, test_data['dropoff_longitude'].values)
# Check distance values

train_data.boxplot(column='distance', return_type='axes');
# Remove distance outliers

train_data = train_data[(train_data.distance < 200)]
# Add speed feature

train_data['speed'] = train_data.distance / train_data.trip_duration
columns = ['passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',

                    'dropoff_latitude', 'pickup_month' , 'pickup_hour','pickup_minutes',

                    'pickup_seconds', 'pickup_weekday', 'pickup_day', 'distance']



X_train = train_data[columns]

y_train = train_data['trip_duration']
X_train.head()
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
X_test = test_data[columns]

X_test.head()
y_pred = rf.predict(X_test)
X_test.index.shape, y_pred.shape
kaggle_submission['trip_duration'] = y_pred

kaggle_submission.head()
kaggle_submission.to_csv('kaggle_submission.csv', index=False)