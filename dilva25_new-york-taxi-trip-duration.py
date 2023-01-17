import os



import numpy as np

import pandas as pd

import seaborn as sns



import math
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_log_error as MSLE

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_predict

from sklearn.preprocessing import StandardScaler
import matplotlib as mpl

import matplotlib.pyplot as plt



mpl.rcParams['axes.titlesize']=12

mpl.rcParams['xtick.labelsize']=12

mpl.rcParams['ytick.labelsize']=12





%matplotlib inline 
from sklearn.preprocessing import StandardScaler
BASEPATH = os.path.join('../input')

TRAIN_PATH = os.path.join(BASEPATH, 'train.csv')

TEST_PATH = os.path.join(BASEPATH, 'test.csv')
train = pd.read_csv(TRAIN_PATH)
train.head()
train.describe()
print("Train size :",len(train))
mpl.rcParams['figure.figsize']=(10,5)



plt.hist(train['trip_duration'])

plt.title("Trip Duration Distribution");
train.trip_duration[train['trip_duration'] > 3000][train['trip_duration'] < 10000].hist()

plt.title("Under 50 Minutes Trips Duration Distribution");
train.trip_duration[train['trip_duration'] > 9500][train['trip_duration'] < 15000].hist()

plt.title("Under 50 Minutes Trips Duration Distribution");
test = pd.read_csv(TEST_PATH)

test.head()
train.dtypes, test.dtypes
NUM_VARS = ['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'pickup_latitude',

           'dropoff_longitude', 'dropoff_latitude']

CAT_VARS = ['pickup_datetime','dropoff_datetime', 'store_and_fwd_flag']
train['vendor_id'].hist();
train['passenger_count'].hist();
train['pickup_longitude'].plot.box()

plt.title("Pickup Longitude Distribution");
plt.boxplot(train['pickup_latitude'])

plt.title("Pickup Lattitude Distribution");
plt.scatter(train['pickup_longitude'],train['pickup_latitude']);
mpl.rcParams['figure.figsize']=(15,7)

plt.scatter(train['pickup_longitude'],train['pickup_latitude'])

plt.axis([-74.02,-73.92,40.7,40.82])
mpl.rcParams['figure.figsize']=(10,5)

plt.scatter(train['pickup_longitude'],train['trip_duration'],color='r');
train2 = train[train['pickup_longitude'] > -80][train['pickup_longitude'] < -60]

train2 = train2[train2['trip_duration'] <= 500000]

plt.scatter(train2['pickup_longitude'],train2['trip_duration'],color='r');
train2 = train2[train2['pickup_longitude'] > -75][train2['pickup_longitude'] < -72]

plt.scatter(train2['pickup_longitude'],train2['trip_duration'],color='r');
plt.scatter(train2['pickup_latitude'],train2['trip_duration'],color='g');
train2 = train2[train2['pickup_latitude'] < 42][train2['pickup_latitude'] > 39]

plt.scatter(train2['pickup_latitude'],train2['trip_duration'],color='g');
plt.scatter(train2['pickup_longitude'],train2['pickup_latitude']);
mpl.rcParams['figure.figsize']=(10,5)

plt.boxplot(train2['dropoff_longitude']);
plt.boxplot(train2['dropoff_latitude']);
plt.scatter(train2['dropoff_longitude'],train2['dropoff_latitude']);
mpl.rcParams['figure.figsize']=(15,7)

plt.scatter(train2['dropoff_longitude'],train2['dropoff_latitude'])

plt.axis([-74.02,-73.92,40.7,40.82])
mpl.rcParams['figure.figsize']=(10,5)

plt.scatter(train2['dropoff_longitude'],train2['trip_duration'],color='r');
train2 = train2[train2['dropoff_longitude'] > -76][train2['dropoff_longitude'] < -72]

plt.scatter(train2['dropoff_longitude'],train2['trip_duration'],color='r');
plt.scatter(train2['dropoff_latitude'],train2['trip_duration'],color='g');
train2 = train2[train2['dropoff_latitude'] > 40][train2['dropoff_latitude'] < 41.5]

plt.scatter(train2['dropoff_latitude'],train2['trip_duration'],color='g');
plt.scatter(train2['dropoff_longitude'],train2['dropoff_latitude']);
missing_val_count = (train2.isnull().sum())

missing_val_count
# train

for column in CAT_VARS:

    train2[column] = train2[column].astype('category').cat.codes

train2.head()
# test

for column in CAT_VARS:

    if(column != 'dropoff_datetime'):

        test[column] = test[column].astype('category').cat.codes

test.head()
train2.dtypes
train3 = np.abs(train2[['vendor_id', 'pickup_datetime', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag']]) 

train3 = np.log1p(train3)

train3.head()
X_train = train3 #= train2[['vendor_id', 'pickup_datetime', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag']]

X_train.head()
y_train = np.log1p(train2['trip_duration'])

y_train.head()
t_X, val_X, t_y, val_y = train_test_split(X_train, y_train, test_size=0.2, random_state = 0)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(t_X)
X_val_scaled = scaler.transform(val_X)
rf = RandomForestRegressor(random_state=1, n_jobs=-1)

rf.fit(X_train_scaled, t_y)

preds = rf.predict(X_val_scaled)
print(np.sqrt(MSLE(np.exp(val_y), np.exp(preds))))
cv_preds = cross_val_predict(rf, X_train, y_train, cv=10, n_jobs=-1)
# check cv_preds size

print(cv_preds)



print(np.sqrt(MSLE(np.exp(y_train), np.exp(cv_preds))))
test_p = np.log1p(np.abs(test[['vendor_id', 'pickup_datetime', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag']]))

test_scaled = scaler.transform(test_p)
preds = rf.predict(test_scaled)

np.exp(preds)
sub = pd.DataFrame({'id':test.id,'trip_duration':np.exp(preds)})

sub.head(5)
sub.to_csv('submission.csv', index=0)