# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import zipfile

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
zf_train = zipfile.ZipFile('../input/nyc-taxi-trip-duration/train.zip')

data = pd.read_csv(zf_train.open('train.csv'))



zf_test = zipfile.ZipFile('../input/nyc-taxi-trip-duration/test.zip')

test_data = pd.read_csv(zf_test.open('test.csv'))
data.describe()
#On ne garde pas les voyages sans passagers.

data = data[data.passenger_count != 0]
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])

data['dropoff_datetime'] = pd.to_datetime(data['dropoff_datetime'])

test_data['pickup_datetime'] = pd.to_datetime(test_data['pickup_datetime'])



data['hour'] = data.pickup_datetime.dt.hour

data['day'] = data.pickup_datetime.dt.dayofweek

data['month'] = data.pickup_datetime.dt.month

test_data['hour'] = test_data.pickup_datetime.dt.hour

test_data['day'] = test_data.pickup_datetime.dt.dayofweek

test_data['month'] = test_data.pickup_datetime.dt.month







X = data[["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", 'month', 'hour', 'day',]]

y = data["trip_duration"]
X.shape, y.shape
data.keys()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5, random_state = 42)

X_train.shape, y_train.shape, X_test.shape, y_test.shape
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=10, min_samples_split=15, max_features='auto', max_depth=90, bootstrap=True)

rf.fit(X_train, y_train)

rf.score(X_train, y_train)
my_prediction = rf.predict(X_test)

my_prediction = pd.DataFrame(my_prediction)
output = pd.concat([pd.DataFrame(test_data["id"]), my_prediction], axis=1)

output.columns = ["id", "trip_duration"]

output = output.drop_duplicates(keep = False)

output = output.dropna()
output.tail()
output.to_csv('submission.csv', index=False)