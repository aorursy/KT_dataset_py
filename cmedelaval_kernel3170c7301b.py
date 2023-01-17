# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/nyc-taxi-trip-duration/train.zip")

test_data = pd.read_csv("/kaggle/input/nyc-taxi-trip-duration/test.zip")
print("Test DataFrame")

test_data.info()

test_data.head()
print("Train DataFrame")

train_data.info()

train_data.head()
print("Il y a {} données dupliqué dans le train".format(train_data.duplicated().sum()))

print("Il y a {} données dupliqué dans le test".format(test_data.duplicated().sum()))
train_data.isnull().sum()
test_data['pickup_datetime'] = pd.to_datetime(test_data['pickup_datetime'])



test_data['month'] = test_data['pickup_datetime'].dt.month

test_data['day'] = test_data['pickup_datetime'].dt.day

test_data['hour'] = test_data['pickup_datetime'].dt.hour

test_data['minute'] = test_data['pickup_datetime'].dt.minute

train_data['pickup_datetime'] = pd.to_datetime(train_data['pickup_datetime'])

train_data['dropoff_datetime'] = pd.to_datetime(train_data['dropoff_datetime'])

train_data['month'] = train_data['pickup_datetime'].dt.month

train_data['day'] = train_data['pickup_datetime'].dt.day

train_data['hour'] = train_data['pickup_datetime'].dt.hour

train_data['minute'] = train_data['pickup_datetime'].dt.minute



train_data.head()
# On prend toutes les valeurs avec au moins un passager

train_data = train_data[(train_data.passenger_count > 0)]
# On prend toutes les valeurs avec un temps de courses inférieur à 24h

train_data = train_data[(train_data.trip_duration < 86400)]
X = train_data[["pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude","month", "day", "hour", "minute"]]

y = train_data["trip_duration"]



train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, random_state = 42)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5, random_state = 42)

X_train.shape, y_train.shape, X_test.shape, y_test.shape
rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=10, min_samples_split=15, max_features='auto', max_depth=90, bootstrap=True)

rf.fit(X_train, y_train)

predictions = rf.predict(test_features)

predictions = predictions.round(0)

ids = pd.DataFrame(test_data["id"])

my_prediction = rf.predict(X_test)

my_prediction = pd.DataFrame(my_prediction)


output = pd.concat([pd.DataFrame(test_data["id"]), my_prediction], axis=1)

output.columns = ["id", "trip_duration"]
output = output.drop_duplicates(keep = False)

output = output.dropna()

output.tail()
output.to_csv("submission.csv", index = False)
rf.score(train_features, train_labels)