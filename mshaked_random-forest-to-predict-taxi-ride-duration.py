# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



train = pd.read_csv("/kaggle/input/nyc-taxi-trip-duration/train.csv")

test = pd.read_csv("/kaggle/input/nyc-taxi-trip-duration/test.csv")
train_features = train[["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]]

train_labels = train["trip_duration"]



test_features = test[["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]]



print('Training Features Shape:', train_features.shape)

print('Training Labels Shape:', train_labels.shape)

print('Testing Features Shape:', test_features.shape)
# Import the model we are using

from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 20 decision trees

rf = RandomForestRegressor(n_estimators = 20, random_state = 42)

# Train the model on training data

rf.fit(train_features, train_labels)



# Use the forest's predict method on the test data

predictions = rf.predict(test_features)

predictions = predictions.round(0).astype(int)
ids = pd.DataFrame(test["id"])

predictions = pd.DataFrame(predictions)

output = pd.concat([ids, predictions], axis=1)

output.columns = ["id", "trip_duration"]



output.to_csv("submission.csv", index = False)