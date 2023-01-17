import os

import numpy as np

import pandas as pd

import seaborn as sns

import datetime as dt

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error
train = '../input/train.csv'

test = '../input/test.csv'

dfTrain = pd.read_csv(train)

dfTest = pd.read_csv(test)
dfTrain.head()
dfTest.head()
dfTrain.isna().sum()
dfTest.isna().sum()
plt.subplots(figsize=(18,7))

plt.title("Outliers de train.csv")

dfTrain.boxplot();
print(dfTrain.loc[dfTrain['trip_duration'] > 350000])
print(f"{dfTrain.shape}")

dfTrain = dfTrain.loc[dfTrain['trip_duration']< 350000]

print(f"{dfTrain.shape}")
sns.heatmap(dfTrain.corr())
dfTrain = dfTrain[dfTrain['passenger_count']>= 1]
dfTrain['pickup_datetime'] = pd.to_datetime(dfTrain['pickup_datetime'])

dfTest['pickup_datetime'] = pd.to_datetime(dfTest['pickup_datetime'])



dfTrain['month'] = dfTrain['pickup_datetime'].dt.month

dfTrain['day'] = dfTrain['pickup_datetime'].dt.day

dfTrain['weekday'] = dfTrain['pickup_datetime'].dt.weekday

dfTrain['hour'] = dfTrain['pickup_datetime'].dt.hour

dfTrain['minute'] = dfTrain['pickup_datetime'].dt.minute



dfTest['month'] = dfTest['pickup_datetime'].dt.month

dfTest['day'] = dfTest['pickup_datetime'].dt.day

dfTest['weekday'] = dfTest['pickup_datetime'].dt.weekday

dfTest['hour'] = dfTest['pickup_datetime'].dt.hour

dfTest['minute'] = dfTest['pickup_datetime'].dt.minute



dfTrain['trip_duration'] = np.log1p(dfTrain['trip_duration'].values)



dfTrain.head()
dfTrain.columns
selection_train = ["passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude","month", "day", "weekday", "hour", "minute"]

selection_test = ["passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude", "month", "day", "weekday", "hour", "minute"]

y_train = dfTrain["trip_duration"]

X_train = dfTrain[selection_train] 

X_test = dfTest[selection_test]
y_train.head()
print("Training ...")

m = RandomForestRegressor()

m.fit(X_train, y_train)

#m1.fit(XdfTrain, YdfTrain)

#m2.fit(XdfTrain, YdfTrain)

%timeit

print('Done!!!')
pred = m.predict(X_test)

print('Done!!!')
pred = np.expm1(pred)
my_submission = pd.DataFrame({'id':dfTest.id, 'trip_duration':pred})
my_submission.head(10)
my_submission.to_csv("my_submission.csv", index=False)

print("Done!!!")