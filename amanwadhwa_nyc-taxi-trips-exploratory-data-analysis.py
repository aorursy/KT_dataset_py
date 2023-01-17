# #Python Libraries
import numpy as np
import scipy as sp
import pandas as pd
import statsmodels
import pandas_profiling

from sklearn import linear_model

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys
import time
import requests
import datetime

import missingno as msno
# #Datasets
!ls ../input/
# #Train and Test Datasets
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")

df_sample_submission = pd.read_csv("../input/sample_submission.csv")
print("Total number of samples in train dataset: ", df_train.shape[0])
print("Number of columns in train dataset: ", df_train.shape[1])
df_train.head()
df_train.describe()
df_train.info()
print("Total number of samples in test dataset: ", df_test.shape[0])
print("Number of columns in test dataset: ", df_test.shape[1])
df_test.head()
df_sample_submission.shape
df_sample_submission.head()
df_train.isnull().sum()
df_test.isnull().sum()

print("Number of ids in the train dataset: ", len(df_train["id"]))
print("Number of unique ids in the train dataset: ", len(pd.unique(df_train["id"])), "\n")

print("Number of ids in the test dataset: ", len(df_test["id"]))
print("Number of unique ids in the test dataset: ", len(pd.unique(df_test["id"])), "\n")

print("Number of common ids(if any) between the train and test datasets: ", \
len(set(df_train["id"].values).intersection(set(df_test["id"].values))))
print("Number of vendor_ids in the train dataset: ", len(df_train["vendor_id"]))
print("Number of unique vendor_ids in the train dataset: ", len(pd.unique(df_train["vendor_id"])), "\n")

print("Number of vendor_ids in the test dataset: ", len(df_test["vendor_id"]))
print("Number of unique vendor_ids in the test dataset: ", len(pd.unique(df_test["vendor_id"])), "\n")

# #The number of observations in the dataset from each of the two companies i.e. 1 and 2, seems to be comparable
# #across the train and test datasets
sns.countplot(x="vendor_id", data=df_train)
sns.countplot(x="vendor_id", data=df_test)
sns.countplot(x="passenger_count", data=df_train[df_train["vendor_id"] == 1])
sns.countplot(x="passenger_count", data=df_train[df_train["vendor_id"] == 2])
sns.countplot(x="passenger_count", data=df_test[df_test["vendor_id"] == 1])
sns.countplot(x="passenger_count", data=df_test[df_test["vendor_id"] == 2])

# #String to Datetime conversion
df_train["pickup_datetime"] = pd.to_datetime(df_train["pickup_datetime"])
df_train["dropoff_datetime"] = pd.to_datetime(df_train["dropoff_datetime"])

df_test["pickup_datetime"] = pd.to_datetime(df_test["pickup_datetime"])

# #trip_duration represents the difference between the dropoff_datetime and the pickup_datetime in the
# #train dataset
df_train["trip_duration"].describe()
# #The trip_duration would be a lot more intuitive when the datetime representation is used, 
# #rather than the representation with seconds. 
(df_train["dropoff_datetime"] - df_train["pickup_datetime"]).describe()
plt.figure(figsize=(10,10))
plt.scatter(range(len(df_train["trip_duration"])), np.sort(df_train["trip_duration"]))
plt.xlabel('index')
plt.ylabel('trip_duration in seconds')
plt.show()
# #Removing the outliers in the dataset
df_train = df_train[df_train["trip_duration"] < 500000]
(df_train["dropoff_datetime"] - df_train["pickup_datetime"]).describe()
plt.figure(figsize=(10,10))
plt.scatter(range(len(df_train["trip_duration"])), np.sort(df_train["trip_duration"]))
plt.xlabel('index')
plt.ylabel('trip_duration in seconds')
plt.show()

sns.countplot(x="store_and_fwd_flag", data=df_train)
len(df_train[df_train["store_and_fwd_flag"] == "N"])*100.0/(df_train.count()[0])
set(df_train[df_train["store_and_fwd_flag"] == "Y"]["vendor_id"])

from haversine import haversine
def calculate_haversine_distance(var_row):
    return haversine((var_row["pickup_latitude"], var_row["pickup_longitude"]), 
                     (var_row["dropoff_latitude"], var_row["dropoff_longitude"]), miles = True)
# #Calculating the Haversine Distance
# #The haversine formula determines the great-circle distance between two points on a sphere 
# #given their longitudes and latitudes.
df_train["haversine_distance"] = df_train.apply(lambda row: calculate_haversine_distance(row), axis=1)
df_train.head()
df_train["haversine_distance"].describe()

#plt.figure(figsize=(10,10))
#sns.regplot(x="haversine_distance", y="trip_duration", data=df_train)
df_train[df_train["haversine_distance"] > 100]
#plt.figure(figsize=(10,10))
#sns.regplot(x="haversine_distance", y="trip_duration", data=df_train[df_train["haversine_distance"] < 100])

print("Train dataset start date: ", min(df_train["pickup_datetime"]))
print("Train dataset end date: ", max(df_train["pickup_datetime"]))
# #Conversion to pandas to_datetime has already been performed in section 5.5
# #df_train["pickup_datetime"] = pd.to_datetime(df_train['pickup_datetime'])


df_train["pickup_dayofweek"] = df_train.pickup_datetime.dt.dayofweek
df_train["pickup_weekday_name"] = df_train.pickup_datetime.dt.weekday_name
df_train["pickup_hour"] = df_train.pickup_datetime.dt.hour
df_train["pickup_month"] = df_train.pickup_datetime.dt.month
df_train.head()

plt.figure(figsize=(12,8))
sns.countplot(x="pickup_weekday_name", data=df_train)
plt.show()
plt.figure(figsize=(12,8))
sns.countplot(x="pickup_hour", data=df_train)
plt.show()
plt.figure(figsize=(12,8))
sns.countplot(x="pickup_month", data=df_train)
plt.show()

df_train.trip_duration.describe()
df_train_agg = df_train.groupby('pickup_weekday_name')['trip_duration'].aggregate(np.median).reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(df_train_agg.pickup_weekday_name.values, df_train_agg.trip_duration.values)
plt.show()
df_train.groupby('pickup_weekday_name')['trip_duration'].describe()
df_train_agg = df_train.groupby('pickup_hour')['trip_duration'].aggregate(np.median).reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(df_train_agg.pickup_hour.values, df_train_agg.trip_duration.values)
plt.show()
df_train.groupby('pickup_hour')['trip_duration'].describe()
df_train_agg = df_train.groupby('pickup_month')['trip_duration'].aggregate(np.median).reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(df_train_agg.pickup_month.values, df_train_agg.trip_duration.values)
plt.show()
df_train.groupby('pickup_month')['trip_duration'].describe()


