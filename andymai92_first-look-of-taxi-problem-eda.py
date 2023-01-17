import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error

from sklearn.model_selection import train_test_split

taxi_train = pd.read_csv('../input/train.csv')
taxi_test = pd.read_csv('../input/test.csv')
taxi_train.info()

print('\n')

taxi_test.info()
taxi_train['pick_date'] = pd.to_datetime(taxi_train['pickup_datetime'])
taxi_train['drop_date'] = pd.to_datetime(taxi_train['dropoff_datetime'])
taxi_train.drop(['pickup_datetime', 'dropoff_datetime'], 1, inplace=True)
taxi_train.head()
from sklearn.neighbors import DistanceMetric
dist = DistanceMetric.get_metric('manhattan')
def manhattan_dist(x):

    pick_long = x[0]

    pick_lat = x[1]

    drop_long = x[2]

    drop_lat = x[3]

    V = [[pick_long, pick_lat],[drop_long,drop_lat]]

    return dist.pairwise(V)[0][1]
taxi_train['distance'] = taxi_train[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(manhattan_dist,1)
taxi_train.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],1,inplace=True)
taxi_train.head()
plt.figure(figsize=(12,8))

sns.heatmap(taxi_train.corr()*100,annot=True)
def rush_hours(x):

    hour = x.hour

    if (hour >= 7 and hour <= 9) or (hour >= 16 and hour <= 18):

        return 1

    else:

        return 0
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
taxi_train['Day of week'] = taxi_train['pick_date'].apply(lambda time: time.dayofweek)
taxi_train['Day of week'] = taxi_train['Day of week'].map(dmap)
taxi_train.head()
plt.figure(figsize=(12,8))

sns.barplot(x='Day of week', y='trip_duration', data=taxi_train)
taxi_train['Rush_hours'] = taxi_train['pick_date'].apply(rush_hours)
plt.figure(figsize=(8,5))

sns.barplot(x='Rush_hours', y='trip_duration', data=taxi_train)
fed_holidays = ((16,1),(20,2),(29,5),(4,7),(4,9),(9,10),(10,11),(11,11),(23,11))
state_holidays = ((12,2),(13,2),(9,10),(24,11))
def federal_holidays(x):

    day = x.day

    month = x.month

    if (day >=24 and month == 12) or (day <= 2 and month ==1):

        return 3

    elif (day, month) in fed_holidays:

        return 2

    elif (day, month) in state_holidays:

        return 1

    else:

        return 0
taxi_train['Holidays'] = taxi_train['pick_date'].apply(federal_holidays)
plt.figure(figsize=(10,8))

sns.barplot(x='Holidays', y='trip_duration', data=taxi_train,palette='rainbow')
plt.figure(figsize=(10,8))

sns.barplot(x='passenger_count', y='trip_duration',hue='vendor_id', data=taxi_train,palette='rainbow')
taxi_train['Day of week'] = taxi_train['pick_date'].apply(lambda time: time.dayofweek)
taxi_train.head()
plt.figure(figsize=(12,8))

sns.heatmap(taxi_train.corr()*100,annot=True)