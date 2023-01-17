import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy as sp

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv')

train.head(5)
print (train.shape)
# lat and long number comes from & credit to DrGuillermo: Animation

xlim = [-74.03, -73.77]

ylim = [40.63, 40.85]

train = train[(train.pickup_longitude> xlim[0]) & (train.pickup_longitude < xlim[1])]

train = train[(train.dropoff_longitude> xlim[0]) & (train.dropoff_longitude < xlim[1])]

train = train[(train.pickup_latitude> ylim[0]) & (train.pickup_latitude < ylim[1])]

train = train[(train.dropoff_latitude> ylim[0]) & (train.dropoff_latitude < ylim[1])]
plt.plot(train['pickup_longitude'], train['pickup_latitude'], '.', color='k', alpha=0.8)

plt.title('Pickup Location Lat and Long', weight = 'bold')

plt.show()
plt.plot(train['dropoff_longitude'], train['dropoff_latitude'], '.', color='k', alpha=0.8)

plt.title('Dropoff Location Lat and Long', weight = 'bold')

plt.show()
from math import radians, cos, sin, asin, sqrt



def haversine_distance(row):

#     lon1, lat1, lon2, lat2):

    """

    Calculate the circle distance between two points in lat and lon

    on the earth (specified in decimal degrees)

    returning distance in miles

    """

    # need to convert decimal degrees to radians 

    # a unit of angle, equal to an angle at the center of a circle whose arc is equal in length to the radius.

    lon1, lat1, lon2, lat2 = row['pickup_longitude'], row['pickup_latitude'], row['dropoff_longitude'], row['dropoff_latitude']

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])



    # haversine formula 

    dlon = lon2 - lon1 

    dlat = lat2 - lat1 

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2

    c = 2 * asin(sqrt(a)) 

    r = 6371 # Radius of earth in kilometers. Use 3956 for miles

    return c * r

#applying to the dataset

train['haversine_distance'] = train.apply(haversine_distance, axis=1)
train.dtypes
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])

train["pickup_day"] = train["pickup_datetime"].dt.strftime('%u').astype(int)

train["pickup_hour"] = train["pickup_datetime"].dt.strftime('%H').astype(int)

train["pickup_month"] = train["pickup_datetime"].dt.strftime('%m').astype(int)
train[:4]
weekday_dict = {1: "Mon",

                       2: "Tues",

                       3: "Wed",

                       4: "Thurs",

                       5: "Fri",

                       6: "Sat",

                       7: "Sun"}

train['weekday']=train['pickup_day'].map(weekday_dict)
month_dict = {1: "Jan",

                       2: "Feb",

                       3: "March",

                       4: "April",

                       5: "May",

                       6: "June",

                       7:"July",

                       8:"Aug",

                       9:"Sep",

                       10:"Oct",

                       11:"Nov",

                       12:"Dec"}

train['month']=train['pickup_month'].map(month_dict)
# only select useful columns

subset_train = train[['trip_duration','haversine_distance', 'weekday','month','pickup_hour']]

subset_train[:5]
sns.pairplot(subset_train);
weekday_list = ['Mon','Tues','Wed','Thurs','Fri','Sat','Sun']

g = sns.factorplot(kind='bar',        # Boxplot

               y='trip_duration',       # Y-axis - values for boxplot

               x='weekday',        # X-axis - first factor

               #estimator = np.sum, 

               data=subset_train,        # Dataframe 

               size=6,            # Figure size (x100px)      

               aspect=1.6,        # Width = size * aspect 

               order = list(weekday_list),

               legend_out=False) 

plt.title('Avg Trip Durations by Weekday\n', weight = 'bold', size = 20)

plt.xlabel('Weekday', size = 18,weight = 'bold')

plt.ylabel('Average trip duration', size = 18,weight = 'bold')

g.set_xticklabels(rotation=45)
weekday_list = ['Mon','Tues','Wed','Thurs','Fri','Sat','Sun']

g = sns.factorplot(kind='bar',        # Boxplot

               y='haversine_distance',       # Y-axis - values for boxplot

               x='weekday',        # X-axis - first factor

               estimator = np.sum, 

               data=subset_train,        # Dataframe 

               size=6,            # Figure size (x100px)      

               aspect=1.6,        # Width = size * aspect 

               order = list(weekday_list),

               legend_out=False) 

plt.title('Total Distance (in miles) by Weekday\n', weight = 'bold', size = 20)

plt.xlabel('Weekday', size = 18,weight = 'bold')

plt.ylabel('Total Distance (in miles) ', size = 18,weight = 'bold')

g.set_xticklabels(rotation=45)
sns.set(font_scale=1.3)

g = sns.factorplot('pickup_hour', 

                   'trip_duration', 

                   hue = 'weekday', 

                   estimator = np.mean, 

                   data = subset_train, 

                   size = 8, 

                   aspect = 2, 

                    ci=None,

                   legend_out=False)

sns.plt.title('Average Duration by Hour of Day and Day of Week \n',weight='bold', size = 20)

plt.xlabel('start hour', size = 18,weight = 'bold')

plt.ylabel('avg duration', size = 18,weight = 'bold')

#g.set_xticklabels(rotation=60)
sns.set(font_scale=1.3)

g = sns.factorplot('pickup_hour', 

                   'haversine_distance', 

                   hue = 'weekday', 

                   estimator = np.sum, 

                   data = subset_train, 

                   size = 8, 

                   aspect = 2, 

                   ci=None,

                   legend_out=False)

sns.plt.title('Total Distance by Hour of Day and Day of Week \n',weight='bold', size = 20)

plt.xlabel('start hour', size = 18,weight = 'bold')

plt.ylabel('Total Distance', size = 18,weight = 'bold')

#g.set_xticklabels(rotation=60)
# running on a sampling due to time constraits and computational power

plt.figure(figsize=(10,6))

sns.distplot(subset_train['haversine_distance'][:10000], kde=False, rug=True)

plt.title('distribution: haversine distance \n', weight = 'bold', size = 15)
plt.figure(figsize=(10,6))

sns.distplot(subset_train['trip_duration'][:10000], kde=False, rug=True)

plt.title('distribution: duration \n', weight = 'bold', size = 15)
#log1p (x) Return the natural logarithm of 1+x (base e)

subset_train['log_haversine_distance'] = np.log1p(subset_train['haversine_distance'])

subset_train['log_duration'] = np.log1p(subset_train['trip_duration'])
plt.figure(figsize=(10,6))

sns.distplot(subset_train['log_duration'][:10000], kde=False, rug=True)

plt.title('log distribution: duration \n', weight = 'bold', size = 15)
plt.figure(figsize=(10,6))

sns.distplot(subset_train['log_haversine_distance'][:10000], kde=False, rug=True)

plt.title('log distribution: haversine distance \n', weight = 'bold', size = 15)
subset_train['pickup_hour']=subset_train.pickup_hour.astype(str)

df = subset_train[['log_duration','log_haversine_distance', 'weekday','month','pickup_hour']]
from sklearn.cross_validation import train_test_split

X = df.drop("log_duration",axis=1)

y = df["log_duration"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print (X_train.shape)

print (y_train.shape)

print (X_test.shape)

print (y_test.shape)