# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# IMPORT LIBRARIES

import pandas as pd               # DataFrame support

import numpy as np                # algebra / computations



import matplotlib.pyplot as plt   # plotting

import seaborn as sns             # fancier plotting

%matplotlib inline 

# IMPORT DATA

orders_filepath = "../input/orders.csv"

trips_filepath = "../input/trips.csv"





# IMPORT DATA

base_trips = pd.read_csv(trips_filepath,

                         engine='c',

                         infer_datetime_format=True, # to speed-up datetime parsing

                         parse_dates=['pickup_datetime', 'dropoff_datetime'])



base_orders = pd.read_csv(orders_filepath, 

                          engine='c',

                          infer_datetime_format=True, 

                          parse_dates=['pickup_datetime'])
# MAKING SAMPLE OF DATA TO SPEEDUP CALC

sample_number = 2000000

strips = base_trips.sample(sample_number)

sorders = base_orders.sample(sample_number)





# CHECK DATA USAGE

print('Memory usage trips, Mb: {:.2f}\n'.format(strips.memory_usage().sum()/2**20))

print('Memory usage orders, Mb: {:.2f}\n'.format(sorders.memory_usage().sum()/2**20))
# OVERALL INFO

print('Trips Info: ---------------------')

print(strips.info())

print('Orders Info: ---------------------')

print(sorders.info())



#CHECK FOR MISSING VALUES

print(strips.isnull().sum()) 

print(sorders.isnull().sum()) 
# CHECK FOR DUPLICATES: NO DUPLICATES

print('No of Duplicates, strips - order_id: {}'.format(len(strips) - 

                                              len(strips.drop_duplicates(subset='order_id'))))

print('No of Duplicates, sorders - order_id: {}'.format(len(sorders) - 

                                              len(sorders.drop_duplicates(subset='order_id'))))



# CHECK GEOGRAPHICAL BOUNDS, Latitude: 0.0 to 400.9, Longitude: -171.4 to 121.8

print('strips Latitude bounds: {} to {}'.format(

    max(strips.pickup_latitude.min(), strips.dropoff_latitude.min()),

    max(strips.pickup_latitude.max(), strips.dropoff_latitude.max())

))

print('strips Longitude bounds: {} to {}'.format(

    max(strips.pickup_longitude.min(), strips.dropoff_longitude.min()),

    max(strips.pickup_longitude.max(), strips.dropoff_longitude.max())

))



# COUNT UNIQUE DRIVERS: 50000

print('driver_id   count: {}'.format(len(strips.driver_id.unique())))

# COUNT UNIQUE PASSENGERS

print('driver_id   count: {}'.format(len(strips.passenger_id.unique())))

# DATETIME RANGE - 2014-03-01 00:00:00 to 2014-05-31 18:57:00

print('Datetime range: {} to {}'.format(strips.pickup_datetime.min(), 

                                        strips.dropoff_datetime.max()))

# CALCULATE TRIP DURATION IN MINUTES

duration = (strips['dropoff_datetime'] - strips['pickup_datetime']).dt.seconds / 60

strips = strips.assign(trip_duration = duration)



print('Trip duration in minutes: {} to {}'.format(

    strips.trip_duration.min(), strips.trip_duration.max()))

outliers=np.array([False]*len(strips))



y = np.array(strips.trip_duration)



#mark outliers  consider only trip_durations btw 1 and 60 minutes

outliers[y>60]=True 

outliers[y<1]=True

print('There are %d entries that have trip duration too long or too short'% sum(outliers))
#total of 

strips = strips.assign(outliers=outliers)

#drop outliers

strips_clean = strips[outliers==False]

print('There are %d entries that have trip duration too long or too short'% sum(strips_clean.outliers))
# NOW THE TRIP DURATION IS CLEAN

# AND ALL THE CALCULATIONS ARE BEING MADE ON SAMPLE DATAFRAME 
y = np.array(strips_clean.trip_duration)

plt.figure(figsize=(12,5))

plt.subplot(131)

plt.plot(range(len(y)),y,'.');plt.ylabel('trip_duration');plt.xlabel('index');plt.title('val vs. index')

plt.subplot(132)

sns.boxplot(y=strips_clean.trip_duration)

plt.subplot(133)

sns.distplot(y,bins=50, color="m");plt.yticks([]);plt.xlabel('trip_duration');plt.title('strips_clean');plt.ylabel('frequency')

#plt.hist(y,bins=50);
# Remove rides from away area

xlim = [-74.03, -73.77]

ylim = [40.63, 40.85]

strips_clean = strips_clean[(strips_clean.pickup_longitude> xlim[0]) & (strips_clean.pickup_longitude < xlim[1])]

strips_clean = strips_clean[(strips_clean.dropoff_longitude> xlim[0]) & (strips_clean.dropoff_longitude < xlim[1])]

strips_clean = strips_clean[(strips_clean.pickup_latitude> ylim[0]) & (strips_clean.pickup_latitude < ylim[1])]

strips_clean = strips_clean[(strips_clean.dropoff_latitude> ylim[0]) & (strips_clean.dropoff_latitude < ylim[1])]

longitude = list(strips_clean.pickup_longitude) + list(strips_clean.dropoff_longitude)

longitude = list(strips_clean.pickup_longitude) + list(strips_clean.dropoff_longitude)

latitude = list(strips_clean.pickup_latitude) + list(strips_clean.dropoff_latitude)

plt.figure(figsize = (10,10))

plt.plot(longitude,latitude,'.', alpha = 0.4, markersize = 0.05)

plt.show()
loc_strips_clean = pd.DataFrame()

loc_strips_clean['longitude'] = longitude

loc_strips_clean['latitude'] = latitude
from sklearn.cluster import KMeans

from sklearn.neighbors import KNeighborsClassifier



kmeans = KMeans(n_clusters=15, random_state=2, n_init = 5).fit(loc_strips_clean)

loc_strips_clean['label'] = kmeans.labels_
plt.figure(figsize = (10,10))

for label in loc_strips_clean.label.unique():

    plt.plot(loc_strips_clean.longitude[loc_strips_clean.label == label],loc_strips_clean.latitude[loc_strips_clean.label == label],

             '.', alpha = 0.3, markersize = 0.3)



plt.title('Clusters of New York')

plt.show()
strips_clean.head()
loc_strips_clean.head()
fig,ax = plt.subplots(figsize = (10,10))

for label in loc_strips_clean.label.unique():

    ax.plot(loc_strips_clean.longitude[loc_strips_clean.label == label],loc_strips_clean.latitude[loc_strips_clean.label == label],'.', alpha = 0.4, markersize = 0.1, color = 'gray')

    ax.plot(kmeans.cluster_centers_[label,0],kmeans.cluster_centers_[label,1],'o', color = 'r')

    ax.annotate(label, (kmeans.cluster_centers_[label,0],kmeans.cluster_centers_[label,1]), color = 'b', fontsize = 20)

ax.set_title('Cluster Centers')

plt.show()
# MANIPULATING DATE



# DAYS OF WEEK (DOW) MAPPING

dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# MONTHS MAPPING

mm_names = [

    'January', 'February', 'March', 'April']

# MONTH (pickup and dropoff)

strips_clean['mm_pickup'] = strips_clean.pickup_datetime.dt.month.astype(np.uint8)

strips_clean['mm_dropoff'] = strips_clean.dropoff_datetime.dt.month.astype(np.uint8)

# DOW

strips_clean['dow_pickup'] = strips_clean.pickup_datetime.dt.weekday.astype(np.uint8)

strips_clean['dow_dropoff'] = strips_clean.dropoff_datetime.dt.weekday.astype(np.uint8)

# DAY HOUR

strips_clean['hh_pickup'] = strips_clean.pickup_datetime.dt.hour.astype(np.uint8)

strips_clean['hh_dropoff'] = strips_clean.dropoff_datetime.dt.hour.astype(np.uint8)
## PICKUPS BY DIFFERENT TIMEFRAMES





# PICKUP COUNT DISTRIBUTION: HOUR OF DAY

plt.figure(figsize=(12,5))

data = strips_clean.groupby('hh_pickup').aggregate({'order_id':'count'}).reset_index()

sns.barplot(x='hh_pickup', y='order_id', data=data, palette="rocket")

plt.title('Pick-ups Hour Distribution')

plt.xlabel('Hour of Day, 0-23')

plt.ylabel('No of strips_clean made')





# PICKUP COUNT DISTRIBUTION: DOW

plt.figure(figsize=(12,5))

data = strips_clean.groupby('dow_pickup').aggregate({'order_id':'count'}).reset_index()

sns.barplot(x='dow_pickup', y='order_id', data=data, palette="rocket")

plt.title('Pick-ups Weekday Distribution')

plt.xlabel('Day of week')

plt.xticks(range(0,7), dow_names, rotation='horizontal')

plt.ylabel('No of strips_clean made')





# PICKUP COUNT DISTRIBUTION: MONTH

plt.figure(figsize=(12,5))

data = strips_clean.groupby('mm_pickup').aggregate({'order_id':'count'}).reset_index()

sns.barplot(x='mm_pickup', y='order_id', data=data, palette="rocket")

plt.title('Pick-up Month Distribution')

plt.xlabel('Month')

plt.xticks(range(0,3), mm_names[:3], rotation='horizontal')

plt.ylabel('No of strips_clean made')





# PICKUP HEATMAP: DOW X HOUR

plt.figure(figsize=(12,5))

sns.heatmap(data=pd.crosstab(strips_clean.dow_pickup, 

                             strips_clean.hh_pickup, 

                             values=strips_clean.order_id, 

                             aggfunc='count',

                             normalize='index'))

plt.title('Pickup heatmap, Day-of-Week vs. Day Hour')

plt.ylabel('Weekday') ; plt.xlabel('Day Hour, 0-23')

plt.yticks(range(0,7), dow_names[::-1], rotation='horizontal')



# PICKUP HEATMAP: MONTH X HOUR

plt.figure(figsize=(12,5))

sns.heatmap(data=pd.crosstab(strips_clean.mm_pickup, 

                             strips_clean.hh_pickup, 

                             values=strips_clean.order_id, 

                             aggfunc='count',

                             normalize='index'))

plt.title('Pickup heatmap, Month vs. Day Hour')

plt.ylabel('Month') ; plt.xlabel('Day Hour, 0-23')

plt.yticks(range(0,4), mm_names[:4][::-1], rotation='horizontal')



# PICKUP HEATMAP: MONTH X DOW

plt.figure(figsize=(12,5))

sns.heatmap(data=pd.crosstab(strips_clean.mm_pickup, 

                             strips_clean.dow_pickup, 

                             values=strips_clean.order_id, 

                             aggfunc='count',

                             normalize='index'))

plt.title('Pickup heatmap, Month vs. Day-of-Week')

plt.ylabel('Month') ; plt.xlabel('Weekday')

plt.xticks(range(0,7), dow_names, rotation='vertical')

plt.yticks(range(0,4), mm_names[:4][::-1], rotation='horizontal')

strips_clean.info()
print("Total number of cancelled orders file : ", 

      sorders['order_id'].count() - strips_clean['order_id'].count())

print("\n") 



geo_sample = loc_strips_clean.sample(20000)

geo_output = pd.DataFrame(geo_sample)

trips_output = pd.DataFrame(strips_clean)

orders_output = pd.DataFrame(sorders)







geo_output.to_csv('geo_sample.csv', index=False)

trips_output.to_csv('strips_clean.csv', index=False)

orders_output.to_csv('sorders.csv', index=False)