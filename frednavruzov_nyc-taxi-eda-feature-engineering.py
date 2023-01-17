# library import

import pandas as pd # DataFrame support

import numpy as np # algebra / computations



import matplotlib.pyplot as plt # plotting

import seaborn as sns # fancier plotting

%matplotlib inline
# load train data

df_train = pd.read_csv(filepath_or_buffer='../input/train.csv', 

                       engine='c', 

                       infer_datetime_format=True, # to speed-up datetime parsing

                       parse_dates=[2,3] # specify datetime columns

                      )



df_train.store_and_fwd_flag = df_train.store_and_fwd_flag.astype('category')

# check data usage

print('Memory usage, Mb: {:.2f}\n'.format(df_train.memory_usage().sum()/2**20))



# overall df info

print('DataFrame Info: ---------------------')

print(df_train.info())



# optimize dtypes

df_train.passenger_count = df_train.passenger_count.astype(np.uint8)

df_train.vendor_id = df_train.vendor_id.astype(np.uint8)

df_train.trip_duration = df_train.trip_duration.astype(np.uint32)

for c in [c for c in df_train.columns if c.endswith('tude')]:

    df_train.loc[:, c] = df_train[c].astype(np.float32)



# now memory usage is cut by 50%

print('\nMemory usage (optimized), Mb: {:.2f}\n'.format(df_train.memory_usage().sum()/2**20))



# check sample output

df_train.head()
print(df_train.isnull().sum()) # nice, no N/A values
# check for duplicate ids - nice, no duplicates

print('No of Duplicates, Trip IDs: {}'.format(len(df_train) - 

                                              len(df_train.drop_duplicates(subset='id'))))



# check latitude/longitude bounds, Latitude: -85 to +85, Longitude: -180 to +180

print('Latitude bounds: {} to {}'.format(

    max(df_train.pickup_latitude.min(), df_train.dropoff_latitude.min()),

    max(df_train.pickup_latitude.max(), df_train.dropoff_latitude.max())

))

print('Longitude bounds: {} to {}'.format(

    max(df_train.pickup_longitude.min(), df_train.dropoff_longitude.min()),

    max(df_train.pickup_longitude.max(), df_train.dropoff_longitude.max())

))



# check trip duration - oops, looks like:

# 1) someone was on the road for 3526282sec ~ 40 days and forget to switch-off the counter, he-he

# 2) someone has invented quantum teleportation and made trips in 1-2 sec

# more closer look reveals some consecutive measurements, say, distance = 33ft, time=1sec

print('Trip duration in seconds: {} to {}'.format(

    df_train.trip_duration.min(), df_train.trip_duration.max()

))

# let's also check that trip_duration == drop-off time - pick-up time, nice, no errors

print("Incorrect trip duration's calculations: {}".format(

    (df_train.trip_duration != df_train.dropoff_datetime.sub(df_train.pickup_datetime, axis=0) 

     / np.timedelta64(1, 's')).sum())

)



# vendors cnt, only 2

print('Vendors cnt: {}'.format(len(df_train.vendor_id.unique())))

# datetime range - 6 full months, from January 2016 to June 2016

print('Datetime range: {} to {}'.format(df_train.pickup_datetime.min(), 

                                        df_train.dropoff_datetime.max()))



# passenger count - the common sense implies values between 1 and 10(Ford Transit), let's check

# zeroes, hmm...

print('Passengers: {} to {}'.format(df_train.passenger_count.min(), 

                                        df_train.passenger_count.max()))
# Since there are less than 10k rows with anomalies in trip_duration (in common sense), 

# we can safely remove them

duration_mask = ((df_train.trip_duration < 60) | # < 1 min

             (df_train.trip_duration > 3600*2)) # > 2 hours

print('Anomalies in trip duration, %: {:.2f}'.format(

    df_train[duration_mask].shape[0] / df_train.shape[0] * 100

))

df_train = df_train[~duration_mask]

df_train.trip_duration = df_train.trip_duration.astype(np.uint16)

# let's see range now

print('Trip duration in seconds: {} to {}'.format(

    df_train.trip_duration.min(), df_train.trip_duration.max()

))



# let's also drop trips with passenger count = 0, since there are only 17 of them

print('Empty trips: {}'.format(df_train[df_train.passenger_count == 0].shape[0]))

df_train = df_train[df_train.passenger_count > 0]
# Let's add some additional columns to speed-up calculations

# dow names for plot mapping

dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# mm names for plot mapping

mm_names = [

    'January', 'February', 'March', 'April', 'May', 'June', 

    'July', 'August', 'September', 'October', 'November', 'December'

]



# month (pickup and dropoff)

df_train['mm_pickup'] = df_train.pickup_datetime.dt.month.astype(np.uint8)

df_train['mm_dropoff'] = df_train.dropoff_datetime.dt.month.astype(np.uint8)

# day of week

df_train['dow_pickup'] = df_train.pickup_datetime.dt.weekday.astype(np.uint8)

df_train['dow_dropoff'] = df_train.dropoff_datetime.dt.weekday.astype(np.uint8)

# day hour

df_train['hh_pickup'] = df_train.pickup_datetime.dt.hour.astype(np.uint8)

df_train['hh_dropoff'] = df_train.dropoff_datetime.dt.hour.astype(np.uint8)
# pickup time distribution, hour-of-day

plt.figure(figsize=(12,2))



data = df_train.groupby('hh_pickup').aggregate({'id':'count'}).reset_index()

sns.barplot(x='hh_pickup', y='id', data=data)



plt.title('Pick-ups Hour Distribution')

plt.xlabel('Hour of Day, 0-23')

plt.ylabel('No of Trips made')

pass
# pickup distribution, by weekday

plt.figure(figsize=(12,2))



data = df_train.groupby('dow_pickup').aggregate({'id':'count'}).reset_index()

sns.barplot(x='dow_pickup', y='id', data=data)



plt.title('Pick-ups Weekday Distribution')

plt.xlabel('Trip Duration, minutes')

plt.xticks(range(0,7), dow_names, rotation='horizontal')

plt.ylabel('No of Trips made')

pass
# pickup distribution, by months

plt.figure(figsize=(12,2))



data = df_train.groupby('mm_pickup').aggregate({'id':'count'}).reset_index()

sns.barplot(x='mm_pickup', y='id', data=data)



plt.title('Pick-up Month Distribution')

plt.xlabel('Trip Duration, minutes')

plt.xticks(range(0,7), mm_names[:6], rotation='horizontal')

plt.ylabel('No of Trips made')

pass
# Pickup heatmap, dow vs hour

plt.figure(figsize=(12,2))

sns.heatmap(data=pd.crosstab(df_train.dow_pickup, 

                             df_train.hh_pickup, 

                             values=df_train.vendor_id, 

                             aggfunc='count',

                             normalize='index'))



plt.title('Pickup heatmap, Day-of-Week vs. Day Hour')

plt.ylabel('Weekday') ; plt.xlabel('Day Hour, 0-23')

plt.yticks(range(0,7), dow_names[::-1], rotation='horizontal')

pass
# Pickup heatmap, month vs hour

plt.figure(figsize=(12,2))

sns.heatmap(data=pd.crosstab(df_train.mm_pickup, 

                             df_train.hh_pickup, 

                             values=df_train.vendor_id, 

                             aggfunc='count',

                             normalize='index'))



plt.title('Pickup heatmap, Month vs. Day Hour')

plt.ylabel('Month') ; plt.xlabel('Day Hour, 0-23')

plt.yticks(range(0,7), mm_names[:7][::-1], rotation='horizontal')

pass
# Pickup heatmap, month vs dow

plt.figure(figsize=(12,2))

sns.heatmap(data=pd.crosstab(df_train.mm_pickup, 

                             df_train.dow_pickup, 

                             values=df_train.vendor_id, 

                             aggfunc='count',

                             normalize='index'))



plt.title('Pickup heatmap, Month vs. Day-of-Week')

plt.ylabel('Month') ; plt.xlabel('Weekday')

plt.xticks(range(0,7), dow_names, rotation='vertical')

plt.yticks(range(0,7), mm_names[:7][::-1], rotation='horizontal')

pass
# vendor pick-up hours density by weekdays

plt.figure(figsize=(12,6))

sns.violinplot(x=df_train.dow_pickup, 

               y=df_train.hh_pickup, 

               hue=df_train.vendor_id, 

               split=True)



plt.title('Vendor pick-up hours density, by weekday')

plt.xlabel('Weekday') ; plt.ylabel('Day Hour, 0-23')

plt.xticks(range(0,7), dow_names, rotation='horizontal')

pass
# trip duration distribution, minutes

plt.figure(figsize=(12,3))

plt.title('Trip Duration Distribution')

plt.xlabel('Trip Duration, minutes')

plt.ylabel('No of Trips made')

plt.hist(df_train.trip_duration/60, bins=100)

pass
# trip duration, based on hour-of-day vs. weekday

# Pickup heatmap, dow vs hour

plt.figure(figsize=(12,2))

sns.heatmap(data=pd.crosstab(df_train.dow_pickup, 

                             df_train.hh_pickup, 

                             values=df_train.trip_duration/60, 

                             aggfunc='mean',

                             ))



plt.title('Trip duration heatmap (Minutes), Day-of-Week vs. Day Hour')

plt.ylabel('Weekday') ; plt.xlabel('Day Hour, 0-23')

plt.yticks(range(0,7), dow_names[::-1], rotation='horizontal')

pass
# trip duration time-series by day, mean vs. median

data = df_train.groupby(df_train.pickup_datetime.dt.date).aggregate({'trip_duration':['mean', 'median']})/60

plt.figure(figsize=(12,4))

plt.title('Trip Duration Over Time (Mean vs. Median)')

plt.ylabel('Trip Duration, minutes') ; plt.xlabel('Timeline')

plt.plot(data)

plt.legend(['Mean', 'Median'])

pass
# trip duration over time, vendors comparison

# seems like they are almost equal

data = pd.crosstab(index=df_train.pickup_datetime.dt.date, 

                   columns=df_train.vendor_id, 

                   values=df_train.trip_duration/60, 

                   aggfunc='mean')

plt.figure(figsize=(12,4))

plt.title('Mean Trip Duration Over Time (by Vendors)')

plt.ylabel('Trip Duration, minutes') ; plt.xlabel('Timeline')

plt.plot(data)

plt.legend(['Vendor 1', 'Vendor 2'])

pass
# To BE CONTINUED ...