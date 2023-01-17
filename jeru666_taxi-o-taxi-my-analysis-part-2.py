import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Input data files are available in the "../input/" directory.

path = 'D:/BACKUP/Kaggle/New York City Taxi/Data/'

train_df = pd.read_csv('../input/train.csv')



#--- Let's peek into the data

print (train_df.head())
print (train_df.dtypes)
print (train_df['pickup_datetime'][0])

print (type(train_df['pickup_datetime'][0]))
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])

train_df['dropoff_datetime'] = pd.to_datetime(train_df['dropoff_datetime'])



#--- Now let us see the datatype of both those columns ---

print (train_df.dtypes)
train_df['pickup_month'] = train_df.pickup_datetime.dt.month.astype(np.uint8)

train_df['pickup_day'] = train_df.pickup_datetime.dt.weekday.astype(np.uint8)

train_df['pickup_hour'] = train_df.pickup_datetime.dt.hour.astype(np.uint8)



train_df['dropoff_month'] = train_df.dropoff_datetime.dt.month.astype(np.uint8)

train_df['dropoff_day'] = train_df.dropoff_datetime.dt.weekday.astype(np.uint8)

train_df['dropoff_hour'] = train_df.dropoff_datetime.dt.hour.astype(np.uint8)

print (train_df.head())
#--- Correlation between columns 'pickup_month' and 'dropoff_month' ---

print (train_df['pickup_month'].corr(train_df['dropoff_month']))
#--- Correlation between columns 'pickup_day' and 'dropoff_day' ---

print (train_df['pickup_day'].corr(train_df['dropoff_day']))
#--- Unique elements in column pickup_month ---

print (train_df['pickup_month'].unique())

import seaborn as sns   #--- I realized this library helps us in visualizing relations better between columns ---



data = train_df.groupby('pickup_month').aggregate({'id':'count'}).reset_index()

month_list=["Jan","Feb","Mar","Apr","May", "Jun"]

ax = sns.barplot(x='pickup_month', y='id', data=data)

ax.set_xticklabels(month_list)
data = train_df.groupby('pickup_day').aggregate({'id':'count'}).reset_index()

day_list = ["Mon","Tue","Wed","Thu","Fri", "Sat", "Sun"]

ax = sns.barplot(x='pickup_day', y='id', data=data)

ax.set_xticklabels(day_list)

data = train_df.groupby('pickup_hour').aggregate({'id':'count'}).reset_index()

sns.barplot(x='pickup_hour', y='id', data=data)
data = train_df.groupby('dropoff_month').aggregate({'id':'count'}).reset_index()

ax = sns.barplot(x='dropoff_month', y='id', data=data)

ax.set_xticklabels(month_list)
data = train_df.groupby('dropoff_day').aggregate({'id':'count'}).reset_index()

ax = sns.barplot(x='dropoff_day', y='id', data=data)

ax.set_xticklabels(day_list)
data = train_df.groupby('dropoff_hour').aggregate({'id':'count'}).reset_index()

sns.barplot(x='dropoff_hour', y='id', data=data)
print('Mean trip_duration over pickup_month')

print(train_df['trip_duration'].groupby(train_df['pickup_month']).mean())

print(' ')

mean_pickup_month = train_df['trip_duration'].groupby(train_df['pickup_month']).mean()

sns.barplot(month_list, mean_pickup_month)
print('Mean trip_duration over pickup_day')

print(train_df['trip_duration'].groupby(train_df['pickup_day']).mean())

print(' ')

mean_pickup_day = train_df['trip_duration'].groupby(train_df['pickup_day']).mean()

sns.barplot(day_list, mean_pickup_day)
print('Mean trip_duration over pickup_hour')

print(train_df['trip_duration'].groupby(train_df['pickup_hour']).mean())

mean_pickup_hour = train_df['trip_duration'].groupby(train_df['pickup_hour']).mean()

hour_list = []

for i in range(0, 24):

    hour_list.append(i)

    

sns.barplot(hour_list, mean_pickup_hour)
train_df[train_df.columns[1:]].corr()['trip_duration'][:-1]

#train_df.head()
sns.barplot(x='passenger_count', y='trip_duration', data = train_df)
mean_passenger_count = train_df['trip_duration'].groupby(train_df['passenger_count']).mean()

passenger_count_list = []

for i in range(0, 10):

    passenger_count_list.append(i)

    

sns.barplot(passenger_count_list, mean_passenger_count, data = train_df)

print (mean_passenger_count)
print (train_df['passenger_count'].unique())

print(train_df.groupby('passenger_count').passenger_count.count())