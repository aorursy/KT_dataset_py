#importing necessary python packages



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Read the CSV file



#my_local_path = "B:/UPX docs/Machine Learning/Project_datasets/Project datasets modified/NYC Taxi Trip/NYC Taxi Trip/"

taxi_data = pd.read_csv('../input/train.csv')

taxi_data.head(5)
# Seperating the datetime stamp into two seperate columns for pickup_datetime and pickup_datetime



#taxi_data['dropoff_date']=pd.to_datetime(taxi_data['dropoff_datetime']).dt.date

#taxi_data['dropoff_time']=pd.to_datetime(taxi_data['dropoff_datetime']).dt.time

#taxi_data['pickup_date']=pd.to_datetime(taxi_data['pickup_datetime']).dt.date

#taxi_data['pickup_time']=pd.to_datetime(taxi_data['pickup_datetime']).dt.time

#taxi_data['pickup_datetime'] = pd.to_datetime(taxi_data.pickup_datetime) 
taxi_data['pickup_datetime'] = pd.to_datetime(taxi_data.pickup_datetime) 
taxi_data.head()
taxi_mod = taxi_data

taxi_mod.info()
from haversine import haversine
def calc_distance(df):

    pickup = (df['pickup_latitude'], df['pickup_longitude'])

    drop = (df['dropoff_latitude'], df['dropoff_longitude'])

    return haversine(pickup, drop) 
taxi_mod['distance'] = taxi_mod.apply(lambda x: calc_distance(x), axis = 1)
#Calculate the Speed using the trip_duration and distance data



taxi_mod['trip_duration_hrs']=taxi_mod['trip_duration']/3600
taxi_mod['SPEED']=taxi_mod['distance']/taxi_mod['trip_duration_hrs']
taxi_mod.head()
#taxi_mod1=taxi_mod
#taxi_mod1 = taxi_mod1.drop(columns=['id','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'])

#taxi_mod1.head()
import pandas_profiling

profile = pandas_profiling.ProfileReport(taxi_mod)

profile.to_file(outputfile="taxi_mod.html")
corr = taxi_mod.corr()

corr
sns.heatmap(corr,annot=True)

plt.show()
plt.figure(figsize=(5,8))

total = float(len(taxi_mod))

plt.subplot(2,1,1)

ax=sns.countplot(x='passenger_count', data=taxi_mod)

plt.ylabel('number of trips')

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format(height/total),

            ha="center")

    

plt.figure(figsize=(5,8))

plt.subplot(2,1,2)

bx=sns.countplot(x='vendor_id', data=taxi_mod)

plt.ylabel('number of trips')

for p in bx.patches:

    height = p.get_height()

    bx.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format(height/total),

            ha="center")

    

plt.show()
plt.figure(figsize=(5,4))

dx=sns.countplot(x='store_and_fwd_flag', data=taxi_mod)

for p in dx.patches:

    height = p.get_height()

    dx.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format(height/total),

            ha="center")

plt.show()

plt.figure(figsize=(10,8))

cx=sns.factorplot(x='store_and_fwd_flag', col='vendor_id', kind='count', data=taxi_mod);

plt.show()
#Adding the features which separates the pickup_datetime stamp into hour_of_day, month_of_date, day_of_week, day_of_month, day_of_week_num



taxi_mod['hour_of_day']=taxi_mod.pickup_datetime.dt.hour

taxi_mod['month_of_date'] = taxi_mod['pickup_datetime'].dt.month

taxi_mod['day_of_week'] = taxi_mod['pickup_datetime'].dt.weekday_name

taxi_mod['day_of_month'] = taxi_mod['pickup_datetime'].dt.day

taxi_mod['day_of_week_num'] = taxi_mod['pickup_datetime'].dt.dayofweek

taxi_mod.head()
plt.figure(figsize=(10,20))

plt.subplot(4,1,1)

sns.countplot(x='day_of_month', data=taxi_mod)

plt.ylabel('number of trips')

plt.subplot(4,1,2)

sns.countplot(x='hour_of_day', data=taxi_mod)

plt.ylabel('number of trips')

plt.subplot(4,1,3)

sns.countplot(x='day_of_week_num', data=taxi_mod)

plt.ylabel('number of trips')

plt.subplot(4,1,4)

sns.countplot(x='month_of_date', data=taxi_mod)

plt.ylabel('number of trips')

plt.show()
taxi_mod1=taxi_mod
#taxi_mod2=pd.concat([taxi_mod, taxi_mod1]).loc[taxi_mod.index.symmetric_difference(taxi_mod1.index)]

taxi_mod2=taxi_mod1.loc[(taxi_mod1['trip_duration'] >=3600 ) & (taxi_mod1['distance'] <= 1),['trip_duration','distance'] ].reset_index(drop=True)

sns.regplot(taxi_mod2['distance'], taxi_mod2.trip_duration)

taxi_mod2.info()

plt.show()
taxi_mod3=pd.concat([taxi_mod2, taxi_mod1]).loc[taxi_mod1.index.symmetric_difference(taxi_mod2.index)]

taxi_mod3.info()
taxi_mod4=taxi_mod3.loc[(taxi_mod3['trip_duration'] <= 18000) & (taxi_mod3['distance'] <= 100),['trip_duration','distance'] ].reset_index(drop=True)
taxi_mod4.info()
plt.figure(figsize=(20,10))

plt.scatter(taxi_mod4['distance'],taxi_mod4['trip_duration'],s=1, alpha=0.5)

plt.xlabel('Distance in Km/hr')

plt.ylabel('Trip Duation in seconds')

plt.show()
plt.figure(figsize=(20,10))

sns.lmplot(x='distance', y='trip_duration', data=taxi_mod4, aspect=2.5, scatter_kws={'alpha':0.2})

plt.xlabel('Distance in Km/hr')

plt.ylabel('Trip Duation in seconds')

plt.show()
group1 = taxi_mod3.groupby('hour_of_day').trip_duration.mean()

plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.pointplot(group1.index, group1.values,color="#3fbb3f")

plt.ylabel('trip_duration')



group2 = taxi_mod3.groupby('month_of_date').trip_duration.mean()

plt.subplot(2,2,2)

sns.pointplot(group2.index, group2.values,color="#3fbb3f")

plt.ylabel('trip_duration')





group3 = taxi_mod3.groupby('day_of_week_num').trip_duration.mean()

plt.subplot(2,2,3)

sns.pointplot(group3.index, group3.values,color="#3fbb3f")

plt.ylabel('trip_duration')



group4 = taxi_mod3.groupby('day_of_month').trip_duration.mean()

plt.subplot(2,2,4)

sns.pointplot(group4.index, group4.values,color="#3fbb3f")

plt.ylabel('trip_duration')





plt.show()
group5 = taxi_mod3.groupby('hour_of_day').distance.mean()

plt.figure(figsize=(25,15))

plt.subplot(2,2,1)

sns.pointplot(group5.index, group5.values,color="#bb7d3f")

plt.ylabel('distance')



group6 = taxi_mod3.groupby('month_of_date').distance.mean()

plt.subplot(2,2,2)

sns.pointplot(group6.index, group6.values,color="#bb7d3f")

plt.ylabel('distance')





group7 = taxi_mod3.groupby('day_of_week_num').distance.mean()

plt.subplot(2,2,3)

sns.pointplot(group7.index, group7.values,color="#bb7d3f")

plt.ylabel('distance')



group8 = taxi_mod3.groupby('day_of_month').distance.mean()

plt.subplot(2,2,4)

sns.pointplot(group8.index, group8.values,color="#bb7d3f")

plt.ylabel('distance')

plt.show()





plt.show()
group9 = taxi_mod3.groupby('hour_of_day').SPEED.mean()

plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.pointplot(group9.index, group9.values,color="#bb3f3f")

plt.ylabel('SPEED(Km/Hr)')



group10 = taxi_mod3.groupby('month_of_date').SPEED.mean()

plt.subplot(2,2,2)

sns.pointplot(group10.index, group10.values,color="#bb3f3f")

plt.ylabel('SPEED(Km/Hr)')





group11 = taxi_mod3.groupby('day_of_week_num').SPEED.mean()

plt.subplot(2,2,3)

sns.pointplot(group11.index, group11.values,color="#bb3f3f")

plt.ylabel('SPEED(Km/Hr)')



group12 = taxi_mod3.groupby('day_of_month').SPEED.mean()

plt.subplot(2,2,4)

sns.pointplot(group12.index, group12.values,color="#bb3f3f")

plt.ylabel('SPEED(Km/Hr)')

plt.show()





plt.show()
taxi_mod.info()
taxi_mod9 = taxi_mod1.drop(taxi_mod1[(taxi_mod1.distance < 1)&(taxi_mod1.trip_duration > 3600)].index)

taxi_mod9 = taxi_mod1.drop(taxi_mod1[(taxi_mod3['trip_duration'] >= 18000) | (taxi_mod1['distance'] >= 200)].index)
taxi_mod9.info()
#taxi_mod9.to_csv('B:/UPX docs/Machine Learning/Project_datasets/Project datasets modified/NYC Taxi Trip/NYC Taxi Trip/Taxi_new.csv')