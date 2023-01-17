# import all packages and set plots to be embedded inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import glob

import os

import calendar

from os import path, getcwd, makedirs, listdir 



%matplotlib inline



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# reading in data for 2019

df = pd.concat([pd.read_csv(f, low_memory=False) for f in glob.glob('/kaggle/input/bikesharing/*.csv')], ignore_index = True, sort=False)
# starting with visual assessment first to get familiar with the DataFrame

df.head()
df.tail()
df.info(verbose=True, null_counts=True)
df.describe()
# what is the size of our overall data set

df.shape
# how many different bikes do we have

df['bike_id'].nunique()
# check if we have the same stations for the start end ending of the rentals

df['start_station_id'].nunique() == df['end_station_id'].nunique()
# checking for duplicates in the whole DataFrame

df.duplicated().sum()
# how many NaN values do we have

df.isnull().sum()
# copying original DataFrame into 'df_clean'

df_clean = df.copy()



# check if everything has been copied correctly

df_clean.shape == df.shape
#1 Drop the column rental_access_method since it mostly contains NaN values

df_clean.drop(columns=['rental_access_method', 'bike_share_for_all_trip'], inplace=True)



# testing if the column is gone

'rental_access_method' and 'bike_share_for_all_trip' not in df_clean.columns
#2 Fill NaN values for 'start_station_name' and 'end_station_name' by using 'start_station_id' as a dictionary



# creating dictionary with 'start_station_id' as key and 'start_station_name' as value; mapping the NaN values in 'start_station_name' against the dictionary

dict_id_start = dict(zip(df_clean['start_station_id'].unique(), df_clean['start_station_name'].unique()))

df_clean['start_station_name'] = df_clean['start_station_name'].fillna(df_clean['start_station_id'].map(dict_id_start))



# creating dictionary with 'end_station_id' as key and 'end_station_name' as value; mapping the NaN values in 'end_station_name' against the dictionary

dict_id_end = dict(zip(df_clean['end_station_id'].unique(), df_clean['end_station_name'].unique()))

df_clean['end_station_name'] = df_clean['end_station_name'].fillna(df_clean['end_station_id'].map(dict_id_end))



# testing if all NaN values are gone

df_clean['start_station_name'].isna().sum(), df_clean['end_station_name'].isna().sum()
#3 Drop all rows containing NaN values for bike_share_for_all_trip

df_clean.dropna(subset=['start_station_id', 'end_station_id'], inplace=True)



# testing if all NaN values are gone

df_clean[['start_station_id', 'end_station_id']].isna().sum() == 0
#4 Fix issue with numeric appendix for data in start_time and end_time

df_clean[['start_time', 'end_time']] = df_clean[['start_time', 'end_time']].apply(lambda x: x.str.replace(r'\..*', ''))



# visual assessment if the appendix is gone

df_clean[['start_time', 'end_time']].head()
#5 Convert start_time and end_time into proper date/time format

df_clean['start_time'] = pd.to_datetime(df_clean['start_time'], errors='coerce', format='%Y-%m-%d %H:%M:%S')

df_clean['end_time'] = pd.to_datetime(df_clean['end_time'], errors='coerce', format='%Y-%m-%d %H:%M:%S')



# testing if columns are now in date/time format

df_clean[['start_time', 'end_time']].dtypes == 'datetime64[ns]'
#6 Convert start_station_id, end_station_id, bike_id into a string data type

df_clean[['bike_id', 'start_station_id', 'end_station_id']] = df_clean[['bike_id', 'start_station_id', 'end_station_id']].astype(str)



# additional clean up of remaining decimal point

df_clean[['bike_id', 'start_station_id', 'end_station_id']] = df_clean[['bike_id', 'start_station_id', 'end_station_id']].apply(lambda x: x.str.replace(r'\..*', ''))



# testing if columns are now strings

df_clean[['bike_id', 'start_station_id', 'end_station_id']].dtypes == 'object'
# to confirm we have correct data types and no NaN values remaining

df_clean.info(verbose=True, null_counts=True)
# source: https://kanoki.org/2019/12/27/how-to-calculate-distance-in-python-and-pandas-using-scipy-spatial-and-distance-functions/

def haversine_vectorize(lon1, lat1, lon2, lat2):

 

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

 

    newlon = lon2 - lon1

    newlat = lat2 - lat1

 

    haver_formula = np.sin(newlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(newlon/2.0)**2

 

    dist = 2 * np.arcsin(np.sqrt(haver_formula ))

    km = 6367 * dist

    return km



# defining new column 'distance' with the measured distance in KM

df_clean['distance_km'] = haversine_vectorize(df_clean['start_station_longitude'], df_clean['start_station_latitude'], df_clean['end_station_longitude'], df_clean['end_station_latitude'])



df_clean.head()
df_clean['duration_min'] = df_clean['duration_sec']/60

df_clean['duration_min'].sample(5)
df_clean['start_weekday'] = df_clean['start_time'].dt.weekday.apply(lambda x: calendar.day_abbr[x])

df_clean['end_weekday'] = df_clean['end_time'].dt.weekday.apply(lambda x: calendar.day_abbr[x])

df_clean[['start_weekday', 'end_weekday']].sample(5)
df_clean['start_month'] = df_clean['start_time'].dt.month.apply(lambda x: calendar.month_abbr[x])

df_clean['end_month'] = df_clean['end_time'].dt.month.apply(lambda x: calendar.month_abbr[x])

df_clean[['start_time', 'start_month', 'end_time', 'end_month']].sample(5)
df_clean['start_hour'] = df_clean['start_time'].dt.strftime('%H')

df_clean['end_hour'] = df_clean['end_time'].dt.strftime('%H')

df_clean[['start_time', 'start_hour', 'end_time', 'end_hour']].sample(5)
# saving the cleaned master DataFrame

df_clean.to_csv('df_master.csv',index=False)
# first taking a look at the duration in particular

df_clean['duration_min'].describe()
# this data seems to be widely spread out

df_clean['duration_min'].min(), df_clean['duration_min'].mean(), df_clean['duration_min'].max()
# calculating the duration which occurs the most within the data set

mode = df_clean['duration_min'].mode().tolist()



plt.figure(figsize=(20,7))



# plotting the distribution of the duration without any axis limits first

plt.subplot(1,2,1)

bins = np.arange(0, df_clean['duration_min'].max()+1, 1)

plt.hist(data = df_clean, x = 'duration_min', bins = bins, color = 'royalblue')

plt.title('Duration without axis limit')

plt.xlabel('Duration in Seconds')

plt.ylabel('Absolute Frequency')



# since we have a widely spread distribution

plt.subplot(1,2,2)

bins = np.arange(0, df_clean['duration_min'].max()+1, 1)

plt.hist(data = df_clean, x = 'duration_min', bins = bins, color = 'royalblue')

plt.title('Duration with axis limit at 2500')

plt.xlabel('Duration in Seconds')

plt.ylabel('Absolute Frequency')

plt.xticks(np.arange(0, 65, 5))

plt.axvline(x=mode, color='r', lw=1)

plt.xlim(0, 60)



plt.tight_layout();
# doing some calculations for the relative frequency of week days first

week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

color_palette = ['royalblue', 'royalblue', 'royalblue', 'royalblue', 'royalblue', 'lightblue', 'lightblue']

rel_StartDay = (df_clean['start_weekday'].value_counts() / df_clean['start_weekday'].value_counts().sum()).reindex(week)

rel_EndDay = (df_clean['end_weekday'].value_counts() / df_clean['end_weekday'].value_counts().sum()).reindex(week)



plt.figure(figsize=(20,7))



# plotting relative frequency of each weekday for the start time

plt.subplot(1,2,1)

rel_StartDay.plot(kind='bar', color = color_palette, legend=False)

plt.title('Percentage of Bike-Rides per Weekday (Start time)')

plt.xlabel('Weekday')

plt.ylabel('Relative Frequency')

plt.xticks(rotation=360)



# plotting relative frequency of each weekday for the end time

plt.subplot(1,2,2)

rel_EndDay.plot(kind='bar', color = color_palette, legend=False)

plt.title('Percentage of Bike-Rides per Weekday (End time)')

plt.xlabel('Weekday')

plt.ylabel('Relative Frequency')

plt.xticks(rotation=360)



plt.tight_layout();
# getting all rides per month

month = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

year = df_clean['start_time'].dt.strftime('%B').value_counts().reindex(month)



# plotting rides per month for 2019

plt.figure(figsize=(15,8))

sns.lineplot(data=year, sort=False, )

plt.title('Rides per Month')

plt.xlabel('Month')

plt.ylabel('Absolute Frequency')

plt.xticks(month)



plt.tight_layout;
# defining colors

plt_palette = ['royalblue', 'lightblue']

sns_palette = {'Customer':'royalblue', 'Subscriber':'lightblue'}



plt.figure(figsize=(25,8))



# first plot showing average duration for each user type

plt.subplot(1,2,1)

df_clean.groupby('user_type').duration_min.mean().plot(kind='barh', color = plt_palette)

plt.title('Trip Duration per User Type')

plt.xlabel('Average Duration in Minutes')

plt.ylabel('User Type')



# second plot showing distribution of average duration for each user type

plt.subplot(1,2,2)

sns.boxplot(data = df_clean, x = 'duration_min', y = 'user_type', order = ['Subscriber', 'Customer'], palette = sns_palette)

plt.title('Distribution of Duration per User Type')

plt.xlabel('Average Duration in Minutes')

plt.ylabel('User Type')

plt.xlim(0,60)



plt.tight_layout;
# for simplicity we're going to focus on the top 5 stations across the data set

df_clean['start_station_name'].value_counts().head(5)
# define list of top 5 stations and create new DataFrame that only carries these

name_list = df_clean['start_station_name'].value_counts().head(5).index.tolist()

df_stations = df_clean[df_clean['start_station_name'].isin(name_list)]

order = ['Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']



palette = sns.color_palette('Set3', 5)



# plotting the absolute frequency for top 5 stations across all months

plt.figure(figsize=(15,8))

sns.countplot(data = df_stations, x = 'start_month', hue = 'start_station_name', order = order, palette = palette)

plt.title('Frequency of rides top 5 most visited stations throughout the year')

plt.xlabel('Month')

plt.ylabel('Absolute Frequency')

plt.legend(name_list, loc = 6, bbox_to_anchor = (1.0, 0.5), title = 'Station Name')



plt.tight_layout;
# setting the order according to the weekday

df_clean['start_weekday'] = pd.Categorical(df_clean['start_weekday'], categories=['Mon','Tue','Wed','Thu','Fri','Sat', 'Sun'], ordered=True)

plt.figure(figsize=(15,8))



plt.suptitle('Usage per hour/weekday for subscribers vs. customers')



# first heatmap for customers

plt.subplot(1, 2, 1)

df_customer = df_clean[df_clean['user_type'] == 'Customer'].groupby(['start_hour', 'start_weekday'])['bike_id'].size().reset_index()

df_customer = df_customer.pivot('start_hour', 'start_weekday', 'bike_id')

sns.heatmap(df_customer, cmap='Blues')



plt.title('Customers', y=1.015)

plt.xlabel('Day of Week')

plt.ylabel('Start Time (Hour)')



# second heatmap for subscribers

plt.subplot(1, 2, 2)

df_subscriber = df_clean[df_clean['user_type'] == 'Subscriber'].groupby(['start_hour', 'start_weekday'])['bike_id'].size().reset_index()

df_subscriber = df_subscriber.pivot('start_hour', 'start_weekday', 'bike_id')

sns.heatmap(df_subscriber, cmap='Blues')



plt.title('Subscribers', y=1.015)

plt.xlabel('Day of Week')

plt.ylabel('Start Time (Hour)')



plt.tight_layout;