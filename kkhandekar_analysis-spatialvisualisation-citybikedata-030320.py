#importing libraries

import pandas as pd

from pandas import Grouper

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import re

import calendar



# Geovisualization library

import folium

from folium.plugins import CirclePattern, HeatMap, HeatMapWithTime
url = '../input/citi-bike-data/citibike.csv'
data = pd.read_csv(url, header='infer')
data.shape
#checking for null/missing values

data.isna().sum()
#dropping index with null values

data = data.dropna()
#converting Birth Year from Float to Int

data['birth_year'] = data['birth_year'].apply(np.int64)
#dropping trip_id column

data = data.drop(columns='trip_id',axis=1)
data.head()
#dropping the index with gender code  = 0

data = data.drop(data[data.gender == 0].index)
# -- Processing the Start & End date time



#start time

data.start_time = pd.to_datetime(data.start_time, format='%Y-%m-%d %H:%M:%S')

data['start_year'] = data.start_time.apply(lambda x: x.year)

data['start_month'] = data.start_time.apply(lambda x: x.month)

data['start_week'] = data.start_time.apply(lambda x: x.week)

data['start_day'] = data.start_time.apply(lambda x: x.day)

data['start_hour'] = data.start_time.apply(lambda x: x.hour)



#end time

data.end_time = pd.to_datetime(data.end_time, format='%Y-%m-%d %H:%M:%S')

data['end_year'] = data.start_time.apply(lambda x: x.year)

data['end_month'] = data.end_time.apply(lambda x: x.month)

data['end_week'] = data.end_time.apply(lambda x: x.week)

data['end_day'] = data.end_time.apply(lambda x: x.day)

data['end_hour'] = data.end_time.apply(lambda x: x.hour)
# Create a function to categorize Gender

def gender_cat(gender):

    if gender == 1:

        return 'Male'

    elif gender == 2:

        return 'Female'

    else:

        return 'Unknown'
#Applying the function to the Gender Column

data['Gender_Cat'] = data['gender'].apply(gender_cat)
# Create a function to convert trip duration to minutes

def time_convert(secs):

  days = secs//86400

  hours = (secs - days*86400)//3600

  mins = (secs - days*86400 - hours*3600)//60

  return mins  

#Applying the function to the Trip Duration Column

data['trip_duration_min'] = data['trip_duration'].apply(time_convert)
#data backup

data_backup = data.copy()
# Plotting Biker Stats for the entire september month - 2013

bike_daily = pd.DataFrame(data.groupby('start_day').size())

bike_daily['MEAN'] = data.groupby('start_day').size().mean()

bike_daily['STD'] = data.groupby('start_day').size().std()



# Plot total accidents per day, UCL, LCL and moving-average

plt.figure(figsize=(18,6))

data.groupby('start_day').size().plot(label='Bikes per day')

bike_daily['MEAN'].plot(color='red', linewidth=2, label='Average', ls='--')

plt.title('Total bikes per day in Sep 2013', fontsize=16)

plt.xlabel('Day')

plt.xticks(np.arange(1,31))

plt.ylabel('Number of bikes')

plt.tick_params(labelsize=14)

plt.legend(prop={'size':12})

# Create a pivot table by crossing the hour by the day of the week and calculate the average number of bikes for each crossing

bikes_pivot_table = data.pivot_table(values='start_day', index='start_hour', columns='weekday', aggfunc=len)

bikes_pivot_table_date_count = data.pivot_table(values='start_day', index='start_hour', columns='weekday', aggfunc=lambda x: len(x.unique()))

bikes_average = bikes_pivot_table/bikes_pivot_table_date_count

bikes_average.columns = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat']



# Using seaborn heatmap

plt.figure(figsize=(8,8))

plt.title('Average number of bikes per hour and day of the week', fontsize=14)

plt.tick_params(labelsize=12)

sns.heatmap(bikes_average.round(), cmap='seismic', linecolor='grey',linewidths=0.1, cbar=False, annot=True, fmt=".0f")
# Plotting - Bikes per Day

plt.figure(figsize=(15,6))



data.groupby('start_hour').size().plot(label = 'Total Bikes in a Day')

plt.title('Number of Bikes in a Day', fontsize=16)

plt.xlabel('hours')

plt.xticks(np.arange(0,24))

plt.legend(prop={'size':12})

plt.tick_params(labelsize=12)
# Number of Bikes in a week - plot

plt.figure(figsize=(15,6))



data.groupby('weekday').size().plot(label = 'Bikes in a Week')

plt.title('Number of Bikes in a Week', fontsize=16)

plt.xlabel('days in a week')

plt.legend(prop={'size':12})

plt.tick_params(labelsize=12)
# Plotting - Bikes in a Month

plt.figure(figsize=(15,6))



data.groupby('start_week').size().plot(label = 'Bikes per Week')

plt.title('Number of Bikes per week in a Month', fontsize=16)

plt.xlabel('weeks')

plt.legend(prop={'size':12})

plt.tick_params(labelsize=12)
# Bikers Stats - Gender Category

bikes_stat_gender = data.pivot_table(values='gender', index='start_day', columns='Gender_Cat', aggfunc=len).plot(figsize=(15,6), linewidth=2)

plt.title('Bikers Stats per Day - Gender Category', fontsize=16)

plt.xlabel('Days')

plt.xticks(np.arange(1,31))

plt.legend(prop={'size':10})

plt.tick_params(labelsize=10)

# Male Bikers Heat Map

male_biker_df = data[data['gender'] == 1]

male_biker_pivot_table = male_biker_df.pivot_table(values='gender', index='start_hour', columns='weekday', aggfunc=len)

male_biker_pivot_table_date_count = male_biker_df.pivot_table(values='gender', index='start_hour', columns='weekday', aggfunc=lambda x: len(x.unique()))

male_biker_avg = male_biker_pivot_table/male_biker_pivot_table_date_count

male_biker_avg.columns = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat']



# Using seaborn heatmap

plt.figure(figsize=(8,8))

plt.title('Average number of male biker per hour and day of the week', fontsize=14)

plt.tick_params(labelsize=12)

sns.heatmap(male_biker_avg.round(), cmap='seismic', linecolor='grey',linewidths=0.1, cbar=False, annot=True, fmt=".0f")
#  Male Bikers in month of september 2013

male_biker_daily = pd.DataFrame(male_biker_df.groupby('start_day').size())

male_biker_daily['MEAN'] = male_biker_df.groupby('start_day').size().mean()

male_biker_daily['STD'] = male_biker_df.groupby('start_day').size().std()



plt.figure(figsize=(15,6))

male_biker_df.groupby('start_day').size().plot(label='Bikes per day')

male_biker_daily['MEAN'].plot(color='red', linewidth=2, label='Average', ls='--')

plt.title('Total Male biker per day in Sep 2013', fontsize=16)

plt.xlabel('Day')

plt.ylabel('Number of male bikers')

plt.tick_params(labelsize=10)

plt.xticks(np.arange(1,31))

plt.legend(prop={'size':10})
# Plotting - Male Biker in a Day

plt.figure(figsize=(15,6))



male_biker_df.groupby('start_hour').size().plot(label = 'Male Bikers in a Day')

plt.title('Number of Male Bikes in a Day', fontsize=16)

plt.xlabel('Hours')

plt.xticks(np.arange(0,24))

plt.legend(prop={'size':10})

plt.tick_params(labelsize=10)
# Male Bikers Average Trip Duration

male_biker_daily_trip = pd.DataFrame(male_biker_df.groupby('trip_duration_min').size())

male_biker_daily_trip['MEAN'] = male_biker_df.groupby('trip_duration_min').size().mean()

male_biker_daily_trip['STD'] = male_biker_df.groupby('trip_duration_min').size().std()



# Plot total accidents per day, UCL, LCL and moving-average

plt.figure(figsize=(15,6))

male_biker_df.groupby('trip_duration_min').size().plot(label='Trip Duration(mins)')

male_biker_daily_trip['MEAN'].plot(color='red', linewidth=2, label='Average', ls='--')

plt.title('Male biker average trip duration', fontsize=16)

plt.xlabel('Minutes')

plt.xticks(np.arange(1,45))

plt.ylabel('Number of Male Bikers')

plt.tick_params(labelsize=10)

plt.legend(prop={'size':10})
# Female Bikers Heat Map

female_biker_df = data[data['gender'] == 2]

female_biker_pivot_table = female_biker_df.pivot_table(values='gender', index='start_hour', columns='weekday', aggfunc=len)

female_biker_pivot_table_date_count = female_biker_df.pivot_table(values='gender', index='start_hour', columns='weekday', aggfunc=lambda x: len(x.unique()))

female_biker_avg = female_biker_pivot_table/female_biker_pivot_table_date_count

female_biker_avg.columns = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat']



# Using seaborn heatmap

plt.figure(figsize=(8,8))

plt.title('Average number of female biker per hour and day of the week', fontsize=14)

plt.tick_params(labelsize=10)

sns.heatmap(female_biker_avg.round(), cmap='seismic', linecolor='grey',linewidths=0.1, cbar=False, annot=True, fmt=".0f")
#  Female Bikers in month of September 2013

female_biker_daily = pd.DataFrame(female_biker_df.groupby('start_day').size())

female_biker_daily['MEAN'] = female_biker_df.groupby('start_day').size().mean()

female_biker_daily['STD'] = female_biker_df.groupby('start_day').size().std()



plt.figure(figsize=(15,6))

female_biker_df.groupby('start_day').size().plot(label='Bikes per day')

female_biker_daily['MEAN'].plot(color='red', linewidth=2, label='Average', ls='--')

plt.title('Total Female bikers per day in Sep 2013', fontsize=16)

plt.xlabel('Day')

plt.xticks(np.arange(1,31))

plt.ylabel('Number of bikes')

plt.tick_params(labelsize=10)

plt.legend(prop={'size':10})
# Plotting - Female Biker in a Day

plt.figure(figsize=(15,6))



female_biker_df.groupby('start_hour').size().plot(label = 'Female Bikers in a Day')

plt.title('Number of Female Bikers in a Day', fontsize=16)

plt.xlabel('Hours')

plt.xticks(np.arange(0,24))

plt.legend(prop={'size':10})

plt.tick_params(labelsize=10)
# Female Bikers Average Trip Duration

female_biker_daily_trip = pd.DataFrame(female_biker_df.groupby('trip_duration_min').size())

female_biker_daily_trip['MEAN'] = female_biker_df.groupby('trip_duration_min').size().mean()

female_biker_daily_trip['STD'] = female_biker_df.groupby('trip_duration_min').size().std()



# Plot total accidents per day, UCL, LCL and moving-average

plt.figure(figsize=(15,6))

female_biker_df.groupby('trip_duration_min').size().plot(label='Trip Duration(mins)')

female_biker_daily_trip['MEAN'].plot(color='red', linewidth=2, label='Average',ls='--')

plt.title('Female biker average trip duration', fontsize=16)

plt.xlabel('Minutes')

plt.xticks(np.arange(1,45))

plt.ylabel('Number of Female Bikers')

plt.tick_params(labelsize=10)

plt.legend(prop={'size':10})
# Centennials Bikers Analysis - Month

CentennialBikers_df = data[data['birth_year'] >= 1996]



CentennialBikers_daily = pd.DataFrame(CentennialBikers_df.groupby('start_day').size())

CentennialBikers_daily['MEAN'] = CentennialBikers_df.groupby('start_day').size().mean()



plt.figure(figsize=(15,6))



CentennialBikers_df.groupby('start_day').size().plot(label='Bikers per day')

CentennialBikers_daily['MEAN'].plot(color='red', linewidth=2, label='Average',ls='--')

plt.title('Daily Centennials Biker Analysis - Sep 2013', fontsize=16)

plt.xlabel('Days')

plt.ylabel('Number of bikers')

plt.xticks(np.arange(1,31))

plt.tick_params(labelsize=10)

plt.legend(prop={'size':10})



# Centennial Bikers Analysis - Week

plt.figure(figsize=(15,6))



CentennialBikers_df.groupby('weekday').size().plot(label = 'Bikes in a Week')

plt.title('Centennial Bikers Analysis - Week', fontsize=16)

plt.xlabel('days in a week')

plt.legend(prop={'size':12})

plt.tick_params(labelsize=12)
# Centennial Bikers Analysis - Day

plt.figure(figsize=(15,6))



CentennialBikers_df.groupby('start_hour').size().plot(label = 'Total Bikers in a Day')

plt.title('Centennial Bikers Analysis - Day', fontsize=16)

plt.xlabel('hours')

plt.xticks(np.arange(1,24))

plt.legend(prop={'size':10})

plt.tick_params(labelsize=10)
CentennialBikers_Gender = CentennialBikers_df.pivot_table(values='gender', index='start_day', columns='Gender_Cat', aggfunc=len).plot(figsize=(15,6), linewidth=2)

plt.title('Centennial Biker Gender Analysis', fontsize=16)

plt.xlabel('Days')

plt.xticks(np.arange(1,31))

plt.legend(prop={'size':10})

plt.tick_params(labelsize=10)
CentennialBikers_TDur = CentennialBikers_df.pivot_table(values='start_hour', index='trip_duration_min', columns='Gender_Cat', aggfunc=len).plot(figsize=(15,6), linewidth=2)

plt.title('Centennial Bikers Trip Duration Analysis', fontsize=16)

plt.xlabel('Mins')

plt.xticks(np.arange(1,45))

plt.legend(prop={'size':10})

plt.tick_params(labelsize=10)
# Millenials Bikers Analysis - Month

MillenialBikers_df = data[(data['birth_year'] >= 1977) & (data['birth_year'] <= 1995)]



MillenialBikers_daily = pd.DataFrame(MillenialBikers_df.groupby('start_day').size())

MillenialBikers_daily['MEAN'] = MillenialBikers_df.groupby('start_day').size().mean()



plt.figure(figsize=(15,6))



MillenialBikers_df.groupby('start_day').size().plot(label='Bikers per day')

MillenialBikers_daily['MEAN'].plot(color='red', linewidth=2, label='Average',ls='--')

plt.title('Daily Millenials Biker Analysis - Sep 2013', fontsize=16)

plt.xlabel('Days')

plt.ylabel('Number of bikers')

plt.xticks(np.arange(1,31))

plt.tick_params(labelsize=10)

plt.legend(prop={'size':10})



# Millenial Bikers Analysis - Week

plt.figure(figsize=(15,6))



MillenialBikers_df.groupby('weekday').size().plot(label = 'Bikes in a Week')

plt.title('Millenials Bikers Analysis - Week', fontsize=16)

plt.xlabel('days in a week')

plt.legend(prop={'size':12})

plt.tick_params(labelsize=12)
# MIllenial Bikers Analysis - Day

plt.figure(figsize=(15,6))



MillenialBikers_df.groupby('start_hour').size().plot(label = 'Total Bikers in a Day')

plt.title('Millenials Bikers Analysis - Day', fontsize=16)

plt.xlabel('hours')

plt.xticks(np.arange(1,24))

plt.legend(prop={'size':10})

plt.tick_params(labelsize=10)
MillenialBikers_Gender = MillenialBikers_df.pivot_table(values='gender', index='start_day', columns='Gender_Cat', aggfunc=len).plot(figsize=(15,6), linewidth=2)

plt.title('Millenial Bikers Gender Analysis', fontsize=16)

plt.xlabel('Days')

plt.xticks(np.arange(1,31))

plt.legend(prop={'size':10})

plt.tick_params(labelsize=10)
MillenialBikers_TDur = MillenialBikers_df.pivot_table(values='start_hour', index='trip_duration_min', columns='Gender_Cat', aggfunc=len).plot(figsize=(15,6), linewidth=2)

plt.title('Millenial Bikers Trip Duration Analysis', fontsize=16)

plt.xlabel('Mins')

plt.xticks(np.arange(1,45))

plt.legend(prop={'size':10})

plt.tick_params(labelsize=10)
# Gen X Bikers Analysis - Month

GenX_df = data[(data['birth_year'] >= 1965) & (data['birth_year'] <= 1976)]



GenX_daily = pd.DataFrame(GenX_df.groupby('start_day').size())

GenX_daily['MEAN'] = GenX_df.groupby('start_day').size().mean()



plt.figure(figsize=(15,6))



GenX_df.groupby('start_day').size().plot(label='Bikers per day')

GenX_daily['MEAN'].plot(color='red', linewidth=2, label='Average',ls='--')

plt.title('Daily Gen-X Biker Analysis - Sep 2013', fontsize=16)

plt.xlabel('Days')

plt.ylabel('Number of bikers')

plt.xticks(np.arange(1,31))

plt.tick_params(labelsize=10)

plt.legend(prop={'size':10})



# Gen-X Bikers Analysis - Week

plt.figure(figsize=(15,6))



GenX_df.groupby('weekday').size().plot(label = 'Bikes in a Week')

plt.title('Gen-X Bikers Analysis - Week', fontsize=16)

plt.xlabel('days in a week')

plt.legend(prop={'size':12})

plt.tick_params(labelsize=12)
# Gen-X Bikers Analysis - Day

plt.figure(figsize=(15,6))



GenX_df.groupby('start_hour').size().plot(label = 'Total Bikers in a Day')

plt.title('Gen-X Bikers Analysis - Day', fontsize=16)

plt.xlabel('hours')

plt.xticks(np.arange(1,24))

plt.legend(prop={'size':10})

plt.tick_params(labelsize=10)
GenXBikers_Gender = GenX_df.pivot_table(values='gender', index='start_day', columns='Gender_Cat', aggfunc=len).plot(figsize=(15,6), linewidth=2)

plt.title('Gen-X Bikers Gender Analysis', fontsize=16)

plt.xlabel('Days')

plt.xticks(np.arange(1,31))

plt.legend(prop={'size':10})

plt.tick_params(labelsize=10)
GenXBikers_TDur = GenX_df.pivot_table(values='start_hour', index='trip_duration_min', columns='Gender_Cat', aggfunc=len).plot(figsize=(15,6), linewidth=2)

plt.title('Gen-X Bikers Trip Duration Analysis', fontsize=16)

plt.xlabel('Mins')

plt.xticks(np.arange(1,45))

plt.legend(prop={'size':10})

plt.tick_params(labelsize=10)
# Baby Boomers Bikers Analysis - Month

BabyB_df = data[(data['birth_year'] >= 1946) & (data['birth_year'] <= 1964)]



BabyB_daily = pd.DataFrame(BabyB_df.groupby('start_day').size())

BabyB_daily['MEAN'] = BabyB_df.groupby('start_day').size().mean()



plt.figure(figsize=(15,6))



BabyB_df.groupby('start_day').size().plot(label='Bikers per day')

BabyB_daily['MEAN'].plot(color='red', linewidth=2, label='Average',ls='--')

plt.title('Daily Baby Boomer Biker Analysis - Sep 2013', fontsize=16)

plt.xlabel('Days')

plt.ylabel('Number of bikers')

plt.xticks(np.arange(1,31))

plt.tick_params(labelsize=10)

plt.legend(prop={'size':10})



# Baby Boomer Bikers Analysis - Week

plt.figure(figsize=(15,6))



BabyB_df.groupby('weekday').size().plot(label = 'Bikes in a Week')

plt.title('Baby Boomer Bikers Analysis - Week', fontsize=16)

plt.xlabel('days in a week')

plt.legend(prop={'size':10})

plt.tick_params(labelsize=10)
# Baby Boomer Bikers Analysis - Day

plt.figure(figsize=(15,6))



BabyB_df.groupby('start_hour').size().plot(label = 'Total Bikers in a Day')

plt.title('Baby Boomer Bikers Analysis - Day', fontsize=16)

plt.xlabel('hours')

plt.xticks(np.arange(1,24))

plt.legend(prop={'size':10})

plt.tick_params(labelsize=10)
BabyBBikers_Gender = BabyB_df.pivot_table(values='gender', index='start_day', columns='Gender_Cat', aggfunc=len).plot(figsize=(15,6), linewidth=2)

plt.title('Baby Boomer Bikers Gender Analysis', fontsize=16)

plt.xlabel('Days')

plt.xticks(np.arange(1,31))

plt.legend(prop={'size':10})

plt.tick_params(labelsize=10)
BabyBBikers_TDur = BabyB_df.pivot_table(values='start_hour', index='trip_duration_min', columns='Gender_Cat', aggfunc=len).plot(figsize=(15,6), linewidth=2)

plt.title('Baby Boomer Bikers Trip Duration Analysis', fontsize=16)

plt.xlabel('Mins')

plt.xticks(np.arange(1,45))

plt.legend(prop={'size':10})

plt.tick_params(labelsize=10)
data.head()
# Function to create a base map of NY



def generateBaseMap(default_location=[40.693943, -73.985880], default_zoom_start=12):

    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start,width='50%', height='50%')

    return base_map
#Calling the base map function

base_map = generateBaseMap()

base_map
# Centennial Bikers - Heatmap



CentennialBikers_df_copy = CentennialBikers_df.copy()

CentennialBikers_df_copy['count'] = 1

base_map = generateBaseMap()

HeatMap(data=CentennialBikers_df_copy[['start_station_latitude', 'start_station_longitude', 'count']].groupby(['start_station_latitude', 'start_station_longitude']).sum().reset_index().values.tolist(), radius=8, max_zoom=13).add_to(base_map)



#calling the function

base_map
# Millenial Bikers - Heatmap



MillenialBikers_df_copy = MillenialBikers_df.copy()

MillenialBikers_df_copy['count'] = 1

base_map = generateBaseMap()

HeatMap(data=MillenialBikers_df_copy[['start_station_latitude', 'start_station_longitude', 'count']].groupby(['start_station_latitude', 'start_station_longitude']).sum().reset_index().values.tolist(), radius=8, max_zoom=13).add_to(base_map)



#calling the function

base_map
# Gen-X Bikers - Heatmap



GenX_df_copy = GenX_df.copy()

GenX_df_copy['count'] = 1

base_map = generateBaseMap()

HeatMap(data=GenX_df_copy[['start_station_latitude', 'start_station_longitude', 'count']].groupby(['start_station_latitude', 'start_station_longitude']).sum().reset_index().values.tolist(), radius=8, max_zoom=13).add_to(base_map)



#calling the function

base_map
# Baby Boomer Bikers - Heatmap



BabyB_df_copy = BabyB_df.copy()

BabyB_df_copy['count'] = 1

base_map = generateBaseMap()

HeatMap(data=BabyB_df_copy[['start_station_latitude', 'start_station_longitude', 'count']].groupby(['start_station_latitude', 'start_station_longitude']).sum().reset_index().values.tolist(), radius=8, max_zoom=13).add_to(base_map)



#calling the function

base_map
# Overall Male Biker - Heatmap



male_biker_df_copy = male_biker_df.copy()

male_biker_df_copy['count'] = 1

base_map = generateBaseMap()

HeatMap(data=male_biker_df_copy[['start_station_latitude', 'start_station_longitude', 'count']].groupby(['start_station_latitude', 'start_station_longitude']).sum().reset_index().values.tolist(), radius=8, max_zoom=13).add_to(base_map)



#calling the function

base_map
# Overall Female Biker - Heatmap



female_biker_df_copy = female_biker_df.copy()

female_biker_df_copy['count'] = 1

base_map = generateBaseMap()

HeatMap(data=female_biker_df_copy[['start_station_latitude', 'start_station_longitude', 'count']].groupby(['start_station_latitude', 'start_station_longitude']).sum().reset_index().values.tolist(), radius=8, max_zoom=13).add_to(base_map)



#calling the function

base_map