import time
import re

import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

import folium

from geopy.distance import geodesic
from geopy.distance import great_circle
from geopy.geocoders import Nominatim
# Jupyter Notebook Interactive 
from IPython.display import display
from ipywidgets import interact, widgets, IntSlider
from IPython.html.widgets import *
%matplotlib inline 
sns.set_style('dark') # Seaborn ploting style
pd.set_option('display.expand_frame_repr', False) # Expand the dataframe width to display all columns
# Import training dataset
train_dataset_path = "../input/nytaxi-dataset/train.csv"
main_df = pd.read_csv(train_dataset_path)
# Import testing dataset
test_dataset_path = "../input/nytaxi-dataset-taxi/test.csv"
test_df = pd.read_csv(test_dataset_path)
# Import weather dataset
weather_df_path = '../input/nytaxi-dataset/taxi_weather_data.csv'
weather_df = pd.read_csv(weather_df_path)
print('main_df shape: \n{}\n'.format(main_df.shape))
print('test_df shape: \n{}\n'.format(test_df.shape))
print('weather_df shape: \n{}\n'.format(weather_df.shape))
print('test_df is: {}% of the main_df'.format(round(len(test_df) / len(main_df), 2) * 100))
print('main_df columns: \n{}\n'.format(list(main_df.columns)))
print('test_df columns: \n{}\n'.format(list(test_df.columns)))
print('weather_df columns: \n{}\n'.format(list(weather_df.columns)))
print('main_df columns types: \n{}\n'.format(main_df.dtypes))
print('test_df columns types: \n{}\n'.format(test_df.dtypes))
print('weather_df columns types: \n{}\n'.format(weather_df.dtypes))
print('main_df columns types: \n{}\n'.format(main_df.info()))
print('test_df columns types: \n{}\n'.format(test_df.info()))
print('weather_df columns types: \n{}\n'.format(weather_df.info()))
print('main_df columns types: \n{}\n'.format(main_df.isnull().sum()))
print('test_df columns types: \n{}\n'.format(test_df.isnull().sum()))
print('weather_df columns types: \n{}\n'.format(weather_df.isnull().sum()))
print('main_df statistics: \n{}\n'.format(main_df.describe()))
print('test_df statistics: \n{}\n'.format(test_df.describe()))
print('weather_df statistics: \n{}\n'.format(weather_df.describe()))
print('main_df categorical statistics: \n{}\n'.format(main_df.describe(include=['object']))) # Display categorical (dtype = 'object') statistics
print('test_df categorical statistics: \n{}\n'.format(test_df.describe(include=['object']))) 
print('weather_df categorical statistics: \n{}\n'.format(weather_df.describe(include=['object']))) 
print('main_df heaoverview: \n{}\n'.format(main_df.head()))
print('test_df overview: \n{}\n'.format(test_df.head()))
print('weather_df overview: \n{}\n'.format(weather_df.head()))
# Display the distribution of the trip duration with the mean/ median/ 25% & 75% percentiles
plt.figure(figsize=(50, 10))

sns.distplot(main_df.trip_duration.values, bins= 100)

plt.axvline(main_df.trip_duration.mean())
plt.axvline(main_df.trip_duration.median(), linestyle = '--', c = 'r')
plt.axvline(main_df.trip_duration.quantile(0.25), linestyle = ':', c = 'g')
plt.axvline(main_df.trip_duration.quantile(0.75), linestyle = ':', c = 'g')
plt.margins(0.02)

plt.show()
# Visualize the trip duration distribution

fig, ax = plt.subplots(ncols=2, figsize= (20,6))
sns.distplot(main_df.trip_duration.values, bins= 100, ax= ax[0])
sns.distplot(np.log(main_df.trip_duration.values), bins= 100, ax= ax[1])

ax[0].set_title('Trip Duration')
ax[0].set_xlabel('Trip Duration Samples')

ax[1].set_title('Log (Trips Count)')
ax[1].set_xlabel('Trip Duration Samples')

plt.show()
# Focus on the main_df.trip_duration feature

def trip_duration_focus():
    print('Trip Duration Count: {}'.format(len(main_df.trip_duration)))
    print('Trip Duration Max: {}'.format(main_df.trip_duration.max()))
    print('Trip Duration Min: {}'.format(main_df.trip_duration.min()))
    print('Trip Duration Mean: {}'.format(main_df.trip_duration.mean()))
    print('Trip Duration Variance: {}'.format(main_df.trip_duration.var()))
    print('Trip Duration Standard Deviation: {}'.format(main_df.trip_duration.std()))
    print('Trip Duration Interquartile Range: {}'.format(stats.iqr(main_df.trip_duration)))
    print('Trip Duration For The First 5 Values: {}'.format(main_df.trip_duration.sort_values().head().values))
    print('Trip Duration For The Last 5 Values: {}'.format(main_df.trip_duration.sort_values().tail().values))
    
trip_duration_focus()
# Decide on which approach to take for initial outliers clearning 

# Removing outliers based on mean & 2 * std
m, s = main_df.trip_duration.mean(), main_df.trip_duration.std()
mean_main_df = main_df[(main_df.trip_duration <= m + 2 * s) & (main_df.trip_duration >= m - 2 * s)].copy()

before_df, after_df = len(main_df.trip_duration), len(mean_main_df.trip_duration)
print('Mean (2 * STD) approach - Original dataset lenght is: {}. New dataset length is: {}. Diference count is: {} - {:.2f}%'.format(before_df, after_df, before_df - after_df, round((before_df - after_df) / before_df, 3) * 100))

# Remove outliers based on median & 1.5 * iqr
q1, q2, iqr = np.percentile(main_df.trip_duration, 25), np.percentile(main_df.trip_duration, 75), stats.iqr(main_df.trip_duration)
median_main_df = main_df[(main_df.trip_duration >= q1 - (iqr * 1.5)) & (main_df.trip_duration <= q2 + (iqr * 1.5))].copy()

before_df, after_df = len(main_df.trip_duration), len(median_main_df.trip_duration)
print('Median (1.5 * IQR) approach - Original dataset lenght is: {}. New dataset length is: {}. Diference count is: {} - {:.2f}%'.format(before_df, after_df, before_df - after_df, round((before_df - after_df) / before_df, 3) * 100))

# Remove outliers based on median & 2 * iqr
median_2_main_df = main_df[(main_df.trip_duration >= q1 - (iqr * 2)) & (main_df.trip_duration <= q2 + (iqr * 2))].copy()

before_df, after_df = len(main_df.trip_duration), len(median_2_main_df.trip_duration)
print('Median (2 * IQR) approach - Original dataset lenght is: {}. New dataset length is: {}. Diference count is: {} - {:.2f}%\n\n'.format(before_df, after_df, before_df - after_df, round((before_df - after_df) / before_df, 3) * 100))

# Visualization of the trip duration distribution when using log / mean & std / median & iqr to make a decision on which approach to take.
bp_bins = int(np.sqrt(len(main_df.trip_duration)))

fig, ax = plt.subplots(ncols=5, figsize= (45,6))
sns.distplot(main_df.trip_duration.values, bins= bp_bins, ax= ax[0])
sns.distplot(main_df.trip_duration.values[:5000], bins= bp_bins, ax= ax[1])
sns.distplot(mean_main_df.trip_duration.values, bins= 100, ax= ax[2])
sns.distplot(median_main_df.trip_duration.values, bins= 100, ax= ax[3])
sns.distplot(median_2_main_df.trip_duration.values, bins= 100, ax= ax[4])

ax[0].set_title('Normal - Trip Duration Distribution')
ax[0].set_xlabel('Trip Duration Samples')

ax[1].set_title('Normal (5000 Samples Zoom In) - Trip Duration Distribution')
ax[1].set_xlabel('Trip Duration Samples')
ax[1].set_xlim(0,5000)

ax[2].set_title('(Mean -/+ 2 * STD) - Trip Duration Distribution')
ax[2].set_xlabel('Logged Trip Duration Samples')

ax[3].set_title('(Median -/+ 1.5 * IQR) - Trip Duration Distribution')
ax[3].set_xlabel('Logged Trip Duration Samples')

ax[4].set_title('(Median -/+ 2 * IQR) - Trip Duration Distribution')
ax[4].set_xlabel('Logged Trip Duration Samples')

plt.show()
# Delete all unwanted objects
del median_2_main_df, median_main_df, mean_main_df, q1, q2, iqr, m, s, before_df, after_df, bp_bins
# Exclude the outliers

before_df = len(main_df.trip_duration)

m = main_df.trip_duration.mean()
s = main_df.trip_duration.std()

main_df = main_df[main_df.trip_duration <= m + 2 * s]
main_df = main_df[main_df.trip_duration >= m - 2 * s]

after_df = len(main_df.trip_duration)

# About 2000 rows were removed from the original dataset
print('Original dataset lenght is: {}. New dataset length is: {}. Diference is: {}'.format(before_df, after_df, before_df - after_df))

# Check the trip duration again
trip_duration_focus()
# Excluding trips that are less than 60 seconds

less_60_df = len(main_df[main_df.trip_duration < 60]) 

main_df = main_df[~(main_df.trip_duration < 60)]
after_df = len(main_df.trip_duration) 

# About 8562 rows were removed from the original dataset
print('Original dataset lenght is: {}. New dataset length is: {}. Diference is: {}'.format(less_60_df, after_df, after_df - less_60_df))

# Check the trip duration again
trip_duration_focus()
# Remove [passenger_count == 0 / 7 / 9]

print('length of the main_df where the [passenger_count == 0 / 7 / 8 / 9] is: {}\n'.format(len(main_df.loc[(main_df.passenger_count == 0) | (main_df.passenger_count == 7) | (main_df.passenger_count == 8) | (main_df.passenger_count == 9)])))
print('Passengers distribution is: \n{}\n'.format(main_df.passenger_count.value_counts()))
print('Passengers distribution is: \n{}\n'.format(main_df.loc[(main_df.passenger_count == 0) | (main_df.passenger_count == 7) | (main_df.passenger_count == 8) | (main_df.passenger_count == 9), ['pickup_datetime', 'passenger_count', 'trip_duration']].head().sort_values('trip_duration', ascending = True)))

# Delete the rows where [passenger_count == 0 / 7 / 9]
main_df = main_df.loc[~((main_df.passenger_count == 0) | (main_df.passenger_count == 7) | (main_df.passenger_count == 8) | (main_df.passenger_count == 9))].copy() 
# Visualize the updated trip duration distribution

fig, ax = plt.subplots(ncols=2, figsize= (20,6))
sns.distplot(main_df.trip_duration.values, bins= 100, ax= ax[0])
sns.distplot(np.log(main_df.trip_duration.values), bins= 100, ax= ax[1])

ax[0].set_title('Trip Duration Distribution')
ax[0].set_xlabel('Trip Duration Samples')

ax[1].set_title('Transformed - Log (Trip Duration)')
ax[1].set_xlabel('Transformed Trip Duration Samples')

plt.show()
# Convert main_df dates columns into datetime type in all dataframes
main_df.pickup_datetime = pd.to_datetime(main_df.pickup_datetime)
main_df.dropoff_datetime = pd.to_datetime(main_df.dropoff_datetime)
test_df.pickup_datetime= pd.to_datetime(test_df.pickup_datetime)
weather_df.pickup_datetime = pd.to_datetime(weather_df.pickup_datetime)

# Check if they were converted
print('datetime type columns in main_df, pickup_datetime: {}  and dropoff_datetime: {}.\nTotal datetime dtypes are: {}\n'.format(main_df.pickup_datetime.dtype, main_df.dropoff_datetime.dtype, sum(main_df.dtypes == 'datetime64[ns]')))
print('datetime type columns in test_df, pickup_datetime: {}.\nTotal datetime dtypes are: {}\n'.format(test_df.pickup_datetime.dtype, sum(test_df.dtypes == 'datetime64[ns]')))
print('datetime type columns in weather_df, pickup_datetime: {}.\nTotal datetime dtypes are: {}\n'.format(weather_df.pickup_datetime.dtype, sum(weather_df.dtypes == 'datetime64[ns]')))
# Function to convert 'int64' to 'int32'
def convert_int64_32(df):
    for i in df.columns:
        if df[i].dtype == 'int64':
            df[i] = df[i].astype('int32')
convert_int64_32(main_df)
convert_int64_32(test_df)
convert_int64_32(weather_df)
# Function to convert 'float64' to 'float32'
def convert_float64_32(df):
    for i in df.columns:
        if df[i].dtype == 'float64':
            df[i] = df[i].astype('float32')
convert_float64_32(main_df)
convert_float64_32(test_df)
convert_float64_32(weather_df)
# Convert the store_flag_dict to 1 & 0
store_flag_dict = {'Y': 1, 'N': 0}
main_df.store_and_fwd_flag.replace(store_flag_dict, inplace=True)
main_df.store_and_fwd_flag = main_df.store_and_fwd_flag.astype('int32')
# Sort dataframe based on datetime (year > month > day > time)
def reorganize_df_firstround(df):
    df.sort_values('pickup_datetime', inplace= True) # Sort dataframe based on datetime
    df.reset_index(inplace=True) # Reset indexe dataframes 
    df.drop('index', axis= 1, inplace= True) # Dropping unneccesary columns
    df.drop('id', axis= 1, inplace= True) # Dropping unneccesary columns
    
# Reset the index
def reorganize_df_comman(df):
    df.reset_index(inplace=True) # Reset indexe dataframes 
    df.drop('index', axis= 1, inplace= True) # Dropping unneccesary columns
reorganize_df_firstround(main_df) # Calling reorganize dataframe function on main_df
reorganize_df_firstround(test_df) # Calling reorganize dataframe function on test_df
# check dates (pickup/ dropoff) if they are not the same while the pickup time is less than 23:59
len(main_df.loc[(main_df.pickup_datetime.dt.date != main_df.dropoff_datetime.dt.date) & (main_df.pickup_datetime.dt.hour <= 23) & (main_df.dropoff_datetime.dt.minute <= 59)])
# Check initial main_df dataset cleaning effect
print('main_df new length: {}\n'.format(len(main_df))) # original length is: 1458644
print('main_df new info: {}\n'.format(main_df.info())) # original size is: 122+ MB
# Histogram Plotting

plt.figure(figsize=(12,6))
#bp_bins = int(np.sqrt(len(main_df.trip_duration)))

plt.hist(main_df.trip_duration, bins= 100)
plt.xlabel('Trip Duration Samples')
plt.ylabel('Count of Trip')
plt.title('Trip Duration Histogram')
plt.margins(0.02)

plt.show()
# ECDF Plotting

plt.figure(figsize=(12,6))

x = np.sort(main_df.trip_duration.values)
y = range(len(main_df))

plt.plot(x, y, marker = '.', linestyle = 'none')
plt.xlabel('Trip Duration Samples')
plt.ylabel('Count of Trip Duration')
plt.title('Trip Duration Histogram')
plt.margins(0.02)

plt.show()

# Alternative plotting code using Seaborn library
# sns.scatterplot(x = range(main_df.shape[0]), y = np.sort(main_df.trip_duration.values), data= main_df)
# ECDF Plotting with Percentiles

plt.figure(figsize=(12,6))

percentiles = np.array([10, 25, 50, 75, 99])
percentiles_vars = np.percentile(main_df.trip_duration, percentiles)

x = np.sort(main_df.trip_duration.values)
y = np.arange(1, len(main_df.trip_duration) + 1) / len(main_df.trip_duration)

plt.plot(x, y, '--')
plt.plot(percentiles_vars, percentiles / 100, marker = 'o', markersize = 10, linestyle = 'none')

plt.xlabel('Trip Duration Samples')
plt.ylabel('Count of Trip Duration')
plt.title('Trip Duration Histogram')
plt.margins(0.02)

plt.show()
plt.figure(figsize=(12,6))

sns.boxplot(y = np.log(main_df.trip_duration), data= main_df)
plt.show()
print('trip_duration minimum is: {}'.format(main_df.trip_duration.min()))
print('trip_duration maximum is: {}'.format(main_df.trip_duration.max()))
print('trip_duration mean is: {}'.format(round(main_df.trip_duration.mean(), 2)))
print('trip_duration variance is: {}'.format(round(main_df.trip_duration.var(), 2)))
print('trip_duration std is: {}'.format(round(main_df.trip_duration.std(), 2)))
print('trip_duration median is: {}'.format(round(main_df.trip_duration.median(), 2)))
print('trip_duration Interquartile Range (IQR) is: {}'.format(round(stats.iqr(main_df.trip_duration), 2)))

print('log(trip_duration) variance is: {}'.format(round(np.log(main_df.trip_duration).var(), 2)))
# Display scatter plot using pickup/ dropoff lat and long points as given in the training dataset

plt.style.use('dark_background')
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(15,10))

main_df.plot(kind='scatter', x='pickup_longitude', y='pickup_latitude', color='yellow', s=.02, alpha=.6, subplots=True, ax=ax1)
ax1.set_title("Pickups", color= 'white')
ax1.axes.tick_params(color= 'white', labelcolor= 'white')
ax1.set_facecolor('black')

main_df.plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude', color='yellow', s=.02, alpha=.6, subplots=True, ax=ax2)
ax2.set_title("Dropoffs", color= 'white')
ax2.axes.tick_params(color= 'white', labelcolor= 'white')
ax2.set_facecolor('black') 
# .copy() used to avoid warning while doing operations on the new dataframe
# Display scatter plot using pickup/ dropoff lat and long points for NY city block only

west, south, east, north = -74.03, 40.63, -73.77, 40.85 # NY city block

# Create new dataframe excluding the points that are not within the NY city block range
filtered_main_df = main_df.loc[(main_df.pickup_longitude > west) & (main_df.pickup_longitude < east) & 
                               (main_df.dropoff_longitude > west) & (main_df.dropoff_longitude < east) & 
                               (main_df.pickup_latitude < north) & (main_df.pickup_latitude > south) & 
                               (main_df.dropoff_latitude < north) & (main_df.dropoff_latitude > south)].copy()

reorganize_df_comman(filtered_main_df) # Reset the index

# Check df before/after lengths 
print('New dataset size is: \n{}\n'.format(len(filtered_main_df)))
print('The new dataset is less by: \n{}\n'.format(len(main_df) - len(filtered_main_df)))
# NY city area borders
borders = { 'manhattan':{ 'min_lng':-74.0479, 'min_lat':40.6829, 'max_lng':-73.9067, 'max_lat':40.8820 },
            'queens':{ 'min_lng':-73.9630, 'min_lat':40.5431, 'max_lng':-73.7004, 'max_lat':40.8007 },
            'brooklyn':{ 'min_lng':-74.0421, 'min_lat':40.5707, 'max_lng':-73.8334, 'max_lat':40.7395 },
            'bronx':{ 'min_lng':-73.9339, 'min_lat':40.7855, 'max_lng':-73.7654, 'max_lat':40.9176 },
            'staten_island':{ 'min_lng':-74.2558, 'min_lat':40.4960,  'max_lng':-74.0522, 'max_lat':40.6490 },
            'airport_JFK':{ 'min_lng':-73.8352, 'min_lat':40.6195, 'max_lng':-73.7401, 'max_lat':40.6659},
            'airport_EWR':{ 'min_lng':-74.1925, 'min_lat':40.6700, 'max_lng':-74.1531, 'max_lat':40.7081 },
            'airport_LaGuardia':{ 'min_lng':-73.8895, 'min_lat':40.7664, 'max_lng':-73.8550, 'max_lat':40.7931 } }

# Function - Differentiate areas based on LAT and LON
def points_classifier(lat,lng):
    if lat >= borders['manhattan']['min_lat'] and lat <= borders['manhattan']['max_lat'] and lng >= borders['manhattan']['min_lng'] and lng <= borders['manhattan']['max_lng']:
        return 'Manhattan'
    elif lat >= borders['queens']['min_lat'] and lat <= borders['queens']['max_lat'] and lng >= borders['queens']['min_lng'] and lng <= borders['queens']['max_lng']:
        return 'Queens'
    elif lat >= borders['brooklyn']['min_lat'] and lat <= borders['brooklyn']['max_lat'] and lng >= borders['brooklyn']['min_lng'] and lng <= borders['brooklyn']['max_lng']:
        return 'Brooklyn'
    elif lat >= borders['bronx']['min_lat'] and lat <= borders['bronx']['max_lat'] and lng >= borders['bronx']['min_lng'] and lng <= borders['bronx']['max_lng']:
        return 'Bronx'
    elif lat >= borders['staten_island']['min_lat'] and lat <= borders['staten_island']['max_lat'] and lng >= borders['staten_island']['min_lng'] and lng <= borders['staten_island']['max_lng']:
        return 'Staten Island'
    else:
        return 'Unknown'

# Function - Differentiate airports from cities
def is_airport(lat, lng):
    if lat >= borders['airport_JFK']['min_lat'] and lat <= borders['airport_JFK']['max_lat'] and lng >= borders['airport_JFK']['min_lng'] and lng <= borders['airport_JFK']['max_lng']:
        return 'JFK Airport'
    elif lat >= borders['airport_EWR']['min_lat'] and lat <= borders['airport_EWR']['max_lat'] and lng >= borders['airport_EWR']['min_lng'] and lng <= borders['airport_EWR']['max_lng']:
        return 'EWR Airport'
    elif lat >= borders['airport_LaGuardia']['min_lat'] and lat <= borders['airport_LaGuardia']['max_lat'] and lng >= borders['airport_LaGuardia']['min_lng'] and lng <= borders['airport_LaGuardia']['max_lng']:
        return 'La Guardia Aiport'
    else:
        return 'City'
    
# Create new columns for pickup and dropoff area 
filtered_main_df['pickup_area'] = filtered_main_df.apply(lambda x: points_classifier(x['pickup_latitude'], x['pickup_longitude']), axis = 1)
filtered_main_df['dropoff_area'] = filtered_main_df.apply(lambda x: points_classifier(x['dropoff_latitude'], x['dropoff_longitude']), axis = 1)

# Create new column to differentiate airports from cities
filtered_main_df['airport_pickup'] = filtered_main_df.apply(lambda x: is_airport(x['pickup_latitude'], x['pickup_longitude']), axis = 1)
filtered_main_df['airport_dropoff'] = filtered_main_df.apply(lambda x: is_airport(x['dropoff_latitude'], x['dropoff_longitude']), axis = 1)
filtered_main_df.head()
plt.style.use('dark_background')
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(15,10))

filtered_main_df.plot(kind='scatter', x='pickup_longitude', y='pickup_latitude', color='yellow', s=.02, alpha=.6, subplots=True, ax=ax1)
ax1.set_title("Pickups", color= 'white')
ax1.axes.tick_params(color= 'white', labelcolor= 'white')
ax1.set_facecolor('black')

filtered_main_df.plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude', color='orange', s=.02, alpha=.6, subplots=True, ax=ax2)
ax2.set_title("Dropoffs", color= 'white')
ax2.axes.tick_params(color= 'white', labelcolor= 'white')
ax2.set_facecolor('black') 
# Convert the pickup LAT & LON to array (float32) to plot points on map  
pickup_loc = np.array(filtered_main_df[['pickup_latitude', 'pickup_longitude']], dtype= 'float32') # Has to be 'pickup_latitude', 'pickup_longitude' for the map plotting
print(len(pickup_loc) - 1420792)
# Best views are: 'Stamen Terrain', 'Stamen Toner', 'Cartodb Positron'
# Due to the computing time, I will plot 2000 points only.
nymap = folium.Map(location=[40.745208740234375, -73.98473358154298], zoom_start= 12, control_scale= True, tiles='Cartodb dark_matter')

for i in range( 0, 2000):
    folium.CircleMarker(pickup_loc[i], radius=0.01, fill= True, opacity=0.5, color='yellow').add_to(nymap)
    
nymap
# Storing the LAT & LON for pickup/ dropoff with 3 decimal points only into new columns
filtered_main_df['pickup_latitude_round3'] = filtered_main_df.pickup_latitude.apply(lambda x: round(x,3))
filtered_main_df['pickup_longitude_round3'] = filtered_main_df.pickup_longitude.apply(lambda x: round(x,3))
filtered_main_df['dropoff_latitude_round3'] = filtered_main_df.dropoff_latitude.apply(lambda x: round(x,3))
filtered_main_df['dropoff_longitude_round3'] = filtered_main_df.dropoff_longitude.apply(lambda x: round(x,3))
filtered_main_df.head()
# Scatter plot with the new rounded LAT & LON
plt.style.use('dark_background')
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(15,10))

filtered_main_df.plot(kind='scatter', x='pickup_longitude_round3', y='pickup_latitude_round3', color='yellow', s=.02, alpha=.6, subplots=True, ax=ax1)
ax1.set_title("Pickups", color= 'white')
ax1.axes.tick_params(color= 'white', labelcolor= 'white')
ax1.set_facecolor('black')

filtered_main_df.plot(kind='scatter', x='dropoff_longitude_round3', y='dropoff_latitude_round3', color='orange', s=.02, alpha=.6, subplots=True, ax=ax2)
ax2.set_title("Dropoffs", color= 'white')
ax2.axes.tick_params(color= 'white', labelcolor= 'white')
ax2.set_facecolor('black') 
def scatter_plotting(df, p_lng, p_lat, d_lng, d_lat, p_cap, d_cap):
    
    plt.style.use('dark_background')
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(15,10))

    df.plot(kind='scatter', x= p_lng, y= p_lat, color='yellow', s=.02, alpha=.6, subplots=True, ax=ax1)
    ax1.set_title(p_cap, color= 'white')
    ax1.axes.tick_params(color= 'white', labelcolor= 'white')
    ax1.set_facecolor('black')

    df.plot(kind='scatter', x= d_lng, y= d_lat, color='orange', s=.02, alpha=.6, subplots=True, ax=ax2)
    ax2.set_title(d_cap, color= 'white')
    ax2.axes.tick_params(color= 'white', labelcolor= 'white')
    ax2.set_facecolor('black') 
scatter_plotting(filtered_main_df.loc[(filtered_main_df.airport_pickup == 'JFK Airport') | (filtered_main_df.airport_dropoff == 'JFK Airport')] , 'pickup_longitude_round3', 'pickup_latitude_round3', 'dropoff_longitude_round3', 'dropoff_latitude_round3', 'JFK Airport Pickups', 'JFK Airport Dropoffs') 	
scatter_plotting(filtered_main_df.loc[(filtered_main_df.airport_pickup == 'La Guardia Aiport') | (filtered_main_df.airport_dropoff == 'La Guardia Aiport')] , 'pickup_longitude_round3', 'pickup_latitude_round3', 'dropoff_longitude_round3', 'dropoff_latitude_round3', 'La Guardia Aiport Pickups', 'La Guardia Aiport Dropoffs') 	
scatter_plotting(filtered_main_df.loc[(filtered_main_df.pickup_area == 'Manhattan') | (filtered_main_df.dropoff_area == 'Manhattan')] , 'pickup_longitude_round3', 'pickup_latitude_round3', 'dropoff_longitude_round3', 'dropoff_latitude_round3', 'Manhattan Pickups', 'Manhattan Dropoffs') 	
scatter_plotting(filtered_main_df.loc[(filtered_main_df.pickup_area == 'Queens') | (filtered_main_df.dropoff_area == 'Queens')] , 'pickup_longitude_round3', 'pickup_latitude_round3', 'dropoff_longitude_round3', 'dropoff_latitude_round3', 'Queens Pickups', 'Queens Dropoffs') 	
scatter_plotting(filtered_main_df.loc[(filtered_main_df.pickup_area == 'Brooklyn') | (filtered_main_df.dropoff_area == 'Brooklyn')] , 'pickup_longitude_round3', 'pickup_latitude_round3', 'dropoff_longitude_round3', 'dropoff_latitude_round3', 'Brooklyn Pickups', 'Brooklyn Dropoffs') 	
scatter_plotting(filtered_main_df.loc[(filtered_main_df.pickup_area == 'Bronx') | (filtered_main_df.dropoff_area == 'Bronx')] , 'pickup_longitude_round3', 'pickup_latitude_round3', 'dropoff_longitude_round3', 'dropoff_latitude_round3', 'Bronx Pickups', 'Bronx Dropoffs') 	
filtered_main_df.columns
# Convert pickup and dropoff points to array to retreive the distance in KM faster
pick_drop_points = filtered_main_df[['pickup_latitude_round3', 'pickup_longitude_round3', 'dropoff_latitude_round3', 'dropoff_longitude_round3' ]].values
# Check the new array slicing 
print(len(pick_drop_points))
print(pick_drop_points[0])
print(pick_drop_points[0][0:2])
print(pick_drop_points[0][2:4])
print(pick_drop_points[0][0])
print(pick_drop_points[0][1])
# Create empty list to append the new distance points
dist_result = []

# Calculating the distance function
def calc():
    for i in range(0, len(pick_drop_points)):
        pickpoint = (pick_drop_points[i][0], pick_drop_points[i][1])
        droppoint = (pick_drop_points[i][2], pick_drop_points[i][3])
        dist_result.append(geodesic(pickpoint, droppoint).km)

# Executing the function
calc()
# To make sure dist_result & filtered_main_df are in the same length
print('dist_result array length is: {}'.format(len(dist_result)))
print('filtered_main_df length is: {}\n'.format(len(filtered_main_df)))
filtered_main_df['est_distance'] = np.around(dist_result, decimals=1)
filtered_main_df['est_distance'] = filtered_main_df['est_distance'].astype('float32')
print('estimated distance head overview: \n{}\n'.format(filtered_main_df.est_distance.sort_values().head()))
print('estimated distance tail overview: \n{}\n'.format(filtered_main_df.est_distance.sort_values().tail()))
filtered_main_df.loc[filtered_main_df.est_distance == 0, 'est_distance'].count() # The count of observations where 'est_distance == 0'
filtered_main_df.loc[filtered_main_df.est_distance == 0].head()
# Above points (LAT & LON) were randomly checked and manually checked using Google maps. Based on that, I have decided to remove these points as they are invalid.

filtered_main_df = filtered_main_df.loc[~(filtered_main_df.est_distance.values == 0)].copy() # Remove the observations where 'est_distance == 0'
reorganize_df_comman(filtered_main_df) # Reset the index
filtered_main_df.drop(['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'], axis= 1, inplace= True)
# Extracting year, month, day, time from the pickup_datetime column
filtered_main_df['year_pick'] = filtered_main_df['pickup_datetime'].dt.year.astype('int32')
filtered_main_df['month_pick'] = filtered_main_df['pickup_datetime'].dt.month.astype('int32')
filtered_main_df['day_pick'] = filtered_main_df['pickup_datetime'].dt.day.astype('int32')
filtered_main_df['weekday_pick'] = filtered_main_df['pickup_datetime'].dt.weekday.astype('int32')
filtered_main_df['time_pick'] = filtered_main_df['pickup_datetime'].dt.time
filtered_main_df['hour_pick'] = filtered_main_df['pickup_datetime'].dt.hour.astype('int32')

# Format the trip duration from seconds to HH:MM:SS format
filtered_main_df['trip_dur_formated'] = pd.to_datetime(filtered_main_df['trip_duration'], unit='s').dt.strftime('%H:%M:%S')
# Weekdays VS Weekends
def weekdays_weekends(x):
    return 'Weekend' if (x == 5) or (x == 6) else 'Weekday'

# Define the taxi type
def taxi_type(x):
    return 'Limousine / Van' if x > 4 else 'Regular Taxi'

# Define rush hours
def rush_hours(x):
    return 'Rush Hours' if (x >= 7 and x <= 10) or (x >= 16 and x <= 19) else 'Normal'

# Define night surcharge USD 0.5
def night_surcharge(x):
    return 0.5 if (x >= 20 or x < 6) else 0

# Define peak charges USD 1
def peak_charges(x):
    return 1 if (x >= 16 and x < 20) else 0

# Extract the season
def season(x):
    if (x >= 1 and x < 3):
        return 'Winter'
    if (x >= 3 and x < 6):
        return 'Spring'
    if x >= 6:
        return 'Summer'
    else:
        return 'Unknown'
    
# Part of the day   
def part_of_the_day(x):
    if (x >= 5 and x < 12):
        return 'Morning'
    if x == 12:
        return 'Noon'
    if (x > 12 and x < 17):
        return 'Afternoon'
    if (x >= 17 and x < 20):
        return 'Evening'
    if (x >= 20 and x <= 23):
        return 'Night'
    if (x >= 0 and x <= 4):
        return 'Midnight'
    else:
        return 'Unknown'

# Creating new columns for the new features
filtered_main_df['weekdays_weekends'] = filtered_main_df.weekday_pick.apply(weekdays_weekends).astype('category') # Add weekends/ weekdays to the dataframe
filtered_main_df['taxi_type'] = filtered_main_df.passenger_count.apply(taxi_type).astype('category') # # Add van types to the dataframe
filtered_main_df['avg_speed'] = round(filtered_main_df.est_distance / (filtered_main_df.trip_duration / 3600)).astype('int32') # Add average speed to the dataframe
filtered_main_df['season'] = filtered_main_df.month_pick.apply(season).astype('category') # Add seasons hours to the dataframe

filtered_main_df['rush_hours'] = filtered_main_df.hour_pick.apply(rush_hours).astype('category') # Add rush hours to the dataframe
filtered_main_df['night_charges'] = filtered_main_df.hour_pick.apply(night_surcharge).astype('float32') # Add night charges to the dataframe
filtered_main_df['peak_charges'] = filtered_main_df.hour_pick.apply(peak_charges).astype('int32') # Add peak hours to the dataframe
filtered_main_df['day_part'] = filtered_main_df.hour_pick.apply(part_of_the_day).astype('category') # Add part of the day hours to the dataframe

# Fastest approach (Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6) select the rows where the [day_pick = 5 or 6] and assign 0 to peak_charges 
filtered_main_df.loc[(filtered_main_df.day_pick == 6) | (filtered_main_df.day_pick == 5), 'peak_charges'] = 0
filtered_main_df.head()
print('average speed head values overview: \n{}\n'.format(filtered_main_df.avg_speed.sort_values().head()))
print('average speed tail values overview: \n{}\n'.format(filtered_main_df.avg_speed.sort_values().tail()))

print('average speed head with features overview: \n{}\n'.format(filtered_main_df[['pickup_datetime', 'dropoff_datetime', 'pickup_latitude_round3', 'pickup_longitude_round3', 'dropoff_latitude_round3', 'dropoff_longitude_round3', 'rush_hours', 'est_distance', 'trip_duration', 'avg_speed']].sort_values('avg_speed').head()))
print('average speed tail with features overview: \n{}\n'.format(filtered_main_df[['pickup_datetime', 'dropoff_datetime', 'pickup_latitude_round3', 'pickup_longitude_round3', 'dropoff_latitude_round3', 'dropoff_longitude_round3', 'rush_hours', 'est_distance', 'trip_duration', 'avg_speed']].sort_values('avg_speed').tail()))

print('count of the observations where average speed is 0: {}'.format(len(filtered_main_df.loc[filtered_main_df.avg_speed == 0])))
print('Observations count where estimated distance less than 0.2: {}'.format(len(filtered_main_df[filtered_main_df.est_distance < 0.2])))
#Based on the random check on LAT/LON pickup and dropoff points on Google map, I can tell that all checked points are invalid. Action: Remove observations where estimated distance is less that 0.2

filtered_main_df = filtered_main_df.loc[~(filtered_main_df.est_distance.values < 0.2)].copy() # Remove the observations where 'est_distance < 0.2'
reorganize_df_comman(filtered_main_df) # Reset the index
# Quick overview (details are on the top) 
print('Weather dataset length: \n{}\n'.format(len(weather_df)))
print('Weather dataset first 5 rows: \n{}\n'.format(weather_df.head()))
print('Weather dataset available months: \n{}\n'.format(weather_df.pickup_datetime.dt.month.unique()))
print('Weather dataset data types: \n{}\n'.format(weather_df.dtypes))
print('Weather dataset missing values: \n{}\n'.format(weather_df.isnull().sum()))
weather_df = weather_df.loc[weather_df.pickup_datetime.dt.month <= 6] # Extract the first 6 months only
weather_df.fillna(method= 'ffill', inplace= True)

w_conditions_replacment = {'Overcast': 1, 'Partly Cloudy': 2, 'Clear': 3, 'Mostly Cloudy': 4, 'Light Rain': 5, 'Scattered Clouds': 6, 'Heavy Rain': 7, 'Rain': 8, 'Light Snow': 9, 'Snow': 10, 'Heavy Snow': 11, 'Light Freezing Fog': 12, 'Haze': 13, 'Light Freezing Rain': 14, 'Fog': 15, 'Unknown': 16}
weather_df.weather_condition.astype(str).replace(w_conditions_replacment, inplace= True) # Replace conditions strings with numeric values
weather_df['new'] = (weather_df.pickup_datetime.dt.month.astype(str) + weather_df.pickup_datetime.dt.day.astype(str) + weather_df.pickup_datetime.dt.hour.astype(str)).astype('int32') # Create new columns combines month/dateday/hour in weather dataframe with int32 type

weather_df.drop_duplicates(subset='new', inplace= True) # Remove duplicated rows based on the 'new' column values
reorganize_df_comman(weather_df) # Reset dataframe index, assign new indexes to avoid errors
num_array_weather = np.array(weather_df['new'], dtype= 'int32') # Store new values into numpy array of type Int32
num_array_weather_data = np.array(weather_df[['temperature', 'rain', 'snow', 'weather_condition']], dtype= 'int32') # Store the rest of values (temperature, rain, snow, conds) into numpy array of type Int32
# Process filtered_main_df 'new' columns for processing 
filtered_main_df['new'] = (filtered_main_df.month_pick.astype(str) + filtered_main_df.day_pick.astype(str) + filtered_main_df.hour_pick.astype(str)).astype('int32') # Create new columns combines month/dateday/hour in main dataframe with int32 type
num_array_main_df = np.array(filtered_main_df['new'], dtype= 'int32') # Store the new values (month/dateday/hour) codes into numpy array of type Int32
# Initiate empty lists
temperature_list = []
rain_list = []
snow_list = []
weather_condition_list = []

# Add weather data into lists 
def add_weather_data(mdh):
    try:
        indx = np.where(num_array_weather == mdh)[0][0]
        temperature_list.append(num_array_weather_data[indx][0])
        rain_list.append(num_array_weather_data[indx][1])
        snow_list.append(num_array_weather_data[indx][2])
        weather_condition_list.append(num_array_weather_data[indx][3])
        return mdh
    except:
        temperature_list.append(np.nan)
        rain_list.append(np.nan)
        snow_list.append(np.nan)
        weather_condition_list.append(np.nan)  
        return mdh

filtered_main_df.new = filtered_main_df.new.apply(add_weather_data) # Add weather data into the main dataframe
# Create new columns in main dataframe and inserting weather data into them
def add_weather_data_to_main_df():
    filtered_main_df['temperature'] = temperature_list
    filtered_main_df['rain'] = rain_list
    filtered_main_df['snow'] = snow_list
    filtered_main_df['weather_condition'] = weather_condition_list
    
add_weather_data_to_main_df() # Call creating columns/ inserting weather data
#filtered_main_df.loc[(filtered_main_df.temperature == -99) & (filtered_main_df.rain == -99) & (filtered_main_df.snow == -99) & (filtered_main_df.weather_condition == -99), ['temperature', 'rain', 'snow', 'weather_condition']] #= np.NaN # Replace -99 (missing) values with np.NaN
filtered_main_df.fillna(method= 'ffill', inplace= True) # Fill the missing values with the previous ones
# Weather columns converted to float64 when replaced the -99 with np.NaN. Re convert the data type to 'int32'
filtered_main_df.temperature = filtered_main_df.temperature.astype('int32')
filtered_main_df.rain = filtered_main_df.rain.astype('int32')
filtered_main_df.snow = filtered_main_df.snow.astype('int32')
filtered_main_df.weather_condition = filtered_main_df.weather_condition.astype('int32')
# Reverse the weather conditions values (numeric to string) and convert the type to category
def weather_conditions_to_categories():
    new_w_conditions_replacment = {1: 'Overcast', 2: 'Partly Cloudy', 3: 'Clear', 4: 'Mostly Cloudy', 5: 'Light Rain', 6: 'Scattered Clouds', 7: 'Heavy Rain', 8: 'Rain', 9: 'Light Snow', 10: 'Snow', 11: 'Heavy Snow', 12: 'Light Freezing Fog', 13: 'Haze', 14: 'Light Freezing Rain', 15: 'Fog', 16: 'Unknown'}
    filtered_main_df.weather_condition.replace(new_w_conditions_replacment, inplace=True)
    filtered_main_df.weather_condition.astype('category')

weather_conditions_to_categories() # Call function to update the weather_condition columns
#filtered_main_df.weather_condition = filtered_main_df.weather_condition.astype('category')
pd.set_option('display.max_columns', 35)
filtered_main_df.head()
###########################################################################################################################################################################
# Deleting / dropping unwanted objects / columns
del weather_df, weather_df_path, temperature_list, rain_list, snow_list, weather_condition_list, num_array_weather, num_array_weather_data, pickup_loc, west, south, east, north, before_df, after_df, less_60_df, m, s, percentiles, percentiles_vars, x # Remove weather related objects
###########################################################################################################################################################################
# Visualize the distribution of the trip duration, estimated distance and average speed
sns.set(style="darkgrid")
fig, ax = plt.subplots(ncols=3, figsize= (30,6))

sns.distplot(filtered_main_df.trip_duration.values, bins= 50, ax= ax[0])
sns.distplot(filtered_main_df.est_distance.values, bins= 50, ax= ax[1])
sns.distplot(filtered_main_df.avg_speed.values, bins= 30, ax= ax[2])

ax[0].set_title('Trip Duration Distribution')
ax[0].set_xlabel('Trip Duration Samples')

ax[1].set_title('Estimated Distance Distribution')
ax[1].set_xlabel('Estimated Distance Samples')

ax[2].set_title('Average Speed Distribution')
ax[2].set_xlabel('Average Speed Samples')

plt.show()
filtered_main_df.columns
# Prepare variables mapping
map_weekdays = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
map_weekdays_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
map_months = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun'}
# Annotation function based on the number of trips in filtered_main_df. 
def annotations(hight_number, style):
    total = len(filtered_main_df)
    if style == 'both':
        for p in _.patches:
            height = p.get_height()
            _.text(p.get_x()+p.get_width()/2., height + hight_number, '{} ({}%)'.format(int(height), int(round(height/total, 2)*100)), ha="center", va='center', fontsize=10)
    else:
        for p in _.patches:
            height = p.get_height()
            _.text(p.get_x()+p.get_width()/2., height + hight_number, '{}%'.format(int(round(height/total, 2)*100)), ha="center", va='center', fontsize=10)
# Count plot passengers, months, weekdays, seasons

sns.set(style="darkgrid")
fig, ax = plt.subplots(ncols=4, figsize= (50,8))

_ = sns.countplot(filtered_main_df.passenger_count, ax= ax[0])
annotations(25000, 'both')

_ = sns.countplot(filtered_main_df.month_pick.replace(map_months), ax= ax[1])
annotations(5000, 'both')

_ = sns.countplot(filtered_main_df.weekday_pick.replace(map_weekdays), order= map_weekdays_order, ax= ax[2])
annotations(5000, 'both')

_ = sns.countplot(filtered_main_df.season, order=['Winter', 'Spring', 'Summer'], ax= ax[3])
annotations(15000, 'both')

ax[0].set_title('Passenger(s)')
ax[0].set_xlabel('Number of Passengers')
ax[0].set_ylabel('Trips Count')

ax[1].set_title('Months (Jan-Jun 2016)')
ax[1].set_xlabel('Months Samples')
ax[1].set_ylabel('Trips Count')

ax[2].set_title('Weekdays')
ax[2].set_xlabel('Weekdays')
ax[2].set_ylabel('Trips Count')
  
ax[3].set_title('Seasons (Summer Includes The Month of Jun Only)')
ax[3].set_xlabel('Seasons')
ax[3].set_ylabel('Trips Count')

plt.show()
# Count plot weekdays/weekends, taxi types, rush hours, part of the day

fig, ax = plt.subplots(ncols=4, figsize= (50,8))

_ = sns.countplot(filtered_main_df.weekdays_weekends, ax= ax[0])
annotations(18000, 'both')

_ = sns.countplot(filtered_main_df.taxi_type, ax= ax[1])
annotations(20000, 'both')

_ = sns.countplot(filtered_main_df.rush_hours, ax= ax[2])
annotations(15000, 'both')

_ = sns.countplot(filtered_main_df.day_part, order = ['Morning', 'Noon', 'Afternoon', 'Evening', 'Night', 'Midnight'], ax= ax[3])
annotations(6000, 'both')

ax[0].set_title('Weekdays Vs. Weekends (Weekends are Saturday & Sunday)')
ax[0].set_xlabel('Weekdays Vs. Weekends')
ax[0].set_ylabel('Trips Count')

ax[1].set_title('Taxi Type (Regular & Limousine)')
ax[1].set_xlabel('Taxi Type')
ax[1].set_ylabel('Trips Count')

ax[2].set_title('Rush Hours (7:00-10:00 AM & 4:00-7:00 PM)')
ax[2].set_xlabel('Rush Hours')
ax[2].set_ylabel('Trips Count')
  
ax[3].set_title('Part of The Day')
ax[3].set_xlabel('Part of The Day')
ax[3].set_ylabel('Trips Count')

plt.show()
plt.figure(figsize= (30, 8))

_ = sns.countplot(filtered_main_df.weather_condition)

totalLen = len(filtered_main_df)
annotations(12572,'both')

_.set_title('Number of Trips Based on Weather Conditions')
_.set_xlabel('Weather Conditions')
_.set_ylabel('Trips Count')
_.set_xticklabels(_.get_xticklabels(), rotation=90)

plt.show()
# Formula to automatically select the best height value for plotting annotations
int((filtered_main_df.weather_condition.value_counts().max() / filtered_main_df.weather_condition.value_counts().min()) * 2)
# Count plot pickups / dropoffs points

fig, ax = plt.subplots(ncols=4, figsize= (50,8))

_ = sns.countplot(filtered_main_df.pickup_area, ax= ax[0])
annotations(20000, 'both')

_ = sns.countplot(filtered_main_df.dropoff_area, ax= ax[1])
annotations(20000, 'both')

_ = sns.countplot(filtered_main_df.airport_pickup, ax= ax[2])
annotations(20000, 'both')

_ = sns.countplot(filtered_main_df.airport_dropoff, ax= ax[3])
annotations(20000, 'both')

ax[0].set_title('Pickup Areas')
ax[0].set_xlabel('Pickup Areas')
ax[0].set_ylabel('Trips Count')

ax[1].set_title('Dropoff Areas')
ax[1].set_xlabel('Dropoff Areas')
ax[1].set_ylabel('Trips Count')

ax[2].set_title('Pickups From Cities / Airports')
ax[2].set_xlabel('Pickup')
ax[2].set_ylabel('Trips Count')

ax[3].set_title('Dropoff To Cities / Airports')
ax[3].set_xlabel('Dropoff')
ax[3].set_ylabel('Trips Count')

plt.show()
# Count plot passengers, months, weekdays, seasons

sns.set(style="darkgrid")
fig, ax = plt.subplots(ncols=3, figsize= (50,12))

_ = sns.countplot(filtered_main_df.passenger_count, hue= filtered_main_df.airport_pickup, ax= ax[0])
annotations(25000, '')

_ = sns.countplot(filtered_main_df.month_pick.replace(map_months), hue= filtered_main_df.airport_pickup, ax= ax[1])
annotations(5000, '')

_ = sns.countplot(filtered_main_df.weekday_pick.replace(map_weekdays), order= map_weekdays_order, hue= filtered_main_df.airport_pickup, ax= ax[2])
annotations(5000, '')

ax[0].set_title('Passenger(s)')
ax[0].set_xlabel('Number of Passengers')
ax[0].set_ylabel('Trips Count')

ax[1].set_title('Months (Jan-Jun 2016)')
ax[1].set_xlabel('Months Samples')
ax[1].set_ylabel('Trips Count')

ax[2].set_title('Weekdays')
ax[2].set_xlabel('Weekdays')
ax[2].set_ylabel('Trips Count')
  
plt.show()
filtered_main_df.head()
# Check the relation between the trip duration and the estimated distance considering the rush hours.
plt.figure(figsize=(20,10))
sns.scatterplot(data = filtered_main_df, x = 'trip_duration', y = 'est_distance', hue = 'rush_hours', size= 'avg_speed', palette='Set2')
# Check the relation between the trip duration and the estimated distance considering the number of passengers rush hours, and weekends/ weekdays.
sns.relplot(x = 'trip_duration', y = 'est_distance', hue = 'rush_hours', size = 'avg_speed', col = 'passenger_count', row = 'weekdays_weekends', palette = 'bright', data = filtered_main_df)
# Check the relation between trip duration and average speed considering the number of passengers rush hours, estimated distance, and weekends/ weekdays.
sns.relplot(x = 'trip_duration', y = 'avg_speed', hue = 'rush_hours', size = 'est_distance', col = 'passenger_count', row = 'weekdays_weekends', palette = 'bright', data = filtered_main_df)
# Check the relation between estimated distance  and average speed considering the number of passengers rush hours, trip duration, and weekends/ weekdays.
sns.relplot(x = 'est_distance', y = 'avg_speed', hue = 'rush_hours', size = 'trip_duration', col = 'passenger_count', row = 'weekdays_weekends', palette = 'bright', data = filtered_main_df)
filtered_main_df[['trip_duration', 'est_distance', 'avg_speed']].corr()
# Get LAT & LON in one column with '+' to seperate them
lat_lon_temp = filtered_main_df.pickup_latitude_round3.astype(str) + '+' + filtered_main_df.pickup_longitude_round3.astype(str)

# Create new dataframe to store the value_counts results
top_locations = pd.DataFrame()
top_locations['points'] = lat_lon_temp.value_counts().index[:500]
top_locations['counts'] = lat_lon_temp.value_counts().values[:500]

# Seperate the 'points' column into 'lat' and 'lon' columns
top_locations['lat'] = top_locations.points.str.split('+').str.get(0)
top_locations['lon'] = top_locations.points.str.split('+').str.get(1)

# Drop 'points' columns
top_locations.drop('points', axis= 1, inplace= True)
# I get error here because of the timout failure. On my local machine works fine

# Collect the addresses using geolocator function
'''
addresses = []

lat_lon_points = top_locations[['lat', 'lon']].values
geolocator = Nominatim(user_agent="nytaxi-analysis-project")

for i in lat_lon_points:
    v = geolocator.reverse((i), timeout=50)
    addresses.append(v[0])
'''
# Get the length excluding duplicated addresses


'''
print('Length of extracted addresses: {}'.format(len(addresses)))
print('Length of extracted addresses excluding duplicated ones: {}'.format(len(set(addresses))))
'''
'''
top_locations['address'] = addresses
top_locations['post_code'] = top_locations.address.str.split(',').str.get(-2)
top_locations['short_address'] = top_locations.address.str.split(',').str.get(0) + ', ' + top_locations.address.str.split(',').str.get(1) + ', ' + top_locations.address.str.split(',').str.get(2)
'''
'''
pd.set_option('max_colwidth', 200)
top_locations[:5]
'''
# Check missing values and manually collect it from Google to update the cell. Also looking at the address column, you will find the postcode as 10001

'''
print(top_locations.post_code.isnull().sum())
print(top_locations[top_locations.post_code.isnull()])
'''
# Postcode cleaning function 

'''
def cleanPostcode(x):
    
    if 'NY' in x: #x.contains('NY', nan= False):
        x = x.replace('NY', '')
    elif 'New York' in x: 
        x = x.replace('New York', '9999')
    elif '-' in x:
        x = x.split('-')[0]
    elif ':' in x:
        x = x.split(':')[0]
    
    return x

# Clean postcodes
top_locations.post_code = top_locations.post_code.apply(lambda x: cleanPostcode(x)).str.strip()

# Create array with [Index : Postcode]
indxValues = [[27 , 10153], [45 , 10011], [111 , 10003], [119 , 10036], [173 , 10119], [254 , 10020],
              [284 , 10001], [289 , 10002], [303 , 10019], [331 , 10010], [370 , 10017], [406 , 10010], [429 , 10154]]

# Run for loop for the indexes of the 9999 postcodes within indxValues array and assign the new postcodes values
for i in indxValues:
    top_locations.loc[i[0], 'post_code'] = i[1]
    
# Looking at the above graph, I can see two postcodes that are wrong. I will assign the new postcodes using Google map
top_locations.loc[[81,366], 'post_code'] = 10014
top_locations.loc[[253,436], 'post_code'] = 10016
'''
# Matching postcodes and sum the counts
'''
def finalTopLoc(postcode):
    try:
        finalCount = top_locations.loc[top_locations.post_code == postcode, 'counts'].sum()
        return finalCount
    except:
        return 0
    
# Convert post_code and counts columns to int32 
top_locations.post_code = top_locations.post_code.astype('int32')
top_locations.counts = top_locations.counts.astype('int32')

# Create new dataframe to store the unique postcodes along with the counts sum
finalLoc_df = pd.DataFrame({'postcode' : top_locations.post_code.value_counts().keys().values}) # Creating the dataframe on the fly
finalLoc_df['finalCount'] = finalLoc_df.postcode.apply(lambda x: finalTopLoc(x)).astype('int32') # Matching the postcodes and sum the counts using the finalTopLoc function
finalLoc_df = finalLoc_df.sort_values('finalCount', ascending= False).reset_index(drop= True) # Reorder the values based on counts sum, reset the index and assign it to the created dataframe
'''
# Plot the final top locations based on the sum of postcode counts

'''

plt.figure(figsize = (25,8))

_ = sns.barplot(x= 'postcode', y= 'finalCount', order= finalLoc_df.postcode, palette="Blues_d", data= finalLoc_df)
_.set_title('Top locations based on postcode', weight='bold', pad= 50)
_.set_xlabel('Postcode', weight='bold', labelpad= 20)
_.set_ylabel('finalCount', weight='bold', labelpad= 20)
_.set_xticklabels(_.get_xticklabels(), rotation=90)

total = float(finalLoc_df.finalCount.sum()) 

for p in _.patches:
    height = p.get_height()
    _.text(p.get_x()+p.get_width()/2., height + 5800, '{}  ({}%)'.format(int(height), int(round(height/total, 2)*100)), ha="center", va='center', fontsize=10, rotation = 90)
'''
fig, ax = plt.subplots(ncols=4, figsize= (50,8))

filtered_main_df.hour_pick.value_counts().sort_index().plot(ax = ax[0])
filtered_main_df.groupby(['hour_pick', 'passenger_count']).size().unstack('passenger_count').plot(ax = ax[1])
filtered_main_df.groupby(['hour_pick', 'weekdays_weekends']).size().unstack('weekdays_weekends').plot(ax = ax[2])
filtered_main_df.groupby(['hour_pick', 'taxi_type']).size().unstack('taxi_type').plot(ax = ax[3])

ax[0].set_title('Hourly Pickup Trend (Over 6 Months)')
ax[0].set_xlabel('Hours')
ax[0].set_xticks(ticks= filtered_main_df.hour_pick.value_counts().sort_index().keys())

ax[1].set_title('Hourly Pickup Trend By Passengers Count (Over 6 Months)')
ax[1].set_xlabel('Hours')
ax[1].set_xticks(ticks= filtered_main_df.hour_pick.value_counts().sort_index().keys())

ax[2].set_title('Hourly Pickup Trend By Weekends (Over 6 Months)')
ax[2].set_xlabel('Hours')
ax[2].set_xticks(ticks= filtered_main_df.hour_pick.value_counts().sort_index().keys())

ax[3].set_title('Hourly Pickup Trend By Taxi Types (Over 6 Months)')
ax[3].set_xlabel('Hours')
ax[3].set_xticks(ticks= filtered_main_df.hour_pick.value_counts().sort_index().keys())

plt.show()
plt.figure(figsize=(50,8))

filtered_main_df.pickup_datetime.dt.date.value_counts().sort_index().plot()

plt.title('Trips By Date ( Over 6 Months)')
plt.xlabel('Dates')
plt.ylabel('Trip Counts')
plt.xticks(label= filtered_main_df.pickup_datetime.dt.date.value_counts().sort_index().keys(), rotation = 90)
plt.margins(0.005)

plt.show()
plt.figure(figsize=(20,8))

filtered_main_df.loc[(filtered_main_df.month_pick == 1) & (filtered_main_df.day_pick == 23), 'hour_pick'].value_counts().sort_index().plot()

plt.title('Trips By Date ( Over 6 Months)')
plt.xlabel('Dates')
plt.ylabel('Trip Counts')
plt.xticks(label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
plt.margins(0.005)

plt.show()
print(filtered_main_df.loc[(filtered_main_df.month_pick == 1) & (filtered_main_df.day_pick == 23), ['pickup_datetime', 'hour_pick', 'rain', 'snow']].head())
print('')
print(filtered_main_df.loc[(filtered_main_df.month_pick == 1) & (filtered_main_df.day_pick == 23), ['pickup_datetime', 'hour_pick', 'rain', 'snow']].tail())
print('')
print('Total trips in 23-1-2016:  {} trips'.format(len(filtered_main_df[(filtered_main_df.month_pick == 1) & (filtered_main_df.day_pick == 23)])))
fig, ax = plt.subplots(ncols=1, figsize= (50,8))

filtered_main_df.groupby([filtered_main_df.pickup_datetime.dt.date, 'day_part']).size().unstack('day_part').plot(ax = ax)

ax.set_title('Trips By Date & Part of The Day ( Over 6 Months)')
ax.set_xlabel('Dates')
ax.set_xticks(ticks= filtered_main_df.pickup_datetime.dt.date.value_counts().keys())
plt.xticks(rotation = 90)
ax.set_ylabel('Trips')
plt.margins(0.005)
                
plt.show()
fig, ax = plt.subplots(ncols=1, figsize= (50,8))

filtered_main_df.groupby([filtered_main_df.pickup_datetime.dt.date, 'rush_hours']).size().unstack('rush_hours').plot(ax = ax)

ax.set_title('Trips By Date & Rush Hours ( Over 6 Months)')
ax.set_xlabel('Dates')
ax.set_xticks(ticks= filtered_main_df.pickup_datetime.dt.date.value_counts().keys())
plt.xticks(rotation = 90)
ax.set_ylabel('Trips')
plt.margins(0.005)
                
plt.show()
fig, ax = plt.subplots(ncols=1, figsize= (50,8))

filtered_main_df.groupby([filtered_main_df.pickup_datetime.dt.date, 'taxi_type']).size().unstack('taxi_type').plot(ax = ax)

ax.set_title('Trips By Date & Taxi Type ( Over 6 Months)')
ax.set_xlabel('Dates')
ax.set_xticks(ticks= filtered_main_df.pickup_datetime.dt.date.value_counts().keys())
plt.xticks(rotation = 90)
ax.set_ylabel('Trips')
plt.margins(0.005)
                
plt.show()
fig, ax = plt.subplots(ncols=1, figsize= (50,8))

filtered_main_df.groupby([filtered_main_df.pickup_datetime.dt.date, 'pickup_area']).size().unstack('pickup_area').plot(ax = ax)

ax.set_title('Trips By Date & Pickup Areas ( Over 6 Months)')
ax.set_xlabel('Dates')
ax.set_xticks(ticks= filtered_main_df.pickup_datetime.dt.date.value_counts().keys())
plt.xticks(rotation = 90)
ax.set_ylabel('Trips')
plt.margins(0.005)
                
plt.show()
fig, ax = plt.subplots(ncols=1, figsize= (50,8))

filtered_main_df.groupby([filtered_main_df.pickup_datetime.dt.date, 'airport_pickup']).size().unstack('airport_pickup').plot(ax = ax)

ax.set_title('Trips By Date & Pickup Areas ( Over 6 Months)')
ax.set_xlabel('Dates')
ax.set_xticks(ticks= filtered_main_df.pickup_datetime.dt.date.value_counts().keys())
plt.xticks(rotation = 90)
ax.set_ylabel('Trips')
plt.margins(0.005)
                
plt.show()
fig, ax = plt.subplots(ncols=1, figsize= (50,8))

filtered_main_df.groupby([filtered_main_df.pickup_datetime.dt.date, 'vendor_id']).size().unstack('vendor_id').plot(ax = ax)

ax.set_title('Trips By Date & Vendor ID ( Over 6 Months)')
ax.set_xlabel('Dates')
ax.set_xticks(ticks= filtered_main_df.pickup_datetime.dt.date.value_counts().keys())
plt.xticks(rotation = 90)
ax.set_ylabel('Trips')
plt.margins(0.005)
                
plt.show()
months = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun'}

fig, ax = plt.subplots(ncols=1, figsize= (15,8))

filtered_main_df.pickup_datetime.dt.month.value_counts().sort_index().plot()

ax.set_title('Trips Over 6 Months')
ax.set_xlabel('Months')
ax.set_xticks(ticks= filtered_main_df.pickup_datetime.dt.month.value_counts().keys())
ax.set_ylabel('Trips')
plt.margins(0.005)
                

plt.show()
# Temprary convert back 'store_and_fwd_flag' column
store_flag_dict_flip = {1: 'Y', 0: 'N'}
filtered_main_df.store_and_fwd_flag = filtered_main_df.store_and_fwd_flag.replace(store_flag_dict_flip)
fig, ax = plt.subplots(ncols=1, figsize= (50,8))

filtered_main_df.groupby([filtered_main_df.pickup_datetime.dt.date, 'store_and_fwd_flag']).size().unstack('store_and_fwd_flag').plot(ax = ax)

ax.set_title('Storing Data Status (Over 6 Months)')
ax.set_xlabel('Dates')
ax.set_xticks(ticks= filtered_main_df.pickup_datetime.dt.date.value_counts().keys())
plt.xticks(rotation = 90)
ax.set_ylabel('Trips')
plt.margins(0.005)
                
plt.show()
# Annotation function for the customized sns.barplot function 
def annotationsCustom(plotVar, totalLength, style):
 
    if style == 'both':
        for p in plotVar.patches:
            height = p.get_height()
            plotVar.text(p.get_x()+p.get_width()/2., height + (height * 0.02), '{} ({}%)'.format(int(height), int(round(height/totalLength, 2)*100)), ha="center", va='center', fontsize=10)
    else:
        for p in plotVar.patches:
            height = p.get_height()
            plotVar.text(p.get_x()+p.get_width()/2., height + (height * 0.02), '{}%'.format(int(round(height/totalLength, 2)*100)), ha="center", va='center', fontsize=10)

            
# Customized sns.barplot function
def getGroupby(df, col1, col2, titleVar, xLabelVar, yLabelVar, figSize, colNum):
    
    # Check if col2 is a list or a string
    if (colNum > 1) & (type(col2) == list):
        
        # Create the figure and number of columns
        fig, ax = plt.subplots(ncols= colNum, figsize= figSize)
        
        # Loop through the col2 list by taking the index and calling the variable using col2[colIndx]
        for colIndx in range(len(col2)):

            tempVar = df.groupby(col1)[col2[colIndx]].mean()
            x = tempVar.index
            y = tempVar.values

            _ = sns.barplot(x, y, ax = ax[colIndx])
            
            # Calling annotation function
            totalLength = len(tempVar)
            annotationsCustom(plotVar= _, totalLength= totalLength, style= 'both')

            ax[colIndx].set_title(titleVar[colIndx], weight = 'bold', pad = 15)
            
            # Check if axes are list or string
            if type(xLabelVar) == list:
                ax[colIndx].set_xlabel(xLabelVar[colIndx])
            else:
                ax[colIndx].set_xlabel(xLabelVar)

            if type(yLabelVar) == list:
                ax[colIndx].set_ylabel(yLabelVar[colIndx])
            else:
                ax[colIndx].set_ylabel(yLabelVar)   

            # CAN NOT ADD plt.show() in this part of the foor loop
    else:
        
        plt.figure(figsize= figSize)
        
        tempVar = df.groupby(col1)[col2].mean()
        x = tempVar.index
        y = tempVar.values

        _ = sns.barplot(x, y)
        annotationsCustom(plotVar= _, totalLength= totalLength, style= 'both')

        _.set_title(titleVar, weight = 'bold', pad = 15)
        _.set_xlabel(xLabelVar)
        _.set_ylabel(yLabelVar)

    plt.show()
        
        
        
# Plot - Compare betweem storing data statuses 'Y' and 'N' with the mean of vender ID, trip_duration,        
# Calling customized sns.barplot function
getGroupby(df = filtered_main_df,
           col1= 'store_and_fwd_flag', 
           col2=['trip_duration', 'est_distance', 'avg_speed', 'vendor_id'],
           titleVar=['Storing Data Status Based on Trip Duration', 'Storing Data Status Based on Estimated Distance', 'Storing Data Status Based on Average Speed', 'Storing Data Status Based on Vendor ID'], 
           xLabelVar='Storing Data Status', 
           yLabelVar=['Trip Duration Average', 'Distance Average', 'Speed Average', 'Vendor ID'], 
           figSize=(50, 8), 
           colNum=4)
fig, ax = plt.subplots(ncols=1, figsize= (50,12))

#filtered_main_df.groupby([filtered_main_df.pickup_datetime.dt.date, 'temperature']).size().unstack('temperature').plot(ax = ax)
sns.lineplot(x = filtered_main_df.pickup_datetime.dt.date, y = 'temperature', data = filtered_main_df, ax = ax)

ax.set_title('Daily Temperature ( Over 6 Months)')
ax.set_xlabel('Dates')
ax.set_xticks(ticks= filtered_main_df.pickup_datetime.dt.date.value_counts().keys())
plt.xticks(rotation = 90)
ax.set_ylabel('Temperature')
ax.margins(0.005)                

plt.show()
fig, ax = plt.subplots(ncols=1, figsize= (50,12))

filtered_main_df.groupby('temperature').size().plot(ax = ax)
#sns.lineplot( x = filtered_main_df.temperature.value_counts().keys(), y = filtered_main_df.temperature.value_counts().values, data = filtered_main_df, ax = ax)

ax.set_title('Daily Temperature with Number of Trips (Over 6 Months)')
ax.set_xlabel('Temperatures')
ax.set_xticks(ticks= filtered_main_df.temperature.sort_values().unique())
#plt.xticks(rotation = 90)
ax.set_ylabel('Temperature')
ax.margins(0.005)                

plt.show()
fig, ax = plt.subplots(ncols=1, figsize= (50,12))

filtered_main_df.groupby(['temperature', 'taxi_type']).size().unstack('taxi_type').plot(ax = ax)

ax.set_title('Daily Temperature with Number of Trips (Over 6 Months)')
ax.set_xlabel('Temperatures')
ax.set_xticks(ticks= filtered_main_df.temperature.sort_values().unique())
#plt.xticks(rotation = 90)
ax.set_ylabel('Temperature')
ax.margins(0.005)                

plt.show()
fig, ax = plt.subplots(ncols=1, figsize= (50,12))

filtered_main_df.groupby(['temperature', 'rush_hours']).size().unstack('rush_hours').plot(ax = ax)

ax.set_title('Daily Temperature with Number of Trips (Over 6 Months)')
ax.set_xlabel('Temperatures')
ax.set_xticks(ticks= filtered_main_df.temperature.sort_values().unique())
#plt.xticks(rotation = 90)
ax.set_ylabel('Temperature')
ax.margins(0.005)                

plt.show()
# Replace digits with the name of each month
months = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'No', 12:'Dec'}

# Extract key/ values from .value_counts() and display the results into dataframe style
a = filtered_main_df.rush_hours.value_counts().sort_index().keys()
b = filtered_main_df.rush_hours.value_counts().sort_index().values
c = round(filtered_main_df.rush_hours.value_counts(normalize=True).sort_index(), 3).values*100
print('Rush Hours: \n{}\n'.format(pd.DataFrame(data= list(zip(a, b, c)), columns=['rush_hours','counts', 'percentage']).set_index('rush_hours')))

a = filtered_main_df.month_pick.value_counts().sort_index().keys().map(months)
b = filtered_main_df.month_pick.value_counts().sort_index().values
c = round(filtered_main_df.month_pick.value_counts(normalize=True).sort_index(), 3).values*100
print('Monthly Distribution: \n{}\n'.format(pd.DataFrame(data= list(zip(a, b, c)), columns=['months','counts', 'percentage']).set_index('months')))

a = filtered_main_df.passenger_count.value_counts().sort_index().keys()
b = filtered_main_df.passenger_count.value_counts().sort_index().values
c = round(filtered_main_df.passenger_count.value_counts(normalize=True).sort_index(), 3).values*100
print('Number of Passenger Distribution: \n{}\n'.format(pd.DataFrame(data= list(zip(a, b, c)), columns=['number_of_passengers', 'counts', 'percentage']).set_index('number_of_passengers')))

a = filtered_main_df.taxi_type.value_counts().sort_index().keys()
b = filtered_main_df.taxi_type.value_counts().sort_index().values
c = round(filtered_main_df.taxi_type.value_counts(normalize=True).sort_index(), 3).values*100
print('Regular taxi VS Van taxi: \n{}\n'.format(pd.DataFrame(data= list(zip(a, b, c)), columns=['taxi_types', 'counts', 'percentage']).set_index('taxi_types')))
# Interactive plotting 

def plotmonth(month):
    
    data = filtered_main_df[filtered_main_df.month_pick == month]
    circleSize = data.avg_speed
    colors = data.passenger_count.map({1: 'skyblue', 2: 'coral', 3: 'green', 4: 'gold', 5: 'palegreen', 6: 'brown'})
    
    data.plot.scatter('trip_duration', 'est_distance', s = circleSize, c = colors, alpha = 0.8, linewidths = 1, edgecolors = 'white', figsize=(25,12))
    
    plt.show()
    
#plotmonth(1)

# Interactive plotting is not working. Needs to check the nodejs and npm packages first.
interact(plotmonth, month = widgets.IntSlider(min=1, max=6, step=1, value=1))
from sklearn.cluster import MiniBatchKMeans
filtered_main_df.columns
# Prepare the coordinates 
cords = np.vstack((filtered_main_df[['pickup_latitude_round3', 'pickup_longitude_round3']], filtered_main_df[['dropoff_latitude_round3', 'dropoff_longitude_round3']])) # np.vstack function takes tuple
# Randomize the cords
cordsSamples = np.random.permutation(len(cords))[:500000]
# Train the MiniBatchKMeans and fit it on the cordsSamples
miniKmean = MiniBatchKMeans(n_clusters= 50, batch_size= 10000).fit(cords[cordsSamples])
# Predict the LAT and LON points and assign the clusters into their new columns
filtered_main_df['pickup_cluster'] = miniKmean.predict(filtered_main_df[['pickup_latitude_round3', 'pickup_longitude_round3']])
filtered_main_df['dropoff_cluster'] = miniKmean.predict(filtered_main_df[['dropoff_latitude_round3', 'dropoff_longitude_round3']])
filtered_main_df.head()
# Scatter plot with the new rounded LAT & LON
plt.style.use('dark_background')
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(18,10))

filtered_main_df.plot(kind='scatter', x='pickup_longitude_round3', y='pickup_latitude_round3', c= 'pickup_cluster', s=.1, alpha=.2, subplots=True, ax=ax1)
ax1.set_title("Pickups", color= 'white')
ax1.axes.tick_params(color= 'white', labelcolor= 'white')
ax1.set_facecolor('black')
ax1.grid(False)

filtered_main_df.plot(kind='scatter', x='dropoff_longitude_round3', y='dropoff_latitude_round3', c= 'dropoff_cluster', s=.1, alpha=.2, subplots=True, ax=ax2)
ax2.set_title("Dropoffs", color= 'white')
ax2.axes.tick_params(color= 'white', labelcolor= 'white')
ax2.set_facecolor('black') 
ax2.grid(False)
cols = ['pickup_longitude_round3', 'pickup_latitude_round3']
cordsGroupedBy_MeanSpeed = filtered_main_df.groupby(cols)['avg_speed'].mean().reset_index()
cordsGroupedBy_CountVid = filtered_main_df.groupby(cols)['vendor_id'].count().reset_index()
cordsStats = pd.merge(cordsGroupedBy_MeanSpeed, cordsGroupedBy_CountVid, on= cols)
cordsStats = cordsStats[cordsStats.vendor_id > 100]
plt.style.use('dark_background')
fig, ax = plt.subplots(ncols=1, figsize= (10,14))

ax.scatter( x=filtered_main_df.pickup_longitude_round3.values, y= filtered_main_df.pickup_latitude_round3.values, c= 'white', s=.5, alpha=.2)
ax.scatter(x= cordsStats.pickup_longitude_round3.values, y= cordsStats.pickup_latitude_round3.values, c= cordsStats.avg_speed.values, cmap='YlOrBr', s=5, alpha=.8, vmin=1, vmax=8)

ax.set_title("Average Speed", color= 'white')
ax.axes.tick_params(color= 'white', labelcolor= 'white')
ax.set_facecolor('black')
ax.grid(False)

plt.show()
cordsGroupedBy_MeanTripDur = filtered_main_df.groupby(cols)['trip_duration'].mean().reset_index()
cordsStats_TripDur = pd.merge(cordsGroupedBy_MeanTripDur, cordsGroupedBy_CountVid, on= cols)
cordsStats_TripDur = cordsStats_TripDur[cordsStats_TripDur.vendor_id > 100]

plt.style.use('dark_background')
fig, ax = plt.subplots(ncols=1, figsize= (10,14))

ax.scatter( x=filtered_main_df.pickup_longitude_round3.values, y= filtered_main_df.pickup_latitude_round3.values, c= 'white', s=.5, alpha=.2)
ax.scatter(x= cordsStats_TripDur.pickup_longitude_round3.values, y= cordsStats_TripDur.pickup_latitude_round3.values, c= cordsStats_TripDur.trip_duration.values, cmap='YlOrBr', s=5, alpha=.8, vmin=1, vmax=8)

ax.set_title("Trip Duration", color= 'white')
ax.axes.tick_params(color= 'white', labelcolor= 'white')
ax.set_facecolor('black')
ax.grid(False)

plt.show()
cordsGroupedBy_MeanEstDis = filtered_main_df.groupby(cols)['est_distance'].mean().reset_index()
cordsStats_MeanEstDis = pd.merge(cordsGroupedBy_MeanEstDis, cordsGroupedBy_CountVid, on= cols)
cordsStats_MeanEstDis = cordsStats_MeanEstDis[cordsStats_MeanEstDis.vendor_id > 100]

plt.style.use('dark_background')
fig, ax = plt.subplots(ncols=1, figsize= (10,14))

ax.scatter( x=filtered_main_df.pickup_longitude_round3.values, y= filtered_main_df.pickup_latitude_round3.values, c= 'white', s=.5, alpha=.2)
ax.scatter(x= cordsStats_MeanEstDis.pickup_longitude_round3.values, y= cordsStats_MeanEstDis.pickup_latitude_round3.values, c= cordsStats_MeanEstDis.est_distance.values, cmap='YlOrBr', s=5, alpha=.8, vmin=1, vmax=8)

ax.set_title("Estimated Distance", color= 'white')
ax.axes.tick_params(color= 'white', labelcolor= 'white')
ax.set_facecolor('black')
ax.grid(False)

plt.show()
mapped_rain = {0: 'No Rain ', 1: 'Raining'} 
mapped_snow = {0: 'No Snow ', 1: 'Snowing'}
mapped_pickup_dropoff_airport = {'City': 'Not Airport'}

filtered_main_df.weekday_pick.replace(map_weekdays, inplace= True)
filtered_main_df.month_pick.replace(map_months, inplace= True)
filtered_main_df.rain.replace(mapped_rain, inplace= True)
filtered_main_df.snow.replace(mapped_snow, inplace= True)
filtered_main_df.airport_pickup.replace(mapped_pickup_dropoff_airport, inplace= True)
filtered_main_df.columns
# Prepare the columns for training dataset
train_cols = ['vendor_id', 'passenger_count', 'store_and_fwd_flag', 'trip_duration', 'pickup_area', 'dropoff_area', 'airport_pickup', 'airport_dropoff', 'pickup_latitude_round3', 'pickup_longitude_round3', 'dropoff_latitude_round3', 'dropoff_longitude_round3', 'est_distance', 'year_pick', 'month_pick', 'day_pick', 'weekday_pick', 'hour_pick', 'weekdays_weekends', 'taxi_type', 'avg_speed', 'season', 'rush_hours', 'day_part', 'temperature', 'rain', 'snow', 'weather_condition', 'pickup_cluster', 'dropoff_cluster']
# Create training dataset and assign the columns 
train_df = filtered_main_df[train_cols].copy()
# Check
train_df.head()