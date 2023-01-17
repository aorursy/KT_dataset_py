# Packages for data loading and manipulation

import sqlite3

import pandas as pd

import numpy as np

import scipy

import re



# Packages for display

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import matplotlib.dates as mdates

from matplotlib.ticker import AutoMinorLocator

## import mglearn as mg

from tqdm import tqdm

from IPython.display import display, HTML, Image

## import gmplot

## from pandas_ml import ConfusionMatrix



# Plot graphviz

from sklearn.externals.six import StringIO

from sklearn.tree import export_graphviz

## import pydotplus



# Packeages for time series

from time import time

from datetime import datetime, timedelta

## from bdateutil import isbday

## import holidays



# Data manipulation tools

## import more_itertools as mit



# Math

import random

from scipy.stats.stats import pearsonr  



# System

## from wurlitzer import sys_pipes # This is used to read the Jupyter console output.

import warnings

import socket # Check if there is internet connection
# Machine learning models

import sklearn



from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, ShuffleSplit, TimeSeriesSplit, cross_val_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, learning_curve



from sklearn.preprocessing import MinMaxScaler, minmax_scale

from sklearn.decomposition import PCA



# Metrics

from sklearn.metrics import fbeta_score, accuracy_score, mean_squared_log_error, median_absolute_error, make_scorer

from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report



# Unsupervised learning 

from sklearn.cluster import KMeans



# Regression algorithms

from sklearn.dummy import DummyRegressor

from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.gaussian_process import GaussianProcessRegressor

# from sklearn.isotonic import IsotonicRegression # This regressor doesn't have a default parameter setting.

# from sklearn.linear_model import ARDRegression # This regressor takes too much time to model.

from sklearn.linear_model import HuberRegressor, Lasso, LassoCV, LinearRegression, PassiveAggressiveRegressor

from sklearn.linear_model import RANSACRegressor, Ridge, SGDRegressor, TheilSenRegressor 

from sklearn.kernel_ridge import KernelRidge

from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.svm import LinearSVR, NuSVR, SVR

from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from xgboost import XGBRegressor



# Classification algorithms

from sklearn.dummy import DummyClassifier

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier

from sklearn.naive_bayes import GaussianNB # BernoulliNB and MultinomialNB are suitable only for discrete features

from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

# from sklearn.neighbors import RadiusNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import LinearSVC, NuSVC, SVC

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from xgboost import XGBClassifier



import xgboost as xgb
# One can ignore the warning as this is a known issue for tensorflow 1.4.1, but it doesn't affect the usage.

# https://github.com/tensorflow/tensorflow/issues/14182

with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    import tensorflow as tf

    from tensorflow.python.client import device_lib
# Deep learning packages

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.wrappers.scikit_learn import KerasClassifier

# from keras.callbacks import Callback

from keras.callbacks import ModelCheckpoint

import keras.backend as K

from keras import initializers as K_init

from keras.optimizers import Adam
# Set a global seed for randomization

random_state = 16

np.random.seed(random_state)
import sys

print(sys.version)
# Print version of important Python packages

important_packages = [np, scipy, pd, matplotlib, sklearn, xgb, tf, keras]



for package in important_packages:

    print('{}: {}'.format(package.__name__, package.__version__))
notebook_start_time = time()
# First, import all the datasets.

# The status data cannot be directly imported due to its size. Below I will read the data through the database.

# The 'station.csv' needs to be read before one can process the sqlite in the below functions.

# Zip codes should be loaded as strings.

station = pd.read_csv('../input/station.csv')

trip = pd.read_csv('../input/trip.csv', dtype={'zip_code':'str'})

weather = pd.read_csv('../input/weather.csv')
# If sql_status is true, the status data will be obtained from the sqlite file (which takes about 10 min) and saved to a new csv file named "status_change.csv".

# If false, the status data will be obtained from the preprocessed "status_change.csv" file.

sql_status = True
# Define a convenient funtion to use SQL query.

def sql(cursor, command):    

    cursor.execute(command)

    result = cursor.fetchall()

    return result     
def bike_status(station_id, sql_conn):

    '''The bike_status function iterates over stations and records only the changes of "bikes_available" and "docks_available" status.

    A Python 3.6 feature is used in this function. The f-string.'''

    

    status = pd.read_sql_query(f"SELECT * FROM status WHERE station_id = {station_id} ORDER BY time", con=sql_conn)

    

    # Register the change of status once the change is detected.

    # The exploration of status data later suggests that the datetime in status data is generally 1 minute later than that in the trip data. Both are recorded roughly every minute.

    status_compact = status[(status.bikes_available.diff(-1) * status.docks_available.diff(-1)) != 0]

    return status_compact
if sql_status == True:

    # Connect to the database in sqlite format.

    conn = sqlite3.connect('../input/database.sqlite')

    cursor = conn.cursor()  

    

    # Explore the tables in the database.

    find_tables = "SELECT name FROM sqlite_master;"

    tables = sql(cursor, find_tables)

    print(tables)
if sql_status == True:

    # Investigate the "status" columns in the database.

    # Python 3.6 f-string used.

    print(f"Table {tables[1][0]} contains the following columns:")

    display(sql(cursor, f"PRAGMA table_info({tables[1][0]});")) # tables[1][0] is 'status'.
if sql_status == True:

    # It is nearly impossible to directly import the status.csv to a pandas dataframe. It takes too much time and space.

    # In fact, it is not necessary to take all the data as plenty of them contain little information. 

    # The bike_status function will "compress" the original big data.

    # It takes about 11 minutes to process the data.

    # The 'station.csv' needs to be read before running the following code.

    

    status_list = [bike_status(station_id, conn) for station_id in tqdm(station.id)]

    

    # Concatenate all the station status to form a dataframe "status".

    status = pd.concat(status_list, ignore_index=True)

            

    # Double check that the concatenation doesn't remove rows.

    sum_status = 0

    for bike_station in status_list:

        sum_status += len(bike_station)

    

    print(sum_status)

    print(status.shape)

       

    # Write a .csv data for future import to save time. 

    # The size of data shrinks from the original 2.0GB to less than 60MB without losing important information.

    status.to_csv("status_change.csv", index=False)

    conn.close()

else:

    status = pd.read_csv("status_change.csv")
# Set the pandas option to display all the weather columns.

pd.set_option('display.max_columns', 50)



print(f"\nA sample of status data {status.shape}:"); display(status.head(3))

print(f"\nA sample of station data {station.shape}:"); display(station.head(3))

print(f"\nA sample of trip data {trip.shape}:"); display(trip.head(3))

print(f"\nA sample of weather data {weather.shape}:"); display(weather.head(3))
# First check whether there are nan values in station data.

station.isnull().sum()
# Create a location format that is compatible with that for data in the below html file.

locations = np.column_stack((station.name, station.lat, station.long, station.id)).tolist()

print(locations[0])
# Define a function here:

# Codes borrowed from:

# https://gist.github.com/parth1020/4481893

# https://github.com/vgm64/gmplot/blob/master/gmplot/gmplot.py

# Fremont location (37.548270, -121.988572)



para1 = '''

<html>

<head>  

    <title>Google Maps Multiple Markers</title>

    <script src="http://maps.google.com/maps/api/js?sensor=false&key=AIzaSyCsdvHMtyKFqb98ybFMw_q5QSipYEoZU7Y" type="text/javascript"></script>

</head>

<body>

    <div id="map" style="height: 600px; width: 1200px;">

    </div>

    <script type="text/javascript">

'''

para2 = f'var locations = {locations};'



para3 = '''

    var map = new google.maps.Map(document.getElementById('map'), {

      zoom: 10,

      center: new google.maps.LatLng(37.548270, -121.988572),

      mapTypeId: google.maps.MapTypeId.ROADMAP

    });



    var infowindow = new google.maps.InfoWindow();



    var marker, i;



    for (i = 0; i < locations.length; i++) { 

      marker = new google.maps.Marker({

        position: new google.maps.LatLng(locations[i][1], locations[i][2]),

        map: map

      });



      google.maps.event.addListener(marker, 'click', (function(marker, i) {

        return function() {

          infowindow.setContent(locations[i][0]);

          infowindow.open(map, marker);

        }

      })(marker, i));

    }

    </script>

</body>

</html>'''
# File name for writing the html code.

bike_map = "Bike_Map.html"
# Open the file to write the code and then exit it.

f = open(bike_map, 'w')

f.write(para1 + para2 + para3)

f.close()
# Define a function to check if there is internet connection.

# Code borrowed from:

# https://stackoverflow.com/questions/20913411/test-if-an-internet-connection-is-present-in-python



REMOTE_SERVER = "www.google.com"

def is_connected():

  try:

    # see if we can resolve the host name -- tells us if there is a DNS listening.

    host = socket.gethostbyname(REMOTE_SERVER)

    # connect to the host -- tells us if the host is actually reachable.

    s = socket.create_connection((host, 80), 2)

    return True

  except:

     pass

  return False
if is_connected():

    # Plot the markers on the interactive map with the station names, if there is internet

    display(HTML(f'<iframe src={bike_map} height="630px" width="100%"></iframe>'))
# https://manojsaha.com/2017/03/08/drawing-locations-google-maps-python/

# https://pypi.python.org/pypi/gmplot/1.1.1

# Initialize two empty lists to hold the latitude and longitude values

# Obtain coordinates of the stations multiplied by the number of dock counts on each station to see the heatmap of the available bikes.

latitudes = []

longitudes = []

for i in station.index:

    for dock in range(station.dock_count[i]):

        latitudes.append(station.lat[i])

        longitudes.append(station.long[i])
if is_connected():

    # Locate the center of the map to Fremont.

    # Sometimes the geocoding doesn't work very well. Just re-run the cell to clear the error message.

    # center = gmplot.GoogleMapPlotter.geocode("Fremont")

    

    center = (37.5482697, -121.9885719) # This is the Fremont coordinator

    gmap0 = gmplot.GoogleMapPlotter(center[0], center[1], zoom=10.5, apikey='AIzaSyCsdvHMtyKFqb98ybFMw_q5QSipYEoZU7Y')

    

    gmap0.heatmap(latitudes, longitudes)

    gmap0.draw("docks_heatmap.html")

    

    # Show the interactive hot map, if there is internet

    # As expected, most of the bikes are located in San Francisco and San Jose city centers as designed.

    display(HTML('<iframe src="docks_heatmap.html" height="630px" width="150%"></iframe>'))
station.groupby('installation_date').agg({'dock_count': ['count', 'sum']})
# Use clustering techniques to plot locations.

b_loc = station[['lat', 'long']] # b_loc: bike locations



# Use KMeans to find the clusters with the number of 3 and 5.

## fig, axes = plt.subplots(1, 2, figsize=(10,5))



# using three cluster centers:

kmeans_3 = KMeans(n_clusters=3, random_state=random_state).fit(b_loc)

## mg.discrete_scatter(b_loc['lat'], b_loc['long'], kmeans_3.labels_, ax=axes[0])



# using five cluster centers:

kmeans_5 = KMeans(n_clusters=5, random_state=random_state).fit(b_loc)

## mg.discrete_scatter(b_loc['lat'], b_loc['long'], kmeans_5.labels_, ax=axes[1])



## plt.legend()

## plt.show()
# Calculate the percentage of trips between stations. Most of the trips are between stations.

same_station = (trip.start_station_id != trip.end_station_id).nonzero()[0]

print("The percentage of trips between stations is {:.3%}.".format(len(same_station)/len(trip)))
# Define a function to quantify the reasonability of the KMeans clustering.

# Trips between clusters should be minimal.

def cluster2cluster(kmeans_labels, n_cluster, color='g'):

    # Group stations with the same cluster label.

    id_groups = [station.loc[kmeans_labels == i, 'id'].values for i in range(0, n_cluster)]

    

    s2s = trip.loc[:, ['start_station_id', 'end_station_id']]    

    # Create two columns to store the group ids for start and end stations.

    s2s['start_group'] = None

    s2s['end_group'] = None

    

    # Create new columns of group ids for the start station and the end station.

    group_id = 0

    for group in id_groups:

        s2s.loc[s2s.start_station_id.isin(group), 'start_group'] = group_id

        s2s.loc[s2s.end_station_id.isin(group), 'end_group'] = group_id

        group_id += 1

    

    # g2g = s2s[s2s.start_group != s2s.end_group][['start_group', 'end_group']] # One cluster to another cluster.

    g2g = s2s.loc[s2s.start_group != s2s.end_group, ['start_group', 'end_group']] # One cluster to another cluster.

    g2g_size = g2g.groupby(['start_group', 'end_group']).size() # The number of trips between groups.

    print("The percentage of trips between station clusters is {:.3%}.".format(len(g2g)/len(trip)))

    

    # Draw a group of bar charts.

    fig, axes = plt.subplots(1, n_cluster, figsize=(20, 5))    

    for i in range(n_cluster):

        x_ind = np.arange(g2g_size[i].size)

        axes[i].bar(x_ind, g2g_size[i], color=color)

        axes[i].set_xticks(x_ind)

        axes[i].set_xticklabels(g2g_size[i].index)

        axes[i].set_xlabel('Start station group %s' % i)

    

    plt.show()

    # The index of the group-to-group is returned. Those data will be deleted.

    return g2g.index
# Investigate group-to-group trips with 3 groups.

cluster3_index = cluster2cluster(kmeans_3.labels_, n_cluster=3, color='b')
# Investigate group-to-group trips with 5 groups.

cluster5_index = cluster2cluster(kmeans_5.labels_, n_cluster=5, color='g')
# Verify the group is assigned correctly. 

station.loc[:, 'group'] = kmeans_5.labels_

station[['group','name']].head(3)
# There are five unique zip codes from the weather data.

weather.zip_code.unique()
# Arrange region zip codes from north to south

regions = [94107, 94063, 94301, 94041, 95113]

print(regions)
# Group the station data by the group id and then sort the averaged longitude values in ascending order, which suggest from north to south.

north_to_south_groups = station.groupby('group')['long'].mean().sort_values()

north_to_south_idx = north_to_south_groups.index

print(north_to_south_groups)
# Create a mapping from the group ids to zip codes.

region_dict = dict(zip(north_to_south_idx, regions))

print(region_dict)
# Associate the zip codes with the cities.

region_names = dict(zip(regions, ['San Francisco', 'Redwood City', 'Palo Alto', 'Mountain View', 'San Jose']))

print(region_names)
# Create a new column of regions for trip data.

id_groups = [station['id'][kmeans_5.labels_ == i].values for i in range(5)]



trip['station_region'] = None

region_stations_dict = {} # Initiate a dictionary from the region code to stations.

group_id = 0

for group in id_groups:

    region_code = region_dict[group_id]

    region_idx = trip.start_station_id.isin(group)

    

    region_stations_dict[region_code] = group

    trip.loc[region_idx, 'station_region'] = region_code

    group_id += 1
# Investigate the percentage of inter-group trips among the groups. The group is assigned to the start station.

inter_group_trips = trip.station_region.loc[cluster5_index].value_counts()

region_trips = trip.station_region.value_counts()

inter_group_ratio = inter_group_trips / region_trips * 100
for region in region_trips.index:

    print('The percentage of trips from {} to a different region is {:.2f}% out of total {} trips.'.format(region_names[region], inter_group_ratio[region], region_trips[region]))
# Investigate how many inter-group trips are made by a subscriber or a non-subscriber

trip.subscription_type.loc[cluster5_index].value_counts()
# Create a new column to label a trip as an inter-group trip.

trip['inter_group'] = 0

trip.loc[cluster5_index, 'inter_group'] = 1
# Create a new column in station to cluster the stations into groups

station['region'] = None

for region, group in region_stations_dict.items():

    station.loc[station.id.isin(group), 'region'] = region
# trip_date = trip['start_date'].str.split(' |/|:')

# Separate the date and time. Only the start date is important. Anyway, the number of trips over 24 hours is rare.



trip_datetime = trip['start_date'].apply(lambda x: x.split())

trip_date = trip_datetime.apply(lambda x: datetime.strptime(x[0], "%m/%d/%Y").date())

trip_time = trip_datetime.apply(lambda x: x[1])

trip_hour = trip_time.apply(lambda x: int(x.split(':')[0]))

trip['hour'] = trip_hour

trip['date'] = trip_date
# Alternative way to judge whether a day is a business day.

# Codes borrowed from https://www.kaggle.com/currie32/a-model-to-predict-number-of-daily-trips

from pandas.tseries.holiday import USFederalHolidayCalendar

from pandas.tseries.offsets import CustomBusinessDay



calendar = USFederalHolidayCalendar()

holidays = calendar.holidays(start=trip.date.min(), end=trip.date.max())



#Find all of the business days in our time span

us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

business_days = pd.DatetimeIndex(start=trip.date.min(), end=trip.date.max(), freq=us_bd)

business_days = pd.to_datetime(business_days, format='%Y/%m/%d').date
def isbday(x, bday=business_days):

    if x in bday:

        return True

    else:

        return False
# To expedite the process, the judgement of whether a day is a business day will be applied on unique days and then mapped to all rows in trip data.

trip_date_unique = pd.Series(trip_date.unique())

trip_BDay_unique = trip_date_unique.apply(lambda x: isbday(x))



trip_date_dict = dict(zip(trip_date_unique, trip_BDay_unique))

trip_BDay = trip_date.apply(lambda x: trip_date_dict[x])
# Create a new column in trip data for business day judgement.

trip['BDay'] = trip_BDay

print('There are {} business days in the record.'.format(len(trip_date[trip_BDay==True].unique())))

print('There are totally {} days in the record.'.format(len(trip_date.unique())))
# Define a function to visualize the number of trips per hour in order to find a reasonable way to divide the day.

def trip_hour_freq(trip_hour, trip_date, trip_day_idx, data_idx, ax, xlabel):

    # Slice the trip_hour, count each value, sort the count series by index and then plot it.

    trip_counts = trip_hour[trip_day_idx & data_idx].value_counts().sort_index() # Calculate the number of trips for each hour

    freq_day = trip_counts/len(trip_date[trip_day_idx].unique()) # Calculate the frequency per day

    ax.bar(freq_day.index, freq_day) # Plot the trip hours.

    ax.set_xlabel(xlabel)

    ax.set_ylabel('Trip Frequency Per Day')

    ax.set_xticks(freq_day.index) # Display all the xticks, which are 24 hours.
# Counts for subscribers and non-subscribers

trip['subscription_type'].value_counts()
# Show bar charts for user behaviors on during business days and rest days.

fig, axes = plt.subplots(3, 2, figsize=(20,10))



trip_hour_freq(trip_hour, trip_date, trip_BDay==True, trip.subscription_type=='Subscriber', axes[0][0], 'Subscriber on Business Day')

trip_hour_freq(trip_hour, trip_date, trip_BDay==False, trip.subscription_type=='Subscriber', axes[0][1], 'Subscriber on non-Business Day')



trip_hour_freq(trip_hour, trip_date, trip_BDay==True, trip.subscription_type=='Customer', axes[1][0], 'Customer on Business Day')

trip_hour_freq(trip_hour, trip_date, trip_BDay==False, trip.subscription_type=='Customer', axes[1][1], 'Customer on non-Business Day')



trip_hour_freq(trip_hour, trip_date, trip_BDay==True, True, axes[2][0], 'All on Business Day')

trip_hour_freq(trip_hour, trip_date, trip_BDay==False, True, axes[2][1], 'All on non-Business Day')



plt.show()
# Create a new column to record weekdays

trip['weekday'] = pd.to_datetime(trip['date']).dt.weekday
# Plot the trip counts for each weekday. 0 is Monday and 6 is Sunday.

trip.groupby('weekday')['id'].count().plot('bar')

plt.show()
# Create a new column for month.

trip['month'] = trip_datetime.apply(lambda x: int(x[0].split('/')[0]))
# Group the trip first by subscription type and then by month.

trip_monthly = trip.groupby(['subscription_type', 'month'])['id'].count()
# View the number of trips on each month.

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].bar(trip_monthly['Subscriber'].index, trip_monthly['Subscriber'])

axes[0].set_xlabel('Month (Subscriber)')

axes[1].bar(trip_monthly['Customer'].index, trip_monthly['Customer'])

axes[1].set_xlabel('Month (Customer)')

plt.show()
# Group the trips first by region and then by hour.

trip_groupby_station = trip.groupby(['station_region', 'hour'])['id'].count()
# Set the hour out as a feature column. Keep the station region as the index.

trip_station_region = trip_groupby_station.reset_index(level=1)
# The station groups in the trip data match those in the weather data.

print(trip_groupby_station.index.levels[0])

print(regions)
trip.groupby(['station_region'])['id'].count()
# Plot the trips per hour for each city.

fig, axes = plt.subplots(1, 5, figsize=(20,5))

i = 0

for region in regions:

    region_counts = trip_station_region.loc[region]

    axes[i].bar(region_counts.hour, region_counts.id, color='b')

    axes[i].set_xlabel(region_names[region])

    i += 1

plt.show()
# Get the mean trip duration for each region in minute.

bike_durations = trip.groupby('station_region')['duration'].mean()/60

bike_durations_median = trip.groupby('station_region')['duration'].median()/60

trip_region_counts = trip.groupby('station_region')['id'].count()
for region, duration in bike_durations.iteritems():

    print('There are {0} trips in {1}, lasting {2:.1f} minutes in average.'.format(trip_region_counts[region], region_names[region], duration))
for region, duration in bike_durations_median.iteritems():

    print('There are {0} trips in {1}, lasting {2:.1f} minutes in average.'.format(trip_region_counts[region], region_names[region], duration))
bike_durations.index = bike_durations.index.to_series().map(region_names)

bike_durations_median.index = bike_durations.index
plt.figure(figsize=(10, 5))



plt.subplot(121)

bike_durations.plot.bar()

plt.ylabel('Average Trip Duration (min)')



plt.subplot(122)

bike_durations_median.plot.bar()

plt.ylabel('Median Trip Duration (min)')

plt.show()
trip_duration = trip.duration.sort_values()
div = 10 # Set the number of pieces to divide the trip_duration

trip_duration_idx = np.linspace(0, len(trip_duration), num=div, endpoint=False, dtype='int')

division_points = trip_duration.iloc[trip_duration_idx[1:]]/60

print("All trips can be divided into {} pieces with minute division points at:".format(div))

print(['%.1f' % elem for elem in division_points])
trip_duration_subscriber = trip_duration[trip.subscription_type=='Subscriber'].mean()/60

trip_duration_customer = trip_duration[trip.subscription_type=='Customer'].mean()/60



print('The average trip duration for a subscriber is %.1f minutes.' % trip_duration_subscriber)

print('The average trip duration for a non-subscriber is %.1f minutes.' % trip_duration_customer)
# Calculate the bike usage based on the bike id.

bike_usage = trip.bike_id.value_counts()
# It would be interesting to figure out where the most used bikes come from and the potential lifetime of the bike

plt.figure(figsize=(15,6))

plt.plot(bike_usage.index, bike_usage, '.')

plt.ylabel("Number of Trips")

plt.xlabel("Bike ID")

plt.title("Bike Usage View")

plt.show()
# Get the bike ids 

most_used_bikes = bike_usage[bike_usage > 1500].index

least_used_bikes = bike_usage[bike_usage < 500].index
# Get the bike trips

most_used_bikes_trips = trip[trip.bike_id.isin(most_used_bikes)]

least_used_bikes_trips = trip[trip.bike_id.isin(least_used_bikes)]
# Group by the station regions

most_used_bikes_group = most_used_bikes_trips.groupby('station_region')['bike_id'].count()

least_used_bikes_group = least_used_bikes_trips.groupby('station_region')['bike_id'].count()
# Change the index from zip code to city name

most_used_bikes_group.index = most_used_bikes_group.index.to_series().map(region_names)

least_used_bikes_group.index = least_used_bikes_group.index.to_series().map(region_names)
plt.figure(figsize=(15, 5))



plt.subplot(121)

most_used_bikes_group.plot.bar(logy=True)

plt.ylabel('Logarithm of Trip Counts')

plt.title('Trips with Most Used Bikes')



plt.subplot(122)

least_used_bikes_group.plot.bar()

plt.ylabel('Trip Counts')

plt.title('Trip Counts with Least Used Bikes')

plt.show()
station.head()
# Calculate the number of docks in each region

region_dock_counts = station.groupby('region')['dock_count'].sum()

print(region_dock_counts)
region_dock_counts.index = region_dock_counts.index.to_series().map(region_names)
least_used_bikes_group / region_dock_counts
# Fill NaN with an empty string.

trip.zip_code.fillna('', inplace=True)
# Display bay area with 3-digit zip codes.

## display(Image(filename='Zipcode_Map.png'))
# Sectional Center Facility: 

# Obtain zip codes (first three digits) for bay area from online source: http://maps.huge.info/zip3.htm

zip3 = ['948', '945', '947', '946', '941', '944', '940', '943', '950']
# Create a new column with True and False values indicating whether the rider is from Bay Area or not (roughly).

trip['local_zip'] = trip.zip_code.apply(lambda x: x[0:3] in zip3 and len(x)>=5)
# Calculate the ratio of the local zip number and the total zip number

local_percentage = trip.groupby('subscription_type')['local_zip'].sum() / trip.subscription_type.value_counts() * 100

print('{:.2f}% of the subscribers are local people.'.format(local_percentage[1]))

print('{:.2f}% of the non-subscribers are local people.'.format(local_percentage[0]))
# Obtain the daily trips groupby the business day column

daily_trips = trip.groupby('date').agg({'BDay': ['median', 'count']})
# Separate the business and non-business day trips

BDay_trips = daily_trips[daily_trips['BDay']['median'] == True].reset_index()

nonBDay_trips = daily_trips[daily_trips['BDay']['median'] == False].reset_index()
# Plot the trip counts per day 

fig, ax = plt.subplots(figsize=(20, 8))

plt.plot(BDay_trips.date, BDay_trips['BDay']['count'], 'bo', label='Business Day')

plt.plot(nonBDay_trips.date, nonBDay_trips['BDay']['count'], 'rx', label='Non-business Day')



# Set the x axis so that every month will be displayed

months = mdates.MonthLocator()

year_month_Fmt = mdates.DateFormatter('%y/%m')

ax.xaxis.set_major_locator(months)

ax.xaxis.set_major_formatter(year_month_Fmt)



plt.legend()

plt.ylabel('Trip Counts')

ax.grid(True)





plt.show()
# Find out the outliers of the non-business day trips. Four are found on the above graph.

nonBDay_outliers = nonBDay_trips.loc[nonBDay_trips['BDay'].sort_values('count').iloc[-4:].index]

print(nonBDay_outliers)
nonBDay_outliers.date
# https://pypi.python.org/pypi/holidays

# Apparently some companies don't have Veterans Day and Columbus Day off, and this seems to be a common situation.

## us_holidays = holidays.US()

## [us_holidays.get_list(hol) for hol in nonBDay_outliers.date.tolist()]
# Exclude these two holidays from the non-business days.

trip.loc[trip.date.isin(nonBDay_outliers.date), 'BDay'] = True
# Data before 2013/10 should be treated differently

before_Oct2013 = np.count_nonzero(trip.date < datetime(2013, 10, 1).date())

print("Percentage of trips happening before 2013/10 is {:.2%}".format(before_Oct2013/len(trip)))
# Clearly the patterns before and after october are different. In fact, I can just delete the data, but it is also fine to just label it.

trip['before_Oct2013'] = trip.date < datetime(2013, 10, 1).date()
# There are no nan values in the trip data.

trip.isnull().sum()
# Create a dictionary of station to its originally designed dock counts.

# Create two more columns for the status data: the originally designed dock counts and the real number of docks.

station_docks = dict(zip(station.id, station.dock_count))

status['dock_count'] = status.station_id.apply(lambda x: station_docks[x])

status['total'] = status['bikes_available'] + status['docks_available']
# Check situations when the sum of bikes and docks is larger than the originally designed number of docks.

more_docks = status.loc[status.total - status.dock_count > 0, :]

more_docks.groupby(['station_id', 'total']).count()
# Show both the head and the tail of station 22. 

# It shows that the number of docks for station 22 is actually 27 in the beginning and the end. The number of dock count for station 22 needs to be updated.

status[status.station_id == 22].iloc[np.r_[0:5, -5:0]]
# Show both the head and the tail of station 39.

# It shows that the number of docks for station 39 is 19 in the beginning and the end. No change. The larger count could be just a system error.

status[status.station_id == 39].iloc[np.r_[0:5, -5:0]]
# Update the station 22 dock count to 27

station_22_index = (station.id == 22).nonzero()[0][0]

station.loc[station_22_index, 'dock_count'] = 27

station_docks = dict(zip(station.id, station.dock_count))

status['dock_count'] = status.station_id.apply(lambda x: station_docks[x])
# Check situations when the sum of bikes and docks is larger than the originally designed number of docks.

more_docks = status[status.total - status.dock_count > 0]

more_docks.groupby(['station_id', 'total']).count()
# Count the number of status changes when the sum of available bikes and docks is smaller than the actual dock count, which shouldn't change in the two years' operation.

less_docks = status.loc[status.total - status.dock_count < 0, :]

less_docks_counts = less_docks.groupby(['station_id'])['total'].count()

less_docks_station_id = less_docks_counts.index
fig, ax = plt.subplots(figsize=(15, 5))

for region, color in zip(region_stations_dict, ['b', 'r', 'k', 'g', 'y']):

    station_docks_count = less_docks_counts.loc[region_stations_dict[region]]

    ax.bar(station_docks_count.index, station_docks_count, color=color, label=region_names[region])



plt.xlabel('Station ID')

plt.ylabel('Status Changes')

plt.title('Counts of status changes on suspected dock malfunction')

plt.xticks(range(0, max(less_docks_station_id)+1, 2))

plt.legend()

plt.show()
# Demonstrate how to obtain the hour of the status data.

print(less_docks.time.iloc[1])

print(int(less_docks.time.iloc[1].split()[1].split(':')[0]))
less_docks['hour'] = less_docks.time.apply(lambda x: int(x.split()[1].split(':')[0]))
less_docks_counts_hour = less_docks.groupby('hour')['time'].count()



plt.figure(figsize=(10, 5))

less_docks_counts_hour.plot.bar(color='blue')

plt.ylabel('Counts of dock counts discrepancy')

plt.show()
status['hour'] = status.time.apply(lambda x: int(x.split()[1].split(':')[0]))
%%time

status['datetime'] = pd.to_datetime(status.time)

status['date'] = status.datetime.apply(lambda x: x.date())

status['weekday'] = status.datetime.dt.weekday
# Get the total dock count change events per station

station_status_list = []

for station_id in status.station_id.unique():

    station_status = status[status.station_id == station_id]

    station_status_list.append(station_status[station_status.total.diff() != 0])
# Concatenate the list to form the full dataframe

dock_changes = pd.concat(station_status_list)

print(dock_changes.shape)

print('The total dock count change events occurs in {:.2f}% of the total status changes.'.format(len(dock_changes) / len(status) * 100))
plt.figure(figsize=(10, 5))

dock_changes.groupby('hour')['total'].count().plot.bar(color='green')

plt.ylabel('Counts of total dock count change events')

plt.show()
# Get the number of station status changes from the trip data. Start suggests the decrease of available bikes. End suggests the increase.

trip_status_start = trip.loc[:, ['start_date', 'weekday']]

trip_status_end = trip.loc[:, ['end_date', 'weekday']]



trip_status_start['start_hour'] = trip_status_start.loc[:, 'start_date'].apply(lambda x: int(x.split()[1].split(':')[0]))

trip_status_end['end_hour'] = trip_status_end.loc[:, 'end_date'].apply(lambda x: int(x.split()[1].split(':')[0]))
# Calculate the event counts per hour for trip data

trip_status_start_count = trip_status_start.groupby(['weekday', 'start_hour'])['start_hour'].count()

trip_status_end_count = trip_status_end.groupby(['weekday', 'end_hour'])['end_hour'].count()
# Calculate the change of number of the available bikes

status['bike_diff'] = status.bikes_available.diff()
# Calculate the bike count increase and decrease in status data

bike_incr = status.loc[status.bike_diff > 0, :]

bike_decr = status.loc[status.bike_diff < 0, :]
# Calculate the event counts per hour for status data

bike_incr_count = bike_incr.groupby(['weekday', 'hour'])['bike_diff'].sum()

bike_decr_count = - bike_decr.groupby(['weekday', 'hour'])['bike_diff'].sum() # All bike_diff values are negative. Need to be negated.
# Calculate the extra trips in status data than in trip data

bike_incr_diff = bike_incr_count - trip_status_end_count

bike_decr_diff = bike_decr_count - trip_status_start_count
bike_incr_diff_reset = bike_incr_diff.reset_index()

bike_decr_diff_reset = bike_decr_diff.reset_index()



bike_incr_diff_reset.columns = ['weekday', 'hour', 'counts']

bike_decr_diff_reset.columns = ['weekday', 'hour', 'counts']
number_of_weeks = len(trip.date.unique()) / 7

bike_incr_diff_reset.counts = bike_incr_diff_reset.counts / number_of_weeks # Calculate the averaged daily extra trips

bike_decr_diff_reset.counts = bike_decr_diff_reset.counts / number_of_weeks # Calculate the averaged daily extra trips



# Convert into a 2D table.

bike_incr_diff_pivot = bike_incr_diff_reset.pivot(index='hour', columns='weekday', values='counts')

bike_decr_diff_pivot = bike_decr_diff_reset.pivot(index='hour', columns='weekday', values='counts')
# Generate the contour plot.

X = bike_incr_diff_pivot.columns.values

Y = bike_incr_diff_pivot.index.values

Z_incr = bike_incr_diff_pivot.values

Z_decr = bike_decr_diff_pivot.values



Xi,Yi = np.meshgrid(X, Y)

plt.figure(figsize=(15,10))



plt.subplot(211)

plt.xlabel('Hour')

plt.ylabel('Weekday')

plt.title('Extra trips when the number of bikes increases')

plt.xticks(range(0, 24, 1))

plt.yticks(range(0, 7, 1))

plt.contourf(Yi, Xi, Z_incr, 300, cmap=plt.cm.jet)

plt.colorbar()



plt.subplot(212)

plt.xlabel('Hour')

plt.ylabel('Weekday')

plt.title('Extra trips when the number of bikes decreases')

plt.xticks(range(0, 24, 1))

plt.yticks(range(0, 7, 1))

plt.contourf(Yi, Xi, Z_decr, 300, cmap=plt.cm.jet)

plt.colorbar()



plt.show()
print('There are {:.1f}% more trips with bike count increase in status data.'.format((bike_incr.bike_diff.sum() - len(trip_status_end)) / len(trip_status_end) * 100))

print('There are {:.1f}% more trips with bike count decrease in status data.'.format((- bike_decr.bike_diff.sum() - len(trip_status_start)) / len(trip_status_start) * 100))
# Clean up the status data so that only available bike changes are used. It contributes more than 99% to the data.

bike_avail_changes = status[status.bikes_available.diff() != 0]

len(bike_avail_changes) / len(status)
bike_incr_diff_hour = bike_incr_diff_reset.groupby('hour')['counts'].sum()

bike_decr_diff_hour = bike_decr_diff_reset.groupby('hour')['counts'].sum()
# Plot the feature importances from the daily, am, pm trip models

# Matplotlib code borrowed from: https://chrisalbon.com/python/matplotlib_grouped_bar_plot.html

X_tick_labels = bike_incr_diff_hour.index

pos = np.array(range(len(X_tick_labels)))



plt.figure(figsize=(20, 7))

ax = plt.subplot(111)

width = 0.25

ax.bar(pos-0.5*width, bike_incr_diff_hour, width=width, color='#7BC024', label='Bike Count Increase')

ax.bar(pos+0.5*width, bike_decr_diff_hour, width=width, color='#17C9BF', label='Bike Count Decrease')

ax.legend()



ax.set_xlabel('Hour')

ax.set_ylabel('Extra status changes')

ax.set_xticks(pos)

ax.set_xticklabels(X_tick_labels)



# Adding the legend and showing the plot

plt.legend(loc='upper right')

plt.grid()

plt.show()
%%time

# Read all the status data that have zero bikes available.

conn = sqlite3.connect('../input/database.sqlite')

cursor = conn.cursor()

no_bikes = pd.read_sql_query(f"SELECT * FROM status WHERE bikes_available = 0 ORDER BY time", con=conn)

conn.close()
no_bikes.head()
# Get the business day

no_bikes['datetime'] = pd.to_datetime(no_bikes.time)

no_bikes['date'] = no_bikes.datetime.apply(lambda x: x.date())

no_bikes['hour'] = no_bikes.time.apply(lambda x: int(x.split()[1].split(':')[0])) # Get the hours in a day.



no_bikes_date_unique = pd.Series(no_bikes.date.unique())

no_bikes_BDay_unique = no_bikes_date_unique.apply(lambda x: isbday(x))



no_bikes_date_dict = dict(zip(no_bikes_date_unique, no_bikes_BDay_unique))

no_bikes['BDay'] = no_bikes.date.apply(lambda x: no_bikes_date_dict[x])
# Assign the regions to stations.

no_bikes['station_region'] = None

no_bikes_region_dict = {} # Initiate a dictionary from the region code to stations.

station_num_dict = {}

group_id = 0



for group in id_groups:

    region_code = region_dict[group_id]

    region_idx = no_bikes.station_id.isin(group)

    

    no_bikes_region_dict[region_code] = group    

    station_num_dict[region_code] = len(group)

    

    no_bikes.loc[region_idx, 'station_region'] = region_code

    group_id += 1
# Divide the group into BDay and non-BDay

no_bikes_BDay = no_bikes.loc[no_bikes.BDay == True, :]

no_bikes_nonBDay = no_bikes.loc[no_bikes.BDay == False, :]
# Calculate the empty minutes per hour per region in a single day.

number_of_BDays = no_bikes_BDay_unique.sum() # Calculate the number of business days.

number_of_nonBDays = len(no_bikes_date_unique) - number_of_BDays



region_hour_BDay = no_bikes_BDay.groupby(['station_region', 'hour'])['bikes_available'].count() / number_of_BDays

region_hour_nonBDay = no_bikes_nonBDay.groupby(['station_region', 'hour'])['bikes_available'].count() / number_of_nonBDays
# Plot empty station events per station per hour.

region_count = 1

plt.figure(figsize=(12, 12))

for region in region_hour_BDay.index.levels[0]:

    plt.subplot(5, 1, region_count) 

    

    # Show the minutes count per station   

    (region_hour_BDay[region] / station_num_dict[region]).plot.bar(label=region_names[region]) 

    

    plt.ylabel('Empty Station (min)')

    plt.xlabel('')

    plt.legend()

    region_count += 1
# Plot empty station events per station per hour.

region_count = 1

plt.figure(figsize=(12, 12))

for region in region_hour_nonBDay.index.levels[0]:

    plt.subplot(5, 1, region_count) 

    

    # Show the per station minutes count    

    (region_hour_nonBDay[region] / station_num_dict[region]).plot.bar(label=region_names[region]) 

    

    plt.ylabel('Empty Station (min)')

    plt.xlabel('')

    plt.legend()

    region_count += 1
%%time

# Read all the status data that have zero docks available.

conn = sqlite3.connect('../input/database.sqlite')

cursor = conn.cursor()

station_full = pd.read_sql_query(f"SELECT * FROM status WHERE docks_available = 0 ORDER BY time", con=conn)

conn.close()
# The ratio of station full over empty

len(station_full) / len(no_bikes)
station_full['datetime'] = pd.to_datetime(station_full.time)

station_full['date'] = station_full.datetime.apply(lambda x: x.date())

station_full['hour'] = station_full.time.apply(lambda x: int(x.split()[1].split(':')[0])) # Get the hours in a day.
# Assign the regions to stations.

station_full['station_region'] = None

station_full_region_dict = {} # Initiate a dictionary from the region code to stations.

station_num_dict = {}

group_id = 0



for group in id_groups:

    region_code = region_dict[group_id]

    region_idx = station_full.station_id.isin(group)

    

    station_full_region_dict[region_code] = group    

    station_num_dict[region_code] = len(group)

    

    station_full.loc[region_idx, 'station_region'] = region_code

    group_id += 1
# Calculate the full minutes per hour per region in a single day.

number_of_days = len(station_full.date.unique())



region_hour_full = station_full.groupby(['station_region', 'hour'])['bikes_available'].count() / number_of_days
# Plot full station events per station per hour.

region_count = 1

plt.figure(figsize=(12, 12))

for region in region_hour_full.index.levels[0]:

    plt.subplot(5, 1, region_count) 

    

    # Show the minutes count per station   

    (region_hour_full[region] / station_num_dict[region]).plot.bar(label=region_names[region]) 

    

    plt.ylabel('Full Station (min)')

    plt.xlabel('')

    plt.legend()

    region_count += 1
# Status data don't have nan values.

status.isnull().sum()
weather.head(1)
weather.shape
# Precipitation column is not numeric!!

precip = weather.precipitation_inches

precip.dtype
# Show unique values in precipitation data

precip.unique()
print('{:.2}% of the weather data are labeled "T" for precipitation.'.format(len(weather[precip=='T'])/len(weather) * 100))

print('{:.2}% of the weather data are labeled "nan" for precipitation.'.format(len(weather[precip.isnull()])/len(weather) * 100))
weather.loc[precip=='T', 'precipitation_inches'] = 0.005
# Only one entry contains NaN value. This will be kept to analyze later after the merge into the complete datasets.

weather[precip.isnull()]
weather.precipitation_inches = weather.precipitation_inches.astype('float')

print(weather.precipitation_inches.dtype)
# Summarize the number of NaNs in each column.

weather.isnull().sum()
# Function from more_itertools.consecutive_groups

# https://more-itertools.readthedocs.io/en/latest/api.html#more_itertools.consecutive_groups

from itertools import groupby

from operator import itemgetter



def consecutive_groups(iterable, ordering=lambda x: x):

    """Yield groups of consecutive items using :func:`itertools.groupby`.

    The *ordering* function determines whether two items are adjacent by

    returning their position.

    By default, the ordering function is the identity function. This is

    suitable for finding runs of numbers:

        >>> iterable = [1, 10, 11, 12, 20, 30, 31, 32, 33, 40]

        >>> for group in consecutive_groups(iterable):

        ...     print(list(group))

        [1]

        [10, 11, 12]

        [20]

        [30, 31, 32, 33]

        [40]

    For finding runs of adjacent letters, try using the :meth:`index` method

    of a string of letters:

        >>> from string import ascii_lowercase

        >>> iterable = 'abcdfgilmnop'

        >>> ordering = ascii_lowercase.index

        >>> for group in consecutive_groups(iterable, ordering):

        ...     print(list(group))

        ['a', 'b', 'c', 'd']

        ['f', 'g']

        ['i']

        ['l', 'm', 'n', 'o', 'p']

    """

    for k, g in groupby(

        enumerate(iterable), key=lambda x: x[0] - ordering(x[1])

    ):

        yield map(itemgetter(1), g)
def replace_nan(data, col):

    col_nan = data.loc[data[col].isnull(), col] # Get the nan entries of the specified column

    nan_idx = col_nan.index # Get the index of the col_nan

    

    # Group adjacent indices from col_nan_idx according to the method from the following link:

    # https://stackoverflow.com/questions/2154249/identify-groups-of-continuous-numbers-in-a-list

    for group in consecutive_groups(nan_idx):

        col_idx = list(group)

        

        # Set the mean of the adjacent six values as the value for the NaN

        weather.loc[col_idx, col] = weather.loc[(col_idx[0]-3):(col_idx[-1]+3), col].mean()
# The replace_nan function will be applied to all numeric columns except the gust column

nan_cols = ['max_temperature_f', 'mean_temperature_f', 'min_temperature_f',

       'max_dew_point_f', 'mean_dew_point_f', 'min_dew_point_f',

       'max_humidity', 'mean_humidity', 'min_humidity',

       'max_sea_level_pressure_inches', 'mean_sea_level_pressure_inches',

       'min_sea_level_pressure_inches', 'max_visibility_miles',

       'mean_visibility_miles', 'min_visibility_miles', 'max_wind_Speed_mph',

       'mean_wind_speed_mph', 'precipitation_inches',

       'cloud_cover', 'wind_dir_degrees']
for col in nan_cols:

    replace_nan(weather, col)
# If the gust data is missing, it should mean that there is no gust.

# This is somewhat confirmed by the fact that there is no gust value of 0 in the data.

weather.loc[weather.max_gust_speed_mph == 0, 'max_gust_speed_mph']
weather.max_gust_speed_mph.fillna(0, inplace=True) # Replace all the nan values in gust data with 0.
weather.isnull().sum()
# Create a correlation table between all the features.

weather_corr_table = weather.corr(method='pearson', min_periods=1)



display(weather_corr_table)
r_squared = 0.5 # Set the coefficient of determination to be 0.5 so that more than half of the variance can be explanable.



# Obtain the correlated features.

weather_corr = []

for col in weather_corr_table.columns:

    feature = weather_corr_table[col]

    corr_cols = feature[(feature.pow(2) > r_squared) & (feature != 1)]

    corr_idx = corr_cols.index.values.tolist()

    corr_dict = {col: corr_idx}

    if corr_idx != []:

        weather_corr.append(corr_dict)
display(weather_corr)
# Explore the categorical column

weather.events.unique()
weather.loc[weather.events == 'rain', 'events'] = 'Rain'

print(weather.events.unique())
# Create a weather data with independent columns and ONEHOT the events column.

weather_ind = pd.get_dummies(weather.drop(labels=['max_temperature_f', 'min_temperature_f', 'max_dew_point_f', 'mean_dew_point_f', 'max_humidity', 'min_humidity', 'max_sea_level_pressure_inches', 'min_sea_level_pressure_inches'], axis=1), columns=['events'])
print('The correlation between the max wind speed and the max gust speed is: \n')

for region in regions:

    regionWeather = weather[weather.zip_code == region]

    corrCoef = pearsonr(regionWeather.max_wind_Speed_mph, regionWeather.max_gust_speed_mph)[0]

    print('{} : {:.2f}'.format(region_names[region], corrCoef))
plot_count = 1

plt.figure(figsize=(15, 20))

for region in regions:

    weather_region = weather.loc[weather.zip_code == region, :]

    plt.subplot(5, 1, plot_count)

    plt.scatter(weather_region.max_wind_Speed_mph, 

                weather_region.max_gust_speed_mph, 

                label='{}'.format(region_names[region]))

    plt.xticks(np.arange(0, 45, 5))

    plt.xlim([0, 45])

    plt.xlabel('Max Wind (mph)')

    plt.ylabel('Max Gust (mph)')

    plot_count += 1

    plt.legend()

    plt.grid()
weather_ind.columns
weather_ind.columns= ['date', 'Temp', 'Dew', 'Humid', 'Pressure', 'Max_Vis', 'Mean_Vis', 'Min_Vis', 'Max_Wind', 'Mean_Wind', 'Gust', 'Precip', 'Cloud', 'Wind_Deg',  'zip_code', 'Fog', 'Fog_Rain', 'Rain', 'Thunder']
weather_ind.head(1)
# Previous exploration suggests that the wind degree column has an outlier point. I will correct this point before making the plots.

weather_ind[weather_ind.Wind_Deg > 360]
# This seems to be a typo with an extra '7' in the middle.

weather_ind.loc[weather_ind.Wind_Deg > 360, 'Wind_Deg'] = 272
weather_month = weather_ind.date.apply(lambda x: int(x.split('/')[0]))
# Plot all the numeric feature columns versus months

plot_cols = weather_ind.columns[1:14]



plt.figure(figsize=(25, 15))

plot_idx = 1

for col in plot_cols:

    plt.subplot(3, 5, plot_idx)

    plt.scatter(weather_month, weather_ind[col])

    plt.xlabel('Month')

    plt.title(col)

    plot_idx += 1
# The conversion is necessary for the combination with other data

weather_ind.date = weather_ind.date.apply(lambda x: datetime.strptime(x, "%m/%d/%Y").date())
trip.columns
# Drop columns that not relevant for both datasets.

drop_cols = ['id', 'start_date', 'start_station_name', 'start_station_id', 'end_date', 'end_station_name', 'end_station_id', 'bike_id', 'zip_code']

trip_use = trip.drop(drop_cols, axis=1)
print(trip_use.shape)

trip_use.head(1)
# Only consider durations less than 12 hours as valid trips.

duration_12hours = 60*60*12

trip_final = trip_use[trip_use.duration <= duration_12hours]
print('{:.1f}% of trips are within 12 hours.'.format(len(trip_final) / len(trip_use) * 100))
trip_final.isnull().sum()
print(weather_ind.shape)

weather_ind.head(1)
weather_ind.isnull().sum()
trip_final.columns
trip_rgs = trip_final[['date', 'station_region', 'hour', 'month', 'BDay', 'weekday', 'before_Oct2013']]
trip_rgs.head()
# Define a function to be used in the below cell.

def region_weather_trips(data, weather):

    region_data = data.groupby('date')['BDay', 'weekday', 'month', 'before_Oct2013'].median()

    region_data['trip_counts'] = data.groupby('date')['hour'].count()   

    region_data.columns=['BDay', 'weekday', 'month', 'before_Oct2013', 'trip_counts']

    # region_data.rename(columns={'median': 'BDay', 'count': 'trip_counts'}, inplace=True)

    

    # Merge the region data with weather data. The inner join is used as some dates may not be in the region data and the weather data of the missing dates shouldn't be included in the merged data.

    region_weather_merge = pd.merge(region_data, weather, left_index=True, right_on='date') 

    region_weather_merge.set_index('date', inplace=True) # Set the date as the index after merging

    region_weather_merge.drop('zip_code', axis=1, inplace=True)

    

    return region_weather_merge
# Generate five datasets as a dictionary with the zip code as the key.

trip_count_rgs = {}

for group in regions:

    region = trip_rgs[trip_rgs.station_region == group].drop('station_region', axis=1) # Obtain region by zip code

    region_weather = weather_ind[weather_ind.zip_code == group] # Obtain region weather by zip code

    

    region_daily = region_weather_trips(region, region_weather)

    region_am = region_weather_trips(region[region.hour < 13], region_weather)

    region_pm = region_weather_trips(region[region.hour > 12], region_weather)

    

    trip_count_rgs[group] = [region_daily, region_am, region_pm]

    

    # Use OneHot for both the weekday and the month as both columns are actually categorical features rather than numeric features. This may not affect the decision tree algorithms, but will help with the linear regression algorithms.

    # trip_count_rgs[group] = pd.get_dummies(region_daily, columns=['weekday', 'month'])

    # trip_count_rgs[group] = region_daily
len(trip_count_rgs)
# Some regions don't have data for all dates. There are totally 733 days. This also confirms the inner merge.

print(len(trip_count_rgs[94041][0]))

print(len(trip_count_rgs[95113][0]))

print(len(weather)//5) # Five regions.
# Convert the installation date to datetime object

station.installation_date = pd.to_datetime(station.installation_date, format = "%m/%d/%Y").dt.date
# Calculate the total dock counts for each region on each day

# The count_docks function is used to map the date (index) to the total docks.

def count_docks(date):

    return sum(s_group[s_group.installation_date <= date].dock_count)



for region, group in trip_count_rgs.items():

    s_group = station[station.region == region]

    group[0]['total_docks'] = group[0].index.map(count_docks) # group[0] is the daily trip

    group[1]['total_docks'] = group[1].index.map(count_docks) # group[1] is the morning trip

    group[2]['total_docks'] = group[2].index.map(count_docks) # group[2] is the afternoon trip
trip_final.head(1)
weather_ind.head(1)
# Merge the trip and weather data based on the date and the region.

trip_weather_merge = pd.merge(trip_final, weather_ind, left_on=['date', 'station_region'], right_on=['date', 'zip_code'])
# Check the sizes of the data before and after merge

print(trip_final.shape)

print(weather_ind.shape)

print(trip_weather_merge.shape)
# Remove the date column. All features about a day have been extracted, so the date itself is not relevant.

# Remove the zip_code column as it is a redundant column as the station_region column.

trip_weather_merge.drop(['date', 'zip_code'], axis=1, inplace=True)
# Create the dataset for classfication problem

subscriber_cls = pd.get_dummies(trip_weather_merge, columns=['station_region'])
print('{} rows of the classification data contain the NaN values.'.format(len(subscriber_cls[subscriber_cls.isnull().any(axis=1)])))
trip_rgs.head()
# Show an example of a full dataset in a region with data of daily trip counts.

trip_count_rgs[94107][0].head(1)
# Display the regions in the datasets.

print(trip_count_rgs.keys())
region_names[94107]
# A function to predict the label with a certain regressor

def rgs_pred(trip_data, rgs):

    features = trip_data.drop('trip_counts', axis=1)

    counts = trip_data.trip_counts

    

    trainX = features.loc[X_train.index]

    trainY = counts.loc[y_train.index]

    testX = features.loc[X_test.index]

    

    rgs.fit(trainX, trainY)

    

    return rgs.predict(testX)
# This regressor is the one found to have the best performance.

gbr = GradientBoostingRegressor(learning_rate = 0.3,

                                n_estimators = 50,

                                max_depth = 8,

                                min_samples_leaf = 3,

                                random_state = random_state)



for region in regions:

    region_trips = trip_count_rgs[region][0]

    

    region_features = region_trips.drop('trip_counts', axis=1)

    region_counts = region_trips.trip_counts

    

    X_train, X_test, y_train, y_test = train_test_split(region_features, region_counts, test_size=0.2, random_state=random_state)

    

    y_pred = rgs_pred(region_trips, gbr)

    median_daily_trips = np.median(y_train)

    y_benchmark = np.ones(len(y_test)) * median_daily_trips

    average_total_docks = np.mean(X_train.total_docks)

    

    print('For {} (median {:.0f} trips daily with {:.0f} docks):'.format(region_names[region], median_daily_trips, average_total_docks))

    print('The median absolute error for the regular prediction is {:.1f}.'.format(median_absolute_error(y_test, y_pred)))

    print('The median absolute error for the benchmark prediction is {:.1f}. \n'.format(median_absolute_error(y_test, y_benchmark)))

    # print('The root mean square logarithmic error is {:.1f}%.'.format(np.sqrt(mean_squared_log_error(y_test, y_pred))*100))

    # print('The root mean square logarithmic error is {:.1f}%. \n\n'.format(np.sqrt(mean_squared_log_error(y_test, y_benchmark))*100))
# Collect all numeric features in the trip data

trip_num_features = ['weekday', 'month', 'Temp', 'Dew', 'Humid', 'Pressure', 'Max_Vis', 'Mean_Vis', 'Min_Vis', 'Max_Wind', 'Mean_Wind', 'Gust', 'Precip', 'Cloud', 'Wind_Deg', 'total_docks']
# Investigate the trips in San Francisco. The numeric features are normalized to accommodate all the regressors.

trips_SF = trip_count_rgs[94107][0]

features_SF = trips_SF.drop('trip_counts', axis=1)

features_SF[trip_num_features] = MinMaxScaler().fit_transform(features_SF[trip_num_features])

counts_SF = trips_SF.trip_counts



X_train, X_test, y_train, y_test = train_test_split(features_SF, counts_SF, test_size=0.2, random_state=random_state)
mae_scorer = make_scorer(median_absolute_error, greater_is_better=False) # mae: median absolute error

# Define a clear cross validation set

# Shuffle Split is used instead of KFold. A better method for a small dataset.

cv_sets = ShuffleSplit(n_splits=15, test_size = 0.20, random_state=random_state) 

# cv_sets = KFold(n_splits=15, shuffle=True, random_state=random_state)
# The original functions comes from: https://www.kaggle.com/currie32/a-model-to-predict-number-of-daily-trips/notebook

def scoring(rgs):

    scores = cross_val_score(rgs, X_train, y_train, cv=cv_sets, n_jobs=1, scoring = mae_scorer)

    return np.mean(scores)
# Check the performance of all regressors with default setting

regressors = [DummyRegressor,

              AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor,

              GaussianProcessRegressor,

              HuberRegressor, Lasso, LinearRegression, PassiveAggressiveRegressor, RANSACRegressor, Ridge, SGDRegressor, TheilSenRegressor,

              KernelRidge,

              KNeighborsRegressor,

              MLPRegressor,

              LinearSVR, NuSVR, SVR,

              DecisionTreeRegressor, ExtraTreeRegressor,

              XGBRegressor]



rgs_dict = {} # Create a regressor dictionary to record the regressor with its score

for rgs in regressors:

    begTime = time() # Get the beginning time

    

    # All parameters are set by default.

    # Unfortunately, not all regressors have a random_state parameter, so each run will give a slightly different ranking.

    try:

        rgs_function = rgs(random_state=random_state) 

    except:

        rgs_function = rgs()

        

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        rgs_error = scoring(rgs_function) * (-1) # Error is opposite of score

        

    rgs_dict[rgs.__name__] = rgs_error

    

    useTime = time() - begTime

    print('The regressor {} takes {:.2f} seconds and has a median absolute error of {:.1f}.'.format(rgs.__name__, useTime, rgs_error))    
# Plot the scores for all the regressor

pd.Series(rgs_dict).sort_values(ascending=False).plot.barh(figsize=(15, 10), grid=True, fontsize=12)

plt.xticks(np.arange(0, 600, 50))

plt.xlabel('Mean Median Absolute Error from Cross Validation')

plt.show()
rfr_params = {'n_estimators':[20, 30, 40], 'min_samples_leaf':[1, 2, 3]}

br_params = {'n_estimators':[20, 30, 40], 'max_samples':[0.2, 0.5, 1.0], 'max_features':[0.2, 0.5, 1.0]}

gbr_params = {'learning_rate':[0.02, 0.05, 0.08], 'n_estimators':[150, 200, 250], 'min_samples_leaf':[3, 4, 5], 'max_depth':[8, 9, 10]}
regressors = [RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor] # Note that the regressor needs to be a function rather than the name itself.

rgs_params = [rfr_params, br_params, gbr_params]



best_rgs = {}

for rgs, rgs_params in zip(regressors, rgs_params):

    

    grid = GridSearchCV(estimator = rgs(random_state=random_state), param_grid = rgs_params, scoring = mae_scorer, cv = cv_sets, n_jobs=-1, verbose=1)

    grid.fit(X_train, y_train)

    

    print('It takes {:.1f} seconds to grid search the regressor {}. The median absolute error is {:.1f}.'.format(useTime, rgs.__name__, (-1) * grid.best_score_))

    best_rgs[rgs.__name__] = grid.best_estimator_
best_rgs
bestRegressor = best_rgs['GradientBoostingRegressor']
# Predict the daily trips using daily data

bestRegressor.fit(X_train, y_train)

SF_daily_y_pred = bestRegressor.predict(X_test)



print('The median absolute error is {:.1f} with the best estimator using the daily trip data.'.format(median_absolute_error(y_test, SF_daily_y_pred)))
def pred_trip_count(trip_data, bestRegressor, train_index, test_index):

    features = trip_data.drop('trip_counts', axis=1)

    features[trip_num_features] = MinMaxScaler().fit_transform(features[trip_num_features])

    counts = trip_data.trip_counts

    

    trip_X_train = features.loc[train_index]

    trip_y_train = counts.loc[train_index]

    trip_X_test =features.loc[test_index]

    

    bestRegressor.fit(trip_X_train, trip_y_train)

    

    # feature_importance = pd.Series(dict(zip(trip_X_train.columns, bestRegressor.feature_importances_)))

    

    return bestRegressor.predict(trip_X_test), bestRegressor.feature_importances_
SF_am = trip_count_rgs[94107][1]

SF_pm = trip_count_rgs[94107][2]



train_index = X_train.index

test_index = X_test.index
# Predict the daily trips using morning and afternoon data

SF_am_y_pred, SF_am_importance = pred_trip_count(SF_am, bestRegressor, train_index, test_index)

SF_pm_y_pred, SF_pm_importance = pred_trip_count(SF_pm, bestRegressor, train_index, test_index)

SF_ampm_y_pred = SF_am_y_pred + SF_pm_y_pred
# Calculate the median value of the y train as the benchmark prediction

y_pred_benchmark = np.ones(len(y_test)) * y_train.median()
# Check the median absolute error metric.

print('The median absolute error is {:.1f} with the best estimator using the daily trip data.'.format(median_absolute_error(y_test, SF_daily_y_pred)))

print('The median absolute error is {:.1f} with the best estimator using the am+pm trip data.'.format(median_absolute_error(y_test, SF_ampm_y_pred)))

print('The median absolute error is {:.1f} with the benchmark model using the daily trip data.'.format(median_absolute_error(y_test, y_pred_benchmark)))
# Check the root mean square logarithmic error metric.

print('The root mean square logarithmic error is {:.4f} with the best estimator using the daily trip data.'.format(np.sqrt(mean_squared_log_error(y_test, SF_daily_y_pred))))

print('The root mean square logarithmic error is {:.4f} with the best estimator using the am+pm trip data.'.format(np.sqrt(mean_squared_log_error(y_test, SF_ampm_y_pred))))

print('The root mean square logarithmic error is {:.4f} with the benchmark using the daily trip data.'.format(np.sqrt(mean_squared_log_error(y_test, y_pred_benchmark))))
# daily_importance = pd.Series(dict(zip(X_train.columns, bestRegressor.feature_importances_)))

daily_importance = bestRegressor.feature_importances_
# Plot the feature importances from the daily, am, pm trip models

# Matplotlib code borrowed from: https://chrisalbon.com/python/matplotlib_grouped_bar_plot.html

X_tick_labels = X_train.columns

pos = np.array(range(len(X_tick_labels)))



plt.figure(figsize=(20, 5))

ax = plt.subplot(111)

width = 0.25

ax.bar(pos-width, daily_importance, width=width, color='#EE3224', label='daily')

ax.bar(pos, SF_am_importance, width=width, color='#F78F1E', label='morning')

ax.bar(pos+width, SF_pm_importance,width=width,color='#FFC222', label='afternoon')

ax.legend()



# Set the y axis label

ax.set_ylabel('Feature Importance')



# Set the chart's title

ax.set_title('Feature Importance from Gradient Boosting Regressor')



# Set the position of the x ticks

ax.set_xticks(pos)



# Set the labels for the x ticks

ax.set_xticklabels(X_tick_labels, {'rotation': 'vertical', 'fontsize': 14})



# Adding the legend and showing the plot

plt.legend(loc='upper right')

plt.grid()

plt.show()
daily_trip_count = daily_trips['BDay']['count']

daily_wind_degree = trips_SF['Wind_Deg']

daily_pressure = trips_SF['Pressure']
# Plot the daily trip count vs. the wind degree

plt.plot(daily_wind_degree, daily_trip_count, '.')

plt.xlabel('Wind Degree')

plt.ylabel('Daily Trip Count')

plt.title('Correlation between Trips and Wind Degree')

plt.show()
# Picture from this link: http://snowfence.umn.edu/Components/winddirectionanddegreeswithouttable3.htm

## display(Image(filename='WindDirection.png', width=600))
plt.plot(daily_pressure, daily_trip_count, '.')

plt.xlabel('Sea Level Pressure')

plt.ylabel('Daily Trip Count')

plt.title('Correlation between Trips and Sea Level Pressure')

plt.show()
importance_series = pd.Series(dict(zip(X_train.columns, daily_importance))).sort_values(ascending=False).reset_index()

importance_series.columns = ['feature_regular', 'importance_regular']

display(importance_series)
# before_time_split = trips_SF.index < time_split

# after_time_split = trips_SF.index >= time_split



split_index = int(features_SF.shape[0] * 0.8) + 1

print('The splitting date is {}.'.format(features_SF.index[split_index]))



X_train_time = features_SF.iloc[:split_index]

y_train_time = counts_SF[:split_index]



X_test_time = features_SF[split_index:]

y_test_time = counts_SF[split_index:]
print('The percentage of the training set is {:.1f}% of the entire dataset.'.format(len(X_train_time) / len(trips_SF) * 100))
gbr_params
gbr_params_time = {'learning_rate': [0.05, 0.06, 0.07],

                   'max_depth': [5, 6, 7],

                   'min_samples_leaf': [1, 2, 3],

                   'n_estimators': [50, 100, 150]}
# Grid search

rgs_time = GradientBoostingRegressor(random_state=random_state)

cv_sets_time = TimeSeriesSplit(n_splits=15)

grid_time = GridSearchCV(estimator = rgs_time, param_grid = gbr_params_time, scoring = mae_scorer, cv = cv_sets_time, n_jobs=-1, verbose=1)

grid_time.fit(X_train_time, y_train_time)

    

print('The grid search of the Gradient Boosting Regressor gives the median absolute error {:.1f} for the best estimator.'.format((-1) * grid_time.best_score_))
# Best model for time series training vs. regular training sets.

bestRegressor_time = grid_time.best_estimator_



print('The best regressor for time-series splitted training set is:')

print(bestRegressor_time)

print('\n')

print('The best regressor for regularly splitted training set is:')

print(bestRegressor)
y_pred_time = bestRegressor_time.predict(X_test_time)

# Calculate the median value of the y train as the benchmark prediction

y_pred_benchmark_time =  np.ones(len(y_test_time)) * y_train_time.median()



print('The median absolute error is {:.1f} with the best estimator for the time-series splitted test set.'.format(median_absolute_error(y_test_time, y_pred_time)))

print('The median absolute error is {:.1f} with the benchmark for the time-series splitted test set.\n'.format(median_absolute_error(y_test_time, y_pred_benchmark_time)))

print('The root mean square logarithmic error is {:.4f} with the best estimator for the time-series splitted test set.'.format(np.sqrt(mean_squared_log_error(y_test_time, y_pred_time))))

print('The root mean square logarithmic error is {:.4f} with the benchmark for the time-series splitted test set.'.format(np.sqrt(mean_squared_log_error(y_test_time, y_pred_benchmark_time))))
print('The median absolute error is {:.1f} with the best estimator using the daily trip data.'.format(median_absolute_error(y_test, SF_daily_y_pred)))

print('The median absolute error is {:.1f} with the benchmark model using the daily trip data.'.format(median_absolute_error(y_test, y_pred_benchmark)))

print('The root mean square logarithmic error is {:.4f} with the best estimator using the daily trip data.'.format(np.sqrt(mean_squared_log_error(y_test, SF_daily_y_pred))))

print('The root mean square logarithmic error is {:.4f} with the benchmark using the daily trip data.'.format(np.sqrt(mean_squared_log_error(y_test, y_pred_benchmark))))
# Show the robustness of the model

scores = cross_val_score(rgs_time, X_train_time, y_train_time, cv=cv_sets, n_jobs=-1, scoring = mae_scorer)
-scores
importance_time = pd.Series(dict(zip(X_train_time.columns, bestRegressor_time.feature_importances_))).sort_values(ascending=False).reset_index()

importance_time.columns = ['feature_time', 'importance_time']

display(pd.concat([importance_time, importance_series], axis=1))
# https://stackoverflow.com/questions/44974360/how-to-visualize-an-sklearn-gradientboostingclassifier

# Pick a decision tree number from the 50 estimators of the bestRegressor_time.

sub_tree_16 = bestRegressor_time.estimators_[16, 0]
# Display the decision tree map. Follow the instruction from the following link:

# https://medium.com/@rnbrown/creating-and-visualizing-decision-trees-with-python-f8e8fa394176

# Change the node color following the instruction from this link:

# https://stackoverflow.com/questions/43214350/color-of-the-node-of-tree-with-graphviz-using-class-names/43218264#43218264



dot_data = StringIO()

export_graphviz(sub_tree_16, out_file=dot_data, max_depth=3,

                feature_names=X_train_time.columns,

                filled=True, rounded=True, rotate=True,

                proportion=True, special_characters=True)

## graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

## graph.write_png('sub_tree_16.png')

## Image(graph.create_png(), width=800)
# LassoCV implementation

# http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html

lassocv_time = LassoCV(cv = cv_sets_time, random_state=random_state).fit(X_train_time, y_train_time)

print('The optimal Lasso model has the alpha {:.2f}.'.format(lassocv_time.alpha_))
# bestLasso = Lasso_grid_time.best_estimator_

y_pred_Lasso = lassocv_time.predict(X_test_time)



print('The median absolute error is {:.1f} with the best Lasso for the time-series splitted test set.'.format(median_absolute_error(y_test_time, y_pred_Lasso)))

print('The median absolute error is {:.1f} with the benchmark for the time-series splitted test set.\n'.format(median_absolute_error(y_test_time, y_pred_benchmark_time)))

print('The root mean square logarithmic error is {:.4f} with the best Lasso for the time-series splitted test set.'.format(np.sqrt(mean_squared_log_error(y_test_time, y_pred_Lasso))))

print('The root mean square logarithmic error is {:.4f} with the benchmark for the time-series splitted test set.'.format(np.sqrt(mean_squared_log_error(y_test_time, y_pred_benchmark_time))))
# Show the robustness of the model

scores = cross_val_score(lassocv_time, X_train_time, y_train_time, cv=cv_sets, n_jobs=-1, scoring = mae_scorer)
-scores
# Obtain the coefficients from the Lasso fitting.

importance_Lasso = pd.Series(dict(zip(X_train_time.columns, lassocv_time.coef_))).reset_index()

importance_Lasso['abs_coef'] = np.abs(importance_Lasso[0])

importance_Lasso.columns = ['features', 'coefficient', 'abs_coef']

importance_Lasso_sorted = importance_Lasso.sort_values('abs_coef', ascending=False).set_index('features')
importance_Lasso_sorted.coefficient
# Calculate the median of nonzero values in the feature column.

feature_mean = []

for col in X_train_time[importance_Lasso_sorted.index]:

    feature_col = X_train_time[col]

    feature_mean.append(feature_col.iloc[feature_col.nonzero()[0]].mean())



importance_Lasso_sorted['mean_adjusted'] = importance_Lasso_sorted['coefficient'] * feature_mean

importance_Lasso_sorted['abs_mean_adjusted'] = importance_Lasso_sorted['abs_coef'] * feature_mean
importance_Lasso_sorted.plot.bar(figsize=(20, 5), ylim=(-600,600), grid=True, fontsize=14)

plt.xlabel('')

plt.ylabel('Coefficient')

plt.show()
# Collect all the features that have the coefficient equal to zero. These features are considered not very relevant and will be dropped.

features_to_drop = importance_Lasso_sorted.index[importance_Lasso_sorted.coefficient == 0].tolist()

print(features_to_drop)
X_train_time_drop = X_train_time.drop(features_to_drop, axis=1)

X_test_time_drop = X_test_time.drop(features_to_drop, axis=1)
X_train_time_drop.head(1)
gbr_params_time
gbr_params_drop = {'learning_rate': [0.06, 0.07, 0.08],

                   'max_depth': [5, 6, 7],

                   'min_samples_leaf': [1, 2, 3],

                   'n_estimators': [150, 200, 250]}
# Do grid search for Gradient Boosting Regressor again to find the optimal hyperparameters.

# With less features, it allows to allocate more computation resources to do the grid search.

rgs_time_drop = GradientBoostingRegressor(random_state=random_state)

grid_time_drop = GridSearchCV(estimator = rgs_time_drop, param_grid = gbr_params_drop, scoring = mae_scorer, cv = cv_sets_time, n_jobs=-1, verbose=1)

grid_time_drop.fit(X_train_time_drop, y_train_time)



print('The grid search of the Gradient Boosting Regressor with irrelevant features dropped gives the median absolute error {:.1f} for the best estimator.'.format((-1) * grid_time_drop.best_score_))
# Compare with the full training set.

bestRegressor_time_drop = grid_time_drop.best_estimator_

y_pred_time_drop = bestRegressor_time_drop.predict(X_test_time_drop)



print('The median absolute error is {:.1f} with features dropped.'.format(median_absolute_error(y_test_time, y_pred_time_drop)))

print('The median absolute error is {:.1f} for the full time-series splitted test set.'.format(median_absolute_error(y_test_time, y_pred_time)))

print('The median absolute error is {:.1f} with the benchmark for the time-series splitted test set.\n'.format(median_absolute_error(y_test_time, y_pred_benchmark_time)))



print('The root mean square logarithmic error is {:.4f} with features dropped.'.format(np.sqrt(mean_squared_log_error(y_test_time, y_pred_time_drop))))

print('The root mean square logarithmic error is {:.4f} for the full time-series splitted test set.'.format(np.sqrt(mean_squared_log_error(y_test_time, y_pred_time))))

print('The root mean square logarithmic error is {:.4f} with the benchmark for the time-series splitted test set.'.format(np.sqrt(mean_squared_log_error(y_test_time, y_pred_benchmark_time))))
grid_time_drop.best_params_
# Show the datasets.

subscriber_cls.head(3)
# Select feature columns that have numeric values except those with only 0 and 1. 

trip_num_features_cls = ['duration', 'hour', 'weekday', 'month', 'Temp', 'Dew', 'Humid', 'Pressure', 'Max_Vis', 'Mean_Vis', 'Min_Vis', 'Max_Wind', 'Mean_Wind', 'Gust', 'Precip', 'Cloud', 'Wind_Deg']

len(trip_num_features_cls)
# Show the distribution of data for each feature.

subscriber_cls[trip_num_features_cls].plot(kind='box', subplots=True, layout=(3,6), figsize=(20, 10))

plt.show()
# Apply the logarithm to the duration feature. Then scale all features to the range of (0, 1).

cls_scaler = MinMaxScaler() # Give this a name to use the inverse transform later

cls_scaled = subscriber_cls.copy()

cls_scaled['duration'] = np.log(cls_scaled['duration'])

cls_scaled[trip_num_features_cls] = cls_scaler.fit_transform(cls_scaled[trip_num_features_cls])
# Display the box-and-whisker plot for the scaled data. Now the data look more reasonable.

cls_scaled[trip_num_features_cls].plot(kind='box', figsize=(20, 5))

plt.show()
# Generate the feature (from the scaled data) and label columns.

features_cls = cls_scaled.drop('subscription_type', axis = 1)

subType_cls = (cls_scaled.subscription_type == 'Customer').astype('int')
features_cls.head(1) # Show the feature columns
# Split the 'features_cls' and 'subType' data into training and testing sets

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(features_cls, subType_cls, test_size=0.2, random_state=random_state)



# Show the results of the split

print("Training set has {} samples.".format(X_train_cls.shape[0]))

print("Testing set has {} samples.".format(X_test_cls.shape[0]))
X_train_cls.shape
# Set the f0.5 as the scorer with more focus on the precision to predict the non-subscriber. The goal of this study is to figure out a special pattern for non-subscriber.

beta = 0.5

f_score = make_scorer(fbeta_score, beta=beta)



# Define a cross validation set with smaller number of split than for the classifier as even the downsampled data is still one order of magnitude than the regression data.

# This will be used for full dataset training as well.

# StratifiedKFold is used instead of the regular KFold to preserve the percentage of samples for each class. This is important for imbalanced datasets.

cv_sets_cls = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
# Define a function to downsample the data by removing adjacent similar samples.

# The squared sum of element-wise difference between adjacent samples is calculated. Only those data with the squared sum above the threshold will be kept.

# Iterate the process until the size of the data doesn't change more than min_diff from the previous run.

def downsample(data, threshold, min_diff):

    before_size = len(data)

    after_size = 0

    while(before_size - after_size > min_diff):

        before_size = len(data)

        data = data[data.diff().pow(2).sum(axis=1) > threshold]

        after_size = len(data)

    return data
# Divide the X_train_cls data by the subscription type

train_subs = X_train_cls[y_train_cls == 0]

train_cust = X_train_cls[y_train_cls == 1]
# Find the number of unique values in each column

unique_dict = {}

for col, data in X_train_cls.iteritems():

    unique_dict[col] = len(data.unique())
# Display the uniqueness of all columns

unique_dict
# Sort the column by the order of uniqueness. The less unique values a column has, the earlier the column will get sorted.

# These features will contribute to square sum of difference substantially: even difference from one such feature will contribute 1 to the total.

# Sorting the data with these features will help downsample the data to a small size while maintaining the sample variance necessary for finding good classifiers.

col_sort = sorted(unique_dict, key=unique_dict.get)



subs_sort = train_subs.sort_values(col_sort)

cust_sort = train_cust.sort_values(col_sort)
len(trip_num_features_cls)
dev = 0.03

threshold = (len(trip_num_features_cls) * dev) ** 2

print(threshold)
%%time

# Don't set the min_diff too small, otherwise the downsample function will take a longer time to converge.

# Downsampling is done for data with different subscription type, respectively.

subs_sort_down = downsample(subs_sort, threshold=threshold, min_diff=10)

cust_sort_down = downsample(cust_sort, threshold=threshold, min_diff=10)
# Combine the two downsized data to form the full data.

X_train_down = pd.concat([subs_sort_down, cust_sort_down])

y_train_down = y_train_cls.loc[X_train_down.index]



down_sample_size = len(X_train_down)

print('After downsampling, the data size shrinks by {:.1f}% to {}.'.format((1-down_sample_size/len(X_train_cls))*100, down_sample_size))
# The original functions comes from: https://www.kaggle.com/currie32/a-model-to-predict-number-of-daily-trips/notebook

# Use all CPUs for the computation.

# The mean is calculated instead of the median to be more comparable with the results from the GridSearchCV

def down_scoring(cls):

    scores = cross_val_score(cls, X_train_down, y_train_down, cv=cv_sets_cls, n_jobs=-1, scoring=f_score)

    return np.mean(scores)
%%time

# Check the performance of all regressors with default setting

# Too slow or gives error: GaussianProcessClassifier, RadiusNeighborsClassifier. 

classifiers = [DummyClassifier,

               AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier,

               LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier,

               GaussianNB,

               KNeighborsClassifier, NearestCentroid,

               MLPClassifier,

               LinearSVC, NuSVC, SVC,

               DecisionTreeClassifier, ExtraTreeClassifier,

               XGBClassifier]



cls_dict = {} # Create a classifier dictionary to record the classifier with its score

for cls in classifiers:

    begTime = time() # Get the beginning time

    

    # All parameters are set by default

    # Unfortunately, not all classifiers have a random_state classifier, so the ranking may change for different runs.

    try:

        cls_function = cls(random_state=random_state)

    except:

        cls_function = cls()

    

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        cls_score = down_scoring(cls_function)    

        

    useTime = time() - begTime

    

    cls_dict[cls.__name__] = (cls_score, useTime)

    print('The classifier {} takes {:.2f} seconds and has a F0.5 score of {:.3f}.'.format(cls.__name__, useTime, cls_score))    
df_cls = pd.DataFrame.from_dict(cls_dict, orient='index')

df_cls.columns = ['f_score', 'time_consumption']

df_cls.time_consumption = df_cls.time_consumption / 60 # Change the unit to minute

df_cls_sort = df_cls.sort_values('f_score')
df_cls_sort.plot.barh(figsize=(15, 10), fontsize=14)

plt.xlabel('F0.5_Score / Time Consumption (min)')

plt.show()
# First select those classifiers that perform at least better than the mean. Then normalize values of the selected classifiers.

df_cls_2 = df_cls_sort.loc[df_cls_sort.f_score > df_cls_sort.f_score.mean(), :]

df_cls_2['f_score_norm'] = minmax_scale(df_cls_2.f_score)
# Give more emphasis on the performane by taking the cubic of each value.

# Bagging Classifier ranks in the top three in both groups.

(df_cls_2.f_score_norm ** 3 / df_cls_2.time_consumption).sort_values(ascending=False)
# The XGBClassifier document can be found here: http://xgboost.readthedocs.io/en/latest/python/python_api.html

# Tuning instruction: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

bc_params = {'n_estimators':[10, 30, 50], 'max_samples':[0.4, 0.5, 0.6, 0.7]} # For Bagging Classifier, the number of estimators shouldn't be much larger than the default (10), otherwise the training time can be untolerable.

lr_params = {'penalty': ['l1', 'l2'], 'C': np.linspace(0.1, 1.0, 10)} # For Logistic Classifier

rc_params = {'alpha': np.linspace(0.1, 1.0, 20)} # For Ridge Classifier
# Initiate a dictionary to hold the best classifiers

best_cls = {}
# Randomized search will be utilized to save time as this dataset, though downsized, is still one order of magnitude than the regression set.

classifiers = [BaggingClassifier, LogisticRegression, RidgeClassifier] # Note that the classifier needs to be a function rather than the name itself.

cls_params = [bc_params, lr_params, rc_params]

n_iter_search = 10



for cls, cls_params in zip(classifiers, cls_params):

    begTime = time()

    

    grid_down = RandomizedSearchCV(estimator=cls(random_state=random_state), param_distributions=cls_params, n_iter=n_iter_search, scoring = f_score, cv = cv_sets_cls, n_jobs=-1, verbose=1)

    # The whole downsampled data is considered a "training set" for the full set, so all data will be used to capture the unique patterns.

    grid_down.fit(X_train_down, y_train_down)

    

    useTime = time() - begTime

    

    print('It takes {:.1f} seconds to grid search the classifier {}. The median F0.5 score is {:.3f}.'.format(useTime, cls.__name__, grid_down.best_score_))

    best_cls[cls.__name__] = grid_down.best_estimator_
# Define a customized evaluation function for the xgboost cross valiation -- the F0.5 score.

# https://ajourneyintodatascience.quora.com/Custom-evaluation-function-and-early-stopping-for-xgboost-with-k-fold-validation-Python

def xgb_f_score(y_predicted, dtrain):

    y_true = dtrain.get_label() # Use get_label() to obtain the y_true. Label means the label column.

    y_pred = np.round(y_predicted) # The y_predicted is the sigmoid output. Rounding it will produce binary results.

    score = fbeta_score(y_true, y_pred, beta=0.5)



    return 'fbeta', score # Must return ('name', value)
# Don't set the n_jobs to -1 as the GPU will be used instead of the CPU.

# The mean score is calculated to be comparable with the results from the GridSearchCV.

def xgb_scoring(cls, X_train, y_train):

    scores = cross_val_score(cls, X_train, y_train, cv=cv_sets_cls, n_jobs=-1, scoring=f_score)

    return np.mean(scores)
# First find the best number of estimators

# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

# Use the customized evaluation function xgb_f_score. Remember to set the maximize to True.

# Set the folds to be cv_sets_cls so that the result is reproducible. cv_sets_cls is a stratified K-Fold.

# Note that the scoring is on the whole training dataset.

def modelfit(alg, X_train, y_train, useTrainCV=True, folds=cv_sets_cls, feval=xgb_f_score, stopping=50, verbose_eval=True):

    

    if useTrainCV:

        print('Start the cross validation...')

        xgb_param = alg.get_xgb_params()

        xgtrain = xgb.DMatrix(X_train.values, label=y_train.values) # Convert the pandas dataframe to the xgboost data format for the cv

        

        # Unfortunately the cross-validation method in xgboost doesn't allow f_score as the metric. AUC should be a similar measure with the f_score.

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], folds=folds, 

                          feval=feval, maximize=True, early_stopping_rounds=stopping, verbose_eval=verbose_eval) 

        

        alg.set_params(n_estimators=cvresult.shape[0])

    

    print('Start the scoring...')

    xgb_f_score = cvresult['test-fbeta-mean'].max()

    

    #Print model report:

    print("\nModel Report")

    print("F0.5 Score of the whole training data: {:.4f}. \n".format(xgb_f_score))

    print(alg)

    

    return xgb_f_score
%%time

# Initiate the XGBClassifier. Use 'gpu_exact' as this is a small dataset. Using 'gpu_hist' doesn't use as many GPU memory as the 'gpu_exact'

# Note that the number of estimators is very large. This is the maximal round that the modelfit can try to find the optimal number of estimators..

## init_params = {'tree_method': 'gpu_exact', 'predictor': 'gpu_predictor'}

init_params = {'predictor': 'cpu_predictor'}

xgb_cls_0 = XGBClassifier(learning_rate= 0.1,

                          n_estimators= 1000,

                          max_depth= 5,

                          min_child_weight= 1,

                          gamma= 0,

                          subsample= 0.8,

                          colsample_bytree= 0.8,

                          reg_alpha= 0,

                          reg_lambda= 1,                          

                          scale_pos_weight= 1,

                          objective= 'binary:logistic',

                          random_state=random_state, **init_params)



# Find the best estimator number.

xgb_cls_0_score = modelfit(xgb_cls_0, X_train_down, y_train_down, folds=cv_sets_cls, verbose_eval=False)
%%time

# Note the parameters: https://github.com/dmlc/xgboost/blob/master/doc/parameter.md

# Also check here: https://github.com/dmlc/xgboost/issues/2819

# Grid search the 'max_depth' and 'min_child_weight' hyperparameters for the XGBClassifier using GPU to improve the training speed.



xgb_params1={'max_depth': range(3,10,2), 'min_child_weight': range(1,6,2)}

xgb_grid_1 = GridSearchCV(xgb_cls_0, param_grid=xgb_params1, scoring=f_score, 

                          cv=cv_sets_cls, n_jobs=-1, verbose=1)

xgb_grid_1.fit(X_train_down, y_train_down)    

xgb_cls_1 = xgb_grid_1.best_estimator_



xgb_cls_1_score = xgb_grid_1.best_score_

print('F0.5 Score of the cross validation: {:.4f}. \n'.format(xgb_cls_1_score))

print(xgb_cls_1)
xgb_cls_1_score - xgb_cls_0_score
%%time

xgb_params2 = {'max_depth':[6,7,8], 'min_child_weight':[4,5,6]}

xgb_grid_2 = GridSearchCV(xgb_cls_1, param_grid=xgb_params2, scoring=f_score, 

                          cv=cv_sets_cls, n_jobs=-1, verbose=1)

xgb_grid_2.fit(X_train_down, y_train_down)

xgb_cls_2 = xgb_grid_2.best_estimator_



xgb_cls_2_score = xgb_grid_2.best_score_

print('F0.5 Score of the cross validation: {:.4f}. \n'.format(xgb_cls_2_score))

print(xgb_cls_2)
xgb_cls_2_score - xgb_cls_1_score
# Copy all the parameters for xgb_cls_2 except setting the n_estimators to a large number.

xgb_cls_2_b = XGBClassifier(learning_rate= 0.1,

                            n_estimators= 1000,

                            max_depth= 7,

                            min_child_weight= 6,

                            gamma= 0,

                            subsample= 0.8,

                            colsample_bytree= 0.8,

                            reg_alpha= 0,

                            reg_lambda= 1,

                            scale_pos_weight=1,

                            objective= 'binary:logistic',                            

                            random_state=random_state, **init_params)



xgb_cls_2_b_score = modelfit(xgb_cls_2_b, X_train_down, y_train_down, folds=cv_sets_cls, verbose_eval=False)
xgb_cls_2_b_score - xgb_cls_2_score
%%time

xgb_params3 = {'gamma':[i/10.0 for i in range(0,5)]}

xgb_grid_3 = GridSearchCV(xgb_cls_2_b, param_grid=xgb_params3, scoring=f_score, 

                          cv=cv_sets_cls, n_jobs=-1, verbose=1)

xgb_grid_3.fit(X_train_down, y_train_down)

xgb_cls_3 = xgb_grid_3.best_estimator_



xgb_cls_3_score = xgb_grid_3.best_score_

print('F0.5 Score of the cross validation: {:.4f}. \n'.format(xgb_cls_3_score))

print(xgb_cls_3)
xgb_cls_3_score - xgb_cls_2_b_score
# Copy all the parameters for xgb_cls_2 except setting the n_estimators to a large number.

xgb_cls_3_b = XGBClassifier(learning_rate= 0.1,

                            n_estimators= 1000,

                            max_depth= 7,

                            min_child_weight= 6,

                            gamma= 0.4,

                            subsample= 0.8,

                            colsample_bytree= 0.8,

                            reg_alpha= 0,

                            reg_lambda= 1,

                            scale_pos_weight=1,

                            objective= 'binary:logistic',                            

                            random_state=random_state, **init_params)



xgb_cls_3_b_score = modelfit(xgb_cls_3_b, X_train_down, y_train_down, folds=cv_sets_cls, verbose_eval=False)
xgb_cls_3_b_score - xgb_cls_3_score
%%time

xgb_params4 = {'subsample':[i/10.0 for i in range(6,10)], 'colsample_bytree':[i/10.0 for i in range(6,10)]}

xgb_grid_4 = GridSearchCV(xgb_cls_3_b, param_grid=xgb_params4, scoring=f_score, 

                          cv=cv_sets_cls, n_jobs=-1, verbose=1)

xgb_grid_4.fit(X_train_down, y_train_down)

xgb_cls_4 = xgb_grid_4.best_estimator_



xgb_cls_4_score = xgb_grid_4.best_score_

print('F0.5 Score of the cross validation: {:.4f}. \n'.format(xgb_cls_4_score))

print(xgb_cls_4)
xgb_cls_4_score - xgb_cls_3_b_score
%%time

xgb_params5 = {'reg_alpha':[0.01, 0.1, 1, 10], 'reg_lambda':[0.01, 0.1, 1, 10]}

xgb_grid_5 = GridSearchCV(xgb_cls_4, param_grid=xgb_params5, scoring=f_score, 

                          cv=cv_sets_cls, n_jobs=-1, verbose=1)

xgb_grid_5.fit(X_train_down, y_train_down)

xgb_cls_5 = xgb_grid_5.best_estimator_



xgb_cls_5_score = xgb_grid_5.best_score_

print('F0.5 Score of the cross validation: {:.4f}. \n'.format(xgb_cls_5_score))

print(xgb_cls_5)
xgb_cls_5_score - xgb_cls_4_score
%%time

xgb_params6 = {'reg_alpha':[0.5, 1, 2, 3], 'reg_lambda':[0.5, 1, 2, 3]}

xgb_grid_6 = GridSearchCV(xgb_cls_5, param_grid=xgb_params6, scoring=f_score, 

                          cv=cv_sets_cls, n_jobs=-1, verbose=1)

xgb_grid_6.fit(X_train_down, y_train_down)

xgb_cls_6 = xgb_grid_6.best_estimator_



xgb_cls_6_score = xgb_grid_6.best_score_

print('F0.5 Score of the cross validation: {:.4f}. \n'.format(xgb_cls_6_score))

print(xgb_cls_6)
xgb_cls_6_score - xgb_cls_5_score
xgb_cls_6_b = XGBClassifier(learning_rate= 0.1,

                            n_estimators= 1000,

                            max_depth= 7,

                            min_child_weight= 6,

                            gamma= 0.4,

                            subsample= 0.6,

                            colsample_bytree= 0.6,

                            reg_alpha= 1,

                            reg_lambda= 1,

                            scale_pos_weight=1,

                            objective= 'binary:logistic',                            

                            random_state=random_state, **init_params)



xgb_cls_6_b_score = modelfit(xgb_cls_6_b, X_train_down, y_train_down, folds=cv_sets_cls, verbose_eval=False)
xgb_cls_6_b_score - xgb_cls_6_score
%%time

xgb_params7 = {'scale_pos_weight': np.linspace(0.1, 1, 10)}

xgb_grid_7 = GridSearchCV(xgb_cls_6_b, param_grid=xgb_params7, scoring=f_score, 

                          cv=cv_sets_cls, n_jobs=-1, verbose=1)

xgb_grid_7.fit(X_train_down, y_train_down)

xgb_cls_7 = xgb_grid_7.best_estimator_



xgb_cls_7_score = xgb_grid_7.best_score_

print('F0.5 Score of the cross validation: {:.4f}. \n'.format(xgb_cls_7_score))

print(xgb_cls_7)
xgb_cls_7_score - xgb_cls_6_b_score
%%time

xgb_params8 = {'scale_pos_weight': np.arange(0.51, 0.69, 0.01)}

xgb_grid_8 = GridSearchCV(xgb_cls_7, param_grid=xgb_params8, scoring=f_score, 

                          cv=cv_sets_cls, n_jobs=-1, verbose=1)

xgb_grid_8.fit(X_train_down, y_train_down)

xgb_cls_8 = xgb_grid_8.best_estimator_



xgb_cls_8_score = xgb_grid_8.best_score_

print('F0.5 Score of the cross validation: {:.4f}. \n'.format(xgb_cls_8_score))

print(xgb_cls_8)
xgb_cls_8_score - xgb_cls_7_score
# Explore different learning rates using the modelfit, in place of the GridSearchCV.

xgb_cls_9 = {}

for lr in [i/10.0 for i in range(1,4)]:

    xgb_cls_temp = XGBClassifier(learning_rate= lr,

                                 n_estimators= 1000,

                                 max_depth= 7,

                                 min_child_weight= 6,

                                 gamma= 0.4,

                                 subsample= 0.6,

                                 colsample_bytree= 0.6,

                                 objective= 'binary:logistic',

                                 reg_alpha= 1,

                                 reg_lambda= 1,

                                 scale_pos_weight= 0.58,

                                 random_state=random_state, **init_params)

    

    xgb_temp_score = modelfit(xgb_cls_temp, X_train_down, y_train_down, folds=cv_sets_cls, verbose_eval=False)    

    xgb_cls_9[lr] = (xgb_cls_temp, xgb_temp_score)
# Explore different learning rates using the modelfit, in place of the GridSearchCV.

xgb_cls_10 = {}

for lr in [i/100.0 for i in range(12, 30, 2)]:

    xgb_cls_temp = XGBClassifier(learning_rate= lr,

                                 n_estimators= 1000,

                                 max_depth= 7,

                                 min_child_weight= 6,

                                 gamma= 0.4,

                                 subsample= 0.6,

                                 colsample_bytree= 0.6,

                                 objective= 'binary:logistic',

                                 reg_alpha= 1,

                                 reg_lambda= 1,

                                 scale_pos_weight= 0.58,

                                 random_state=random_state, **init_params)

    

    xgb_temp_score = modelfit(xgb_cls_temp, X_train_down, y_train_down, folds=cv_sets_cls, verbose_eval=False)    

    xgb_cls_10[lr] = (xgb_cls_temp, xgb_temp_score)
xgb_cls_10[0.2][1] - xgb_cls_8_score 
xgb_cls = xgb_cls_10[0.2][0]
# Define a function to calculate the mean and standard deviation of nonzero values in a numpy array.

def nonzero_mean(np_array):

    array_len = len(np_array)

    mean_data_nonzero = np.zeros(array_len)

    std_data_nonzero = np.zeros(array_len)

    

    for i in range(array_len):

        nonzero_array = np_array[i][np_array[i].nonzero()]

        if nonzero_array.any() == True:

            mean_data_nonzero[i] = np.mean(nonzero_array)

            std_data_nonzero[i] = np.std(nonzero_array)

    

    return mean_data_nonzero, std_data_nonzero
def plot_learning_curve(cls, X_train, y_train, train_sizes, scoring, cv=cv_sets_cls, verbose=1):



    sizes, train_scores, test_scores = learning_curve(cls, X_train, y_train, cv=cv, 

                                                      n_jobs=-1, train_sizes=train_sizes, 

                                                      scoring=scoring, verbose=verbose)

    

    # Find the mean and standard deviation for smoothing.

    train_mean, train_std = nonzero_mean(train_scores)

    test_mean, test_std = nonzero_mean(test_scores)

    

    # Plot the learning curve 

    fig, ax = plt.subplots(figsize=(8, 10))

    plt.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')

    plt.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')

    plt.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha = 0.15, color = 'r')

    plt.fill_between(sizes, test_mean - test_std, test_mean + test_std, alpha = 0.15, color = 'g')     

    

    plt.xlabel('Number of Training Samples')

    plt.ylabel(str(scoring))

    plt.title('Learning Performance of {}'.format(cls.__class__.__name__))

    

    minorLocator = AutoMinorLocator()

    ax.yaxis.set_minor_locator(minorLocator)

    

    plt.grid(which='both')

    plt.legend(loc='upper left')

    plt.show()

    

    # return train_scores
%%time

# Generate the training set sizes increasing exponentially

# train_sizes = np.geomspace(0.1, 1.0, num=9)

train_sizes = np.linspace(0.1, 1.0, num=9)



# Suppress the warnings due to an "ill-defined" f score

# https://stackoverflow.com/questions/29086398/sklearn-turning-off-warnings

# https://docs.python.org/2/library/warnings.html#temporarily-suppressing-warnings

with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    plot_learning_curve(xgb_cls, X_train_down, y_train_down, train_sizes=train_sizes, 

                        cv=cv_sets_cls, scoring=f_score, verbose=1)
%%time

# Explore the accuracy learning curve.

plot_learning_curve(xgb_cls, X_train_down, y_train_down, train_sizes=train_sizes, 

                    cv=cv_sets_cls, scoring='accuracy', verbose=1)
xgb_cls
# Change the  use 'gpu_hist' instead of 'gpu_exact'.

## init_params_full = {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'}

init_params_full = {'predictor': 'cpu_predictor'}

xgb_cls = XGBClassifier(learning_rate= 0.2,

                        n_estimators= 16,

                        max_depth= 7,

                        min_child_weight= 6,

                        gamma= 0.4,

                        subsample= 0.6,                         

                        colsample_bytree= 0.6,

                        reg_alpha= 1,

                        reg_lambda= 1,

                        objective= 'binary:logistic',

                        scale_pos_weight= 0.58,

                        random_state=random_state, **init_params_full)



best_cls[XGBClassifier.__name__] = xgb_cls
best_cls
def train_predict(learner, X_train, y_train, X_test, y_test, best=False): 

    '''

    inputs:

       - learner: the learning algorithm to be trained and predicted on

       - X_train: features training set

       - y_train: income training set

       - X_test: features testing set

       - y_test: income testing set

    '''

    begTime = time()

    

    results = {}

    

    # Fit the learner to the training data

    start = time() # Get start time

    learner = learner.fit(X_train, y_train)

    end = time() # Get end time

    

    # Calculate the total training time

    results['train_time'] = end - start

        

    # Get the predictions on the test set(X_test), then get predictions on all training samples(X_train) using .predict()

    pred_train_size = int(1 * len(X_train))

    

    start = time() # Get start time

    predictions_test = learner.predict(X_test)

    predictions_train = learner.predict(X_train[:pred_train_size])

    end = time() # Get end time

    

    # Calculate the total prediction time

    results['pred_time'] = end - start

            

    # Compute accuracy on all training samples

    results['acc_train'] = accuracy_score(y_train[:pred_train_size], predictions_train)

        

    # Compute accuracy on test set using accuracy_score()

    results['acc_test'] = accuracy_score(y_test, predictions_test)

    

    # Compute F0.5-score on all training samples using fbeta_score()

    results['f_train'] = fbeta_score(y_train[:pred_train_size], predictions_train, beta=0.5)

        

    # Compute F0.5-score on the test set which is y_test

    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)

       

    useTime = time() - begTime

    # Success

    if best == True:

        print("Training with the classifier {} takes {:.1f} seconds.".format((learner.__class__.__name__ + '_best'), useTime))

    if best == False:

        print("Training with the classifier {} takes {:.1f} seconds.".format(learner.__class__.__name__, useTime))



    # Return the results

    return results
clf_A = XGBClassifier(random_state=random_state, **init_params_full)

clf_A_best = best_cls['XGBClassifier']



clf_B = BaggingClassifier(random_state=random_state)

clf_B_best = best_cls['BaggingClassifier']



clf_C = LogisticRegression(random_state=random_state)

clf_C_best = best_cls['LogisticRegression']



clf_D = RidgeClassifier(random_state=random_state)

clf_D_best = best_cls['RidgeClassifier']





# Collect results on the learners

results = {}

for clf in [clf_A, clf_B, clf_C, clf_D]:    

    clf_name = clf.__class__.__name__

    results[clf_name] = {}

    results[clf_name]['default'] = train_predict(clf, X_train_cls, y_train_cls, X_test_cls, y_test_cls)



    

for clf in [clf_A_best, clf_B_best, clf_C_best, clf_D_best]:    

    clf_name = clf.__class__.__name__

    results[clf_name]['best'] = train_predict(clf, X_train_cls, y_train_cls, X_test_cls, y_test_cls, best=True)        
# Visualization function borrowed from the 'finding_donors' project.

# Modify the funtion to display the performance of each classifier with its default setting and the optimized hyper-parameters using the downsampled training set.

def evaluate(results, accuracy, f1):

    """

    Visualization code to display results of various learners.

    

    inputs:

      - learners: a list of supervised learners

      - stats: a list of dictionaries of the statistic results from 'train_predict()'

      - accuracy: The score for the naive predictor

      - f1: The score for the naive predictor

    """

  

    # Create figure

    fig, ax = plt.subplots(2, 3, figsize = (12, 8))



    # Constants

    bar_width = 0.47

    # Check the color code from Google. https://www.google.com/search?q=color+%2300A0A0&oq=color+%2300A0A0&aqs=chrome..69i57.1192j0j7&sourceid=chrome&ie=UTF-8

    # colors = ['#A00000','#08A000','#0092A0', '#8A00A0']

    # colors = ['#E0F794','#F9EB90','#92B9FC', '#C092FC']

    colors = ['#8fb21e','#ffdd00','#79a9fc','#8b6ab7']

    

    # Super loop to plot four panels of data

    for k, learner in enumerate(results.keys()):

        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):

            for i, label in enumerate(['default', 'best']):

                # Creative plot code

                ax[j//3, j%3].bar(k+i*1.05*bar_width, results[learner][label][metric], width = bar_width, color = colors[k])

                ax[j//3, j%3].set_xlabel("Classifier('default', 'best')")

                ax[j//3, j%3].set_xlim((-0.5, 4.0))

    

    # Add unique y-labels

    ax[0, 0].set_ylabel("Time (in seconds)")

    ax[0, 1].set_ylabel("Accuracy Score")

    ax[0, 2].set_ylabel("F0.5-score")

    ax[1, 0].set_ylabel("Time (in seconds)")

    ax[1, 1].set_ylabel("Accuracy Score")

    ax[1, 2].set_ylabel("F0.5-score")

    

    # Add titles

    ax[0, 0].set_title("Model Training")

    ax[0, 1].set_title("Accuracy Score on Training Subset")

    ax[0, 2].set_title("F0.5-score on Training Subset")

    ax[1, 0].set_title("Model Predicting")

    ax[1, 1].set_title("Accuracy Score on Testing Set")

    ax[1, 2].set_title("F0.5-score on Testing Set")

    

    # Add horizontal lines for naive predictors

    ax[0, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')

    ax[1, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')

    ax[0, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')

    ax[1, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')

    

    # Add horizontal lines for the best predictors

    acc_test = []

    acc_train = []

    f_test = []

    f_train = []

    for _, result in results.items():

        for _, values in result.items():

            acc_test.append(values['acc_test'])

            acc_train.append(values['acc_train'])

            f_test.append(values['f_test'])

            f_train.append(values['f_train'])    

    

    ax[0, 1].axhline(y = max(acc_train), xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dotted')

    ax[0, 1].axhline(y = max(acc_test), xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'r', linestyle = 'dotted')

    ax[1, 1].axhline(y = max(acc_test), xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dotted')

    ax[0, 2].axhline(y = max(f_train), xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dotted')

    ax[0, 2].axhline(y = max(f_test), xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'r', linestyle = 'dotted')   

    ax[1, 2].axhline(y = max(f_test), xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dotted')

    

    

    # Set y-limits for score panels

    ax[0, 1].set_ylim((0, 1))

    ax[0, 2].set_ylim((0, 1))

    ax[1, 1].set_ylim((0, 1))

    ax[1, 2].set_ylim((0, 1))



    # Create patches for the legend

    patches = []

    for i, learner in enumerate(results.keys()):

        patches.append(mpatches.Patch(color = colors[i], label = learner))

    plt.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), \

               loc = 'upper center', borderaxespad = 0., ncol = 4, fontsize = 'x-large')

    

    # Aesthetics

    plt.suptitle("Performance Metrics for Four Supervised Learning Models", fontsize = 16, y = 1.10)

    plt.tight_layout()

    plt.show()
# Calculate the accuracy of a naive predictor that always predicts the subscriber as a benchmark for the accuracy.

# Calculate the f0.5 score of a naive predictor that always predicts the non-subscriber as a benchmark for the f0.5 score.

accuracy = np.sum(subType_cls)/float(subType_cls.count()) # If the classifier predicts all to be non-subscribers.

subscriber_accuracy = 1 - accuracy # If the classifier predicts all to be subscribers.

recall = 1

precision = accuracy



# TODO: Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.

# HINT: The formula above can be written as (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

beta = 0.5

fscore = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)



# Run metrics visualization for the three supervised learning models chosen

evaluate(results, subscriber_accuracy, fscore)
# For the full dataset, use 'gpu_hist' instead of 'gpu_exact'.

## init_params_full = {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'}

init_params_full = {'predictor': 'cpu_predictor'}

xgb_full = XGBClassifier(learning_rate= 0.05,

                         n_estimators=250,

                         max_depth= 9,

                         min_child_weight= 5,

                         gamma= 0,

                         subsample= 0.9,                         

                         colsample_bytree= 0.6,

                         reg_alpha= 0.5,

                         reg_lambda= 2,

                         objective= 'binary:logistic',

                         scale_pos_weight= 0.52,

                         random_state=random_state, **init_params_full)
np.geomspace(0.01, 1.0, num=9) * len(y_train_cls)
# Generate the training set sizes increasing exponentially. The whole process takes about 10 min.

train_sizes = np.geomspace(0.01, 1.0, num=9)

## plot_learning_curve(xgb_full, X_train_cls, y_train_cls, train_sizes=train_sizes, cv=cv_sets_cls, scoring=f_score, verbose=1)

## Error from Kaggle: OSError: [Errno 28] No space left on device
xgb_full
# Show the robustness of the model

## scores = cross_val_score(xgb_full, X_train_cls, y_train_cls, cv=cv_sets_cls, n_jobs=-1, scoring=f_score)

## OSError: [Errno 28] No space left on device
## scores
## scores
# rc_full is defined here as there is not enough resource on Kaggle's server.

rc_full = RidgeClassifier(alpha=0.55000000000000004, class_weight=None, copy_X=True,

                          fit_intercept=True, max_iter=None, normalize=False,

                          random_state=16, solver='auto', tol=0.001)
%%time

# Fit the training set with the best classifiers and predict on the X_test_cls data.

xgb_cls.fit(X_train_cls, y_train_cls)

y_pred_xgb_down = xgb_cls.predict(X_test_cls)

fscore_xgb_down = fbeta_score(y_test_cls, y_pred_xgb_down, beta=0.5)

print('The F0.5 score on the testing set with trained model from the downsampled dataset is {:.4f}.'.format(fscore_xgb_down))



xgb_full.fit(X_train_cls, y_train_cls)

y_pred_xgb_full = xgb_full.predict(X_test_cls)

fscore_xgb_full = fbeta_score(y_test_cls, y_pred_xgb_full, beta=0.5)

print('The F0.5 score on the testing set with the optimized XGBClassifier is {:.4f}.'.format(fscore_xgb_full))



rc_full.fit(X_train_cls, y_train_cls)

y_pred_rc_full = rc_full.predict(X_test_cls)

fscore_rc_full = fbeta_score(y_test_cls, y_pred_rc_full, beta=0.5)

print('The F0.5 score on the testing set with the optimized RidgeClassifier is {:.4f}.'.format(fscore_rc_full))
# Plot the confusion matrix with the log scale. The two classifiers have similar graphs, so I plot only one.

# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

cm = confusion_matrix(y_test_cls, y_pred_xgb_full)



classes = ['Subscriber', 'Non-Subscriber']



cmap = plt.cm.Blues

plt.imshow(np.log10(cm), interpolation='nearest', cmap=cmap)

plt.colorbar()

tick_marks = np.arange(2)

plt.xticks(tick_marks, classes, rotation=45)

plt.yticks(tick_marks, classes)



plt.tight_layout()

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()



print(cm)
# Show the classification report for both classifiers for comparison

print('A classification report for XGBClassifier trained with the downsampled: ')

print(classification_report(y_test_cls, y_pred_xgb_down))

print('\n')



print('A classification report for XGBClassifier: ')

print(classification_report(y_test_cls, y_pred_xgb_full))

print('\n')



print('A classification report for RidgeClassifier: ')

print(classification_report(y_test_cls, y_pred_rc_full))
# Plot the importance using xgboost's function. By Gain.

plt.figure(figsize = (20, 15))

ax = plt.subplot()

xgb.plot_importance(xgb_full, ax=ax, importance_type='gain', xlabel='The number of times a feature appears in a tree', height=0.5, grid=False)

xgb_importance_gain = pd.Series(xgb_full.get_booster().get_score(importance_type='gain'))

plt.show()
# Plot the importance using xgboost's function. By Cover.

plt.figure(figsize = (20, 15))

ax = plt.subplot()

xgb.plot_importance(xgb_full, ax=ax, importance_type='cover', xlabel='The number of times a feature appears in a tree', height=0.5, grid=False)

xgb_importance_cover = pd.Series(xgb_full.get_booster().get_score(importance_type='cover'))

plt.show()
# Plot the importance using xgboost's function. By Weight.

plt.figure(figsize = (20, 15))

ax = plt.subplot()

xgb.plot_importance(xgb_full, ax=ax, importance_type='weight', xlabel='The number of times a feature appears in a tree', height=0.5, grid=False)

xgb_importance_weight = pd.Series(xgb_full.get_booster().get_score(importance_type='weight'))

plt.show()
# Concatenate all the importance series together to form a dataframe

xgb_importance_df = pd.concat([xgb_importance_weight, xgb_importance_gain, xgb_importance_cover], axis=1)

xgb_importance_df.columns = ['Weight', 'Gain', 'Cover']
# Change the gain and cover scales to logarithm to minimize the data skew.

xgb_importance_df['LogGain'] = np.log10(xgb_importance_df.Gain)

xgb_importance_df['LogCover'] = np.log(xgb_importance_df.Cover)
# Normalize the three importances to 

xgb_importance_scaled = pd.DataFrame(MinMaxScaler().fit_transform(xgb_importance_df[['Weight', 'LogGain', 'LogCover']]))

xgb_importance_scaled.index = xgb_importance_df.index

xgb_importance_scaled.columns = ['Weight', 'LogGain', 'LogCover']
xgb_importance_scaled.sort_values('LogGain', ascending=False).plot.bar(figsize=(20, 5), grid=True, fontsize=14, )

plt.ylabel('Normalized Importance')

plt.show()
# Plot a decision tree from the xgboost following the instruction below:

# https://machinelearningmastery.com/visualize-gradient-boosting-decision-trees-xgboost-python/

# A pdf file will be opened to see the details. 

# The xgboost plot function for python cannot specify the maximum depth to plot.



tree_plot = xgb.to_graphviz(xgb_full, num_trees=16, rankdir='LR')

# tree_plot.view()
def dot_to_df(tree_body):

    '''Extract the nodes information from a DOT file. The function returns a pandas series with two columns: 

       left nodes and right nodes'''

    

    edge_list = []

    for line in tree_body:

        try:

            edge_left, edge_right = line.split('->')[0:2]

        except:

            continue

            

        node_left = int(edge_left)

        node_right = int(edge_right.split()[0])

        

        edge_list.append((node_left, node_right))

        

    edge_df = pd.DataFrame(edge_list)

    edge_df.columns = ['left_node', 'right_node']

    

    return edge_df
def nodes_depth(tree_body, depth = 3):

    '''This function collects the node indices up to the designated depths.'''

    # Convert the DOT body code to a pandas DataFrame with the edge (node pair)

    edge_df = dot_to_df(tree_body)

       

    d_nodes = [[0]]

    for d in range(0, depth):

        next_node = edge_df.loc[edge_df.left_node.isin(d_nodes[d]), 'right_node'].tolist()

        d_nodes.append(next_node)

        

    return [node_point for d_node in d_nodes for node_point in d_node]
def change_depth(tree_body, depth = 3):

    '''This function changes the depth of a decision tree to display via graphviz.

       It returns a new DOT code with the designated depth for a decision tree.'''

    

    nodes_deep = nodes_depth(tree_body, depth = depth)

    nodes_deeper = nodes_depth(tree_body, depth = depth + 1)

    nodes_diff = list(set(nodes_deeper)^set(nodes_deep))



    keep_line_idx = []

    for idx, line in enumerate(tree_body):

        item_list = line.split()

        left_node = int(item_list[0])



        if left_node in nodes_deep:

            keep_line_idx.append(idx)

        elif (left_node in nodes_diff) and (item_list[1] != '->'):

            keep_line_idx.append(idx)

    

    new_lines = [tree_body[i] for i in keep_line_idx]

    new_string = '\n'.join(new_lines)

    

    new_dot = '\n'.join(['digraph {', '\tgraph [rankdir=LR]', new_string, '}'])

    

    return new_dot
# Create new DOT source code

new_dot_code = change_depth(tree_plot.body, depth = 2)



# Convert the DOT source code to a graph

## graph = pydotplus.graph_from_dot_data(new_dot_code)

## graph.write_png('xgb_tree_16.png')

## Image(graph.create_png(), width=800)
# Enumerate the numeric features to a dictionary

col_dict = {col: idx for idx, col in enumerate(trip_num_features_cls)}    
# Obtain the split value histogram for the duration feature

xgb_split_duration = xgb_full.get_booster().get_split_value_histogram('duration')
def scale_col(x, col):

    '''The function scales the data back for result interpretation.'''

    scaler_min = cls_scaler.data_min_[col_dict[col]]

    scaler_max = cls_scaler.data_max_[col_dict[col]]

    return x * (scaler_max - scaler_min) + scaler_min
# Scale back the duration data and change the unit to minute

xgb_split_duration['duration_minute'] = xgb_split_duration.SplitValue.apply(lambda x: np.exp(scale_col(x, 'duration'))) / 60
# Use 5 min interval counts.

xgb_split_duration['interval_5min'] = (xgb_split_duration.duration_minute // 5 + 1) * 5

xgb_split_duration.groupby('interval_5min')['Count'].sum().plot.bar(figsize=(15, 5))

plt.show()
# Calculate the mean value when duration is 10 min.

xgb_split_duration.loc[xgb_split_duration.interval_5min == 10, 'SplitValue'].mean()
num_features = trip_num_features_cls[1:] # Don't include the duration as it has been explored above.
num_div = 5 # number of division for the whole range

interval = 1 / num_div # the interval on the normalized values



xgb_splits_max = {}

xgb_splits_max['duration'] = 10

plt_idx = 1 # The subplot index starts from 1

r_color = lambda: random.randint(0,255) # function to generate random color: https://stackoverflow.com/questions/13998901/generating-a-random-hex-color-in-python

plt.figure(figsize=(16, 16))

for col in num_features:

    # Obtain the split values from the xgboost

    split_hist = xgb_full.get_booster().get_split_value_histogram(col)

    split_hist['interval'] = (split_hist['SplitValue'] // interval + 1) * interval # Interval starting not starting from zero.

    split_hist['scaled'] = split_hist['interval'].apply(lambda x: scale_col(x, col))

    

    #xgb_splits[col] = split_hist.groupby('scaled')['Count'].sum()

    split_group = split_hist.groupby('scaled')['Count'].sum()

    xgb_splits_max[col] = split_group.idxmax()

    

    bar_width = 0.1 * (max(split_hist.scaled) - min(split_hist.scaled))

    bar_color = '#%02X%02X%02X' % (r_color(),r_color(),r_color())

    

    plt.subplot(4, 4, plt_idx)

    plt.bar(split_group.index, split_group, width=bar_width, color=bar_color)

    plt.xlabel('%s' % col)

    plt_idx += 1
# Only explore the first ten important features.

xgb_importance = pd.Series(xgb_full.feature_importances_, index=X_train_cls.columns)

xgb_splits_max = pd.Series(xgb_splits_max)

xgb_unique = pd.Series(unique_dict)



xgb_important_splits = pd.concat([xgb_importance, xgb_splits_max, xgb_unique], axis=1)

xgb_important_splits.columns = ['Importance', 'MaxSplitPoint', 'Uniqueness']

xgb_important_splits.sort_values('Importance', ascending=False)[0:10]
# Plot the feature value uniqueness vs. the feature importance and find a strong correlation.

plt.figure(figsize=(5, 5))

plt.scatter(xgb_important_splits.Importance, np.log(xgb_important_splits.Uniqueness))

plt.xlabel('Feature Importance')

plt.ylabel('Logarithm of the Number of Unique Values')

plt.show()
# Obtain the coefficients from the Lasso fitting.

importance_Ridge = pd.Series(dict(zip(X_train_cls.columns, rc_full.coef_[0]))).reset_index()

importance_Ridge['abs_coef'] = np.abs(importance_Ridge[0])

importance_Ridge.columns = ['features', 'coefficient', 'abs_coef']

importance_Ridge_sorted = importance_Ridge.sort_values('abs_coef', ascending=False).set_index('features')
# Calculate the mean of nonzero values in the feature column.

# The feature importrance will be adjusted by multipling the mean value. 

# If a feature has a value distribute at the low value end (close to 0 in (0, 1) scale), the feature tends to be overemphasized on its coefficient. It will be adjusted by multipling the feature's mean value.

feature_mean_cls = []

for col in X_train_cls[importance_Ridge_sorted.index]:

    feature_col_cls = X_train_cls[col]

    feature_mean_cls.append(feature_col_cls.iloc[feature_col_cls.nonzero()[0]].mean())



importance_Ridge_sorted['mean_adjusted'] = importance_Ridge_sorted['coefficient'] * feature_mean_cls

importance_Ridge_sorted['abs_mean_adjusted'] = importance_Ridge_sorted['abs_coef'] * feature_mean_cls
# Plot the coefficient column only to demonstrate the signs.

importance_Ridge_sorted[['coefficient', 'mean_adjusted']].plot.bar(figsize=(20, 5), ylim=(-0.8,0.8), grid=True, fontsize=14)

plt.xlabel('')

plt.show()
# Plot columns with absolute values to demonstrate the relative importance. As the coefficient differ significantly from feature to feature, logarithm is used for better observation.

importance_Ridge_sorted[['abs_coef', 'abs_mean_adjusted']].plot.bar(figsize=(20, 5), ylim=(10e-5,3), grid=True, fontsize=14, logy=True)

plt.xlabel('')

plt.show()
# Absolute adjusted coefficients below 10% of the maximum coefficient will be considered not important.

not_important = max(importance_Ridge_sorted.abs_mean_adjusted) / 10

not_important_features = importance_Ridge_sorted[importance_Ridge_sorted.abs_mean_adjusted < not_important].index
X_train_less = X_train_cls.drop(not_important_features, axis=1)

X_test_less = X_test_cls.drop(not_important_features, axis=1)
X_train_less.columns
%%time

# Fit the training set with the best classifiers and predict on the X_test_cls data.

xgb_full.fit(X_train_less, y_train_cls)

y_pred_xgb_less = xgb_full.predict(X_test_less)

fscore_xgb_less = fbeta_score(y_test_cls, y_pred_xgb_less, beta=0.5)

print('The F0.5 score on the testing set with the optimized XGBClassifier is {:.4f}.'.format(fscore_xgb_less))



rc_full.fit(X_train_less, y_train_cls)

y_pred_rc_less = rc_full.predict(X_test_less)

fscore_rc_less = fbeta_score(y_test_cls, y_pred_rc_less, beta=0.5)

print('The F0.5 score on the testing set with the optimized RidgeClassifier is {:.4f}.'.format(fscore_rc_less))
# Absolute adjusted coefficients below 10% of the maximum coefficient will be considered not important.

not_important_2 = max(xgb_importance) / 10

not_important_features_2 = xgb_importance[xgb_importance < not_important_2].index
X_train_less_2 = X_train_cls.drop(not_important_features_2, axis=1)

X_test_less_2 = X_test_cls.drop(not_important_features_2, axis=1)
X_train_less_2.shape
%%time

# Fit the training set with the best classifiers and predict on the X_test_cls data.

xgb_full.fit(X_train_less_2, y_train_cls)

y_pred_xgb_less_2 = xgb_full.predict(X_test_less_2)

fscore_xgb_less_2 = fbeta_score(y_test_cls, y_pred_xgb_less_2, beta=0.5)

print('The F0.5 score on the testing set with the optimized XGBClassifier is {:.4f}.'.format(fscore_xgb_less_2))



rc_full.fit(X_train_less_2, y_train_cls)

y_pred_rc_less_2 = rc_full.predict(X_test_less_2)

fscore_rc_less_2 = fbeta_score(y_test_cls, y_pred_rc_less_2, beta=0.5)

print('The F0.5 score on the testing set with the optimized RidgeClassifier is {:.4f}.'.format(fscore_rc_less_2))
Ridge_Importance = pd.Series(minmax_scale(importance_Ridge_sorted.abs_mean_adjusted))

Ridge_Importance.index = importance_Ridge_sorted.index
cls_importance = pd.concat([xgb_importance_scaled, Ridge_Importance], axis = 1)

cls_importance.rename(columns = {0: 'Ridge'}, inplace=True)
cls_importance.sort_values('Ridge', ascending=False).plot.bar(grid=True, figsize=(20, 5), fontsize=14)

plt.show()
# The unscaled dataset is used instead as tree-based method is not sensitive to value normalization.

features_sel = subscriber_cls.drop('subscription_type', axis = 1).drop(not_important_features, axis=1)

subType_sel = (subscriber_cls.subscription_type == 'Customer').astype('int')
features_sel.shape
# Split the 'features_sel' and 'subType_sel' data into training and testing sets

X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(features_sel, subType_sel, test_size=0.2, random_state=random_state)



# Show the results of the split

print("Training set has {} samples.".format(X_train_sel.shape[0]))

print("Testing set has {} samples.".format(X_test_sel.shape[0]))
X_train_sel.shape
## init_params_sel = {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'}

init_params_sel = {'predictor': 'cpu_predictor'}

xgb_sel = XGBClassifier(learning_rate= 0.1,

                         n_estimators=88,

                         max_depth= 7,

                         min_child_weight= 1,

                         gamma= 0,

                         subsample= 0.9,                         

                         colsample_bytree= 0.7,

                         reg_alpha= 0,

                         reg_lambda= 0,

                         objective= 'binary:logistic',

                         scale_pos_weight= 0.48,

                         random_state=random_state, **init_params_sel)
# Check the performance of the optimized XGBClassifier on the testing set.

xgb_sel.fit(X_train_sel, y_train_sel)

y_pred_xgb_sel = xgb_sel.predict(X_test_sel)

fscore_xgb_sel = fbeta_score(y_test_sel, y_pred_xgb_sel, beta=0.5)

print('The F0.5 score on the testing set with the optimized XGBClassifier is {:.4f}.'.format(fscore_xgb_sel))
xgb_sel_imWeight = pd.Series(xgb_sel.get_booster().get_score(importance_type='weight'))

xgb_sel_imGain = pd.Series(xgb_sel.get_booster().get_score(importance_type='gain'))

xgb_sel_imCover = pd.Series(xgb_sel.get_booster().get_score(importance_type='cover'))
# Concatenate all the importance series together to form a dataframe

xgb_imSel_df = pd.concat([xgb_sel_imWeight, xgb_sel_imGain, xgb_sel_imCover], axis=1)

xgb_imSel_df.columns = ['Weight', 'Gain', 'Cover']
# Change the gain and cover scales to logarithm to minimize the data skew.

xgb_imSel_df['LogGain'] = np.log10(xgb_imSel_df.Gain)

xgb_imSel_df['LogCover'] = np.log10(xgb_imSel_df.Cover)
# Normalize the three importances to 

xgb_imSel_scaled = pd.DataFrame(MinMaxScaler().fit_transform(xgb_imSel_df[['Weight', 'LogGain', 'LogCover']]))

xgb_imSel_scaled.index = xgb_imSel_df.index

xgb_imSel_scaled.columns = ['Weight', 'LogGain', 'LogCover']
Ridge_im = Ridge_Importance[Ridge_Importance >= not_important]

xgb_imSel_comp = pd.concat([xgb_imSel_scaled, Ridge_im], axis=1)

xgb_imSel_comp.rename(columns={0: 'Ridge'}, inplace=True)
xgb_imSel_comp.sort_values('Ridge', ascending=False).plot.bar(figsize=(20, 5), grid=True, fontsize=14, )

plt.ylabel('Normalized Importance')

plt.show()
# Obtain the split value histograms for duration and hour

xgb_sel_duration = xgb_sel.get_booster().get_split_value_histogram('duration') / 60

xgb_sel_hour = xgb_sel.get_booster().get_split_value_histogram('hour')
time_interval = 5

xgb_sel_duration['interval_5min'] = (xgb_sel_duration['SplitValue'] // time_interval + 1) * time_interval

xgb_sel_duration.groupby('interval_5min')['Count'].sum().plot.bar(figsize=(15, 5))

plt.show()
hour_interval = 3

xgb_sel_hour['interval_3h'] = (xgb_sel_hour['SplitValue'] // hour_interval + 1) * hour_interval

xgb_sel_hour.groupby('interval_3h')['Count'].sum().plot.bar(figsize=(15, 5))

plt.show()
xgb_sel_duration.head()
# Explore the most important components when PCA is applied to X_train_less. Four components will explain more than 80% of the variance, so four should be good enough.

pca = PCA(n_components=4, random_state=random_state).fit(X_train_less)

print('Totally explained variance is {:.3f}.'.format(sum(pca.explained_variance_ratio_)))
# Plot the PCA components.

plt_idx = 1

plt.figure(figsize=(20, 5))

for idx, comp in enumerate(pca.components_):    

    bar_color = '#%02X%02X%02X' % (r_color(),r_color(),r_color())

    

    plt.subplot(1, 4, plt_idx)

    pd.Series(comp, index=X_train_less.columns).plot.bar(sharex=True, grid=True)

    plt.title('Explain Variance Ration: {:.3f}'.format(pca.explained_variance_ratio_[idx]))

    plt_idx += 1



plt.show()
def get_available_devices():  

    local_device_protos = device_lib.list_local_devices()

    return [x.name for x in local_device_protos]



## with sys_pipes():

print(get_available_devices())
# Create a f0.5 metric function from keras following the code:

# https://www.kaggle.com/arsenyinfo/f-beta-score-for-keras

def fbeta(y_true, y_pred):

    beta = 0.5 # Set the beta to 0.5 to calculate the f0.5 score



    # just in case of hipster activation at the final layer

    y_pred = K.clip(y_pred, 0, 1)



    # shifting the prediction threshold from .5 if needed

    y_pred_bin = K.round(y_pred)



    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()

    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))

    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))



    precision = tp / (tp + fp)

    recall = tp / (tp + fn)



    beta_squared = beta ** 2

    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())
def get_best_model(model, X_train, y_train, X_cross, y_cross, cp_filepath, class_weight=None, 

                   epochs=100, batch_size=1000, cp_verbose=0, fit_verbose=0):

    '''The monitor is changed from the default 'val_loss' to 'fbeta' to monitor the F0.5 score change.

    The mode must be set to 'max' as the fbeta score increases as the performance increases. 

    https://github.com/keras-team/keras/pull/188'''

    

    checkpointer = ModelCheckpoint(filepath=cp_filepath, monitor='val_fbeta', verbose=cp_verbose, 

                                   save_best_only=True, mode='max')

    

    history = model.fit(X_train, y_train, validation_data=(X_cross, y_cross),                        

                        epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], 

                        verbose=fit_verbose, class_weight=class_weight, shuffle=True)

    

    model.load_weights(cp_filepath)

    

    return history
def deep_cv(model, X_train, y_train, X_test, y_test, cp_filepath, class_weight=None, epochs=150, 

            batch_size=1000, cp_verbose=0, fit_verbose=0):

    '''The default batch size is set to 10000, which should be good for a small neural network.'''

    cvscores = []

    iter_i = 0

    for train, test in cv_sets_cls.split(X_train, y_train):

        

        get_best_model(model, X_train.iloc[train], y_train.iloc[train], X_train.iloc[test], y_train.iloc[test], 

                       cp_filepath=cp_filepath, class_weight=class_weight, epochs=epochs, batch_size=batch_size, 

                       cp_verbose=cp_verbose, fit_verbose=fit_verbose)

        

        scores = model.evaluate(X_train.iloc[test], y_train.iloc[test], verbose=fit_verbose) # evaluate the model

        print("{}: {:.4f}".format(model.metrics_names[1], scores[1]))

        cvscores.append(scores[1])

        

        # https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model

        if cvscores[iter_i] - cvscores[iter_i - 1] > 0:

            model.save_weights(cp_filepath)

    

    model.load_weights(cp_filepath)

    test_beta_score = model.evaluate(X_test, y_test, batch_size=batch_size)[1]

    

    print('\n')

    print('The average F0.5 score of the cross validataion is {:.4f} (+/- {:.4f}).'.format(np.mean(cvscores), np.std(cvscores)))  

    print('The F0.5 score for the testing set is {:.4f}.'.format(test_beta_score))

    print('\n')
def plot_learning_history(model, X_train, y_train, cp_filepath, class_weight=None, epochs=100, 

                          batch_size=1000, cp_verbose=0, fit_verbose=0):

    

    cv_generator = cv_sets_cls.split(X_train, y_train)

    train_idx, cross_idx = next(cv_generator)

    

    history = get_best_model(model, X_train.iloc[train_idx], y_train.iloc[train_idx], 

                                    X_train.iloc[cross_idx], y_train.iloc[cross_idx], 

                             cp_filepath, class_weight=class_weight, epochs=epochs, 

                             batch_size=batch_size, cp_verbose=cp_verbose, fit_verbose=fit_verbose)

    

    model.load_weights(cp_filepath)

    

    # https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/

    plt.figure(figsize=(10, 5))

    plt.plot(history.history['val_fbeta'])

    plt.xlabel('Epoch')

    plt.ylabel('F0.5 Score')

    plt.title('Learning History')

    plt.grid()

    plt.show()

    

    print('The best F0.5 score is {:.4f}.'.format(max(history.history['val_fbeta'])))
# Set a repeatable initializer. The 'TruncatedNormal' is the recommended initializer by Keras.

kernal_init = K_init.TruncatedNormal(mean=0.0, stddev=0.05, seed=random_state)
# baseline model from: https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/

model_dim = X_train_cls.shape[1]

def create_base(dim=model_dim):

    model = Sequential()

    model.add(Dense(dim, input_dim=dim, kernel_initializer=kernal_init, activation='relu'))

    model.add(Dense(1, kernel_initializer=kernal_init, activation='sigmoid'))

    

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[fbeta])

    # model.summary()

    

    return model
%%time

# Test the simplest model trained by the downsampled dataset on the testing set.

base1 = create_base()



batch_size = max(len(y_train_down), len(y_test_cls))

cp_filepath = 'weights.best.base_model_down_noWeight.hdf5'



deep_cv(base1, X_train_down, y_train_down, X_test_cls, y_test_cls, cp_filepath, 

        class_weight=None, batch_size=batch_size, epochs=50)
# Figure out the best class_weight

pos_train_down = y_train_down.sum()

neg_train_down = len(y_train_down) -  pos_train_down

print('The ratio between the negative and positive labels is {:.1f}'.format(neg_train_down / pos_train_down))



pos_weight_down = neg_train_down / pos_train_down / 2

class_weight_down = {0:1, 1:pos_weight_down}
%%time

# Still trained with the downsampled. The sample weight added.

base2 = create_base()



cp_filepath = 'weights.best.base_model_down_withWeight.hdf5'



deep_cv(base2, X_train_down, y_train_down, X_test_cls, y_test_cls, cp_filepath, 

        class_weight=class_weight_down, batch_size=batch_size, epochs=50)
%%time

base3 = create_base()

base3.load_weights('weights.best.base_model_down_noWeight.hdf5')



batch_size = max(len(y_train_cls), len(y_test_cls))

cp_filepath = 'weights.best.base_model_full_noWeight.hdf5'



deep_cv(base3, X_train_cls, y_train_cls, X_test_cls, y_test_cls, cp_filepath, 

        class_weight=None, epochs=10, batch_size=batch_size)
pos_train_full = y_train_cls.sum()

neg_train_full = len(y_train_cls) -  pos_train_full

print('The ratio between the negative and positive labels is {:.1f}'.format(neg_train_full / pos_train_full))



pos_weight_full = neg_train_full / pos_train_full / 2

class_weight_full = {0:1, 1:pos_weight_full}
%%time

base4 = create_base()

base4.load_weights('weights.best.base_model_down_noWeight.hdf5')



cp_filepath = 'weights.best.base_model_full_withWeight.hdf5'



deep_cv(base4, X_train_cls, y_train_cls, X_test_cls, y_test_cls, cp_filepath, 

        class_weight=class_weight_full, epochs=10, batch_size=batch_size)
%%time

base5 = create_base()

base5.load_weights('weights.best.base_model_full_withWeight.hdf5') # Picking up from the model with weights.



epochs = 10 # Check 10 more epochs on top of the original 10 epochs.

batch_size = len(y_train_cls)

cp_filepath = 'weights.best.base_model_full_withWeight_moreEpochs.hdf5'



plot_learning_history(base5, X_train_cls, y_train_cls, cp_filepath=cp_filepath, epochs=epochs, 

                      class_weight=class_weight_full, batch_size=batch_size, cp_verbose=1, fit_verbose=0)
# Evaluate the testing set.

# base5.load_weights(cp_filepath)

base5.evaluate(X_test_cls, y_test_cls, batch_size=batch_size)[1]
# Set a higher learning rate for Adam (default 0.001). The deeper the network, the smaller the maximum learning rate is.

opt_Adam = Adam(lr=0.03, decay=0.0)



# Improved model from: https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/

model_dim = X_train_cls.shape[1]



best_model = Sequential()

best_model.add(Dense(model_dim * 1, input_dim=model_dim, kernel_initializer=kernal_init, activation='relu'))

best_model.add(Dense(model_dim // 2, kernel_initializer=kernal_init, activation='relu'))

best_model.add(Dense(model_dim // 2, kernel_initializer=kernal_init, activation='relu'))

best_model.add(Dense(model_dim // 3, kernel_initializer=kernal_init, activation='relu'))

best_model.add(Dense(1, kernel_initializer=kernal_init, activation='sigmoid'))



best_model.compile(loss='binary_crossentropy', optimizer=opt_Adam, metrics=[fbeta])

best_model.summary()
%%time

# Improve the parameters on the full dataset with the pre-trained model.

# This is just a demo. The best model has been saved.

epochs = 500



# The optimum batch size is about the length of the training set divided by 500, which equals to 1070.

# The class weight is set to None.

batch_size = max(len(y_train_cls), len(y_test_cls)) // 1000

cp_filepath = 'weights.best.best_model.hdf5'

plot_learning_history(best_model, X_train_cls, y_train_cls, cp_filepath=cp_filepath, 

                      class_weight=None, epochs=epochs, batch_size=batch_size, cp_verbose=True)
## best_model.load_weights('saved_models/weights.best.model_8.hdf5')
max_batch_size = len(y_train_cls)
# This is the evaluation with the default batch size.

best_model.evaluate(X_test_cls, y_test_cls)[1]
# This is the evaluation with the maximum batch size.

best_model.evaluate(X_test_cls, y_test_cls, max_batch_size)[1]
%%time

# Predict the y values with the default batch size and the maximum batch size.

# Note that the result needs to be rounded in order to compare with y_test_cls as the raw output is float from sigmoid function.

y_pred_test_MLP_default = best_model.predict(X_test_cls).round()

y_pred_test_MLP_max = best_model.predict(X_test_cls, batch_size=max_batch_size).round()
fbeta_score(y_test_cls, y_pred_test_MLP_default, beta=0.5)
fbeta_score(y_test_cls, y_pred_test_MLP_max, beta=0.5)
# Evaluation on the X_train_cls

best_model.evaluate(X_train_cls, y_train_cls, batch_size=max_batch_size)[1]
# Evaluation on the X_train_down

best_model.evaluate(X_train_down, y_train_down, batch_size=max_batch_size)[1]
%%time

MLP_importance = {}

for col in X_test_cls.columns:

    X_temp = X_test_cls.copy()

    X_temp[col] = 0

    X_temp_score = best_model.evaluate(X_temp, y_test_cls, batch_size=max_batch_size, verbose=0)[1]

    MLP_importance[col] = X_temp_score
MLP_importance_series = pd.Series(MLP_importance).sort_values()
MLP_importance_series.plot.bar(figsize=(15, 5), fontsize=14)

plt.ylabel('F0.5 score')

plt.title('Feature Impact on the Score')

plt.show()
MLP_importance_series[-2:]
# Calculate the importance gain from the least important feature importance.

MLP_impact = -(MLP_importance_series - MLP_importance_series[-1])
# Scale the data to be comparable with other algorithms.

MLP_impact_norm = pd.Series(minmax_scale(MLP_impact))

MLP_impact_norm.index = MLP_impact.index
cls_importance_all = pd.concat([cls_importance[['Ridge', 'LogGain']], MLP_impact_norm], axis = 1)

cls_importance_all.rename(columns = {0: 'MLP', 'LogGain': 'XGB'}, inplace=True)
cls_importance_all.sort_values('Ridge', ascending=False).plot.bar(grid=True, figsize=(20, 5), fontsize=14)

plt.show()
%%time

MLP_importance_series_2 = MLP_importance_series.copy()

MLP_index = MLP_importance_series.index



for col in MLP_index:

    X_temp = X_test_cls.copy()

    

    col_idx = MLP_index.get_loc(col) # Get the index of a specific column from the MLP_importance_series

    col_del = MLP_index[(col_idx+1):] # Set all columns after the specified column to value zero

    X_temp.loc[:, col_del] = 0

    

    X_temp_score = best_model.evaluate(X_temp, y_test_cls, batch_size=max_batch_size, verbose=0)[1]

    MLP_importance_series_2[col] = X_temp_score
MLP_importance_series_2.plot.bar(figsize=(15, 5), fontsize=14)

plt.ylabel('F0.5 score')

plt.title('Feature Impact on the Score')

plt.show()
# These features are not useful for the MLP model.

MLP_importance_series_2[MLP_importance_series_2 > 0.84]
# http://forums.fast.ai/t/how-could-i-release-gpu-memory-of-keras/2023/11

K.clear_session()
# Calculate the total time used for running the whole notebook.

notebook_end_time = time()

notebook_running_time = (notebook_end_time - notebook_start_time) / 60

print('Running the entire notebook takes {:.2f} minutes.'.format(notebook_running_time))