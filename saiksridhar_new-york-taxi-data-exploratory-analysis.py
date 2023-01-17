#Set-up

#Import relevant packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# import geopandas as gpd

import seaborn as sns

# import contextily as ctx

import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster

import math

%matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
trip_data_base = pd.read_csv(r"/kaggle/input/trip_data_1.csv",

                            index_col = ['medallion'], parse_dates = ['pickup_datetime', 'dropoff_datetime'])
print('Trip Base Data:')

trip_data_base.head()
#Check for null data points and exclude these

print(trip_data_base.isnull().sum())
print('Data types:')

print(trip_data_base.dtypes)
print(trip_data_base.describe())
#Drop rows with data issues

trip_data_base = trip_data_base[trip_data_base['passenger_count'] > 0]

trip_data_base = trip_data_base[trip_data_base['passenger_count'] < 10]
#Open a single file within the trip fares folder

#Adjust folder name to local drive if required

trip_fare_base = pd.read_csv(r"/kaggle/input/trip_fare_1.csv",

                            index_col = ['medallion'], parse_dates = [' pickup_datetime'])
print('Trip Fare Data:')

trip_fare_base.head()
print('Data types:')

print(trip_fare_base.dtypes)
#Check for null data points and exclude these

print(trip_fare_base.isnull().sum())
#Perform data exploration

#Max datetime in trip data

print('Trip data summary:')

#Initial assessment of trip data file

print(trip_data_base.describe())



#Implement adjustment for fare data column names

print()

print('Trip fare data initial column names:')

print(trip_fare_base.columns)

#Column names have spaces in them

#This was deduced through an error when selecting from a column

#Reproduce dataframe ommitting spaces in the attribute names

trip_fare_base = trip_fare_base.rename(columns = {' hack_license': 'hack_license', ' vendor_id': 'vendor_id', ' pickup_datetime': 'pickup_datetime', 

                                               ' payment_type': 'payment_type',' fare_amount': 'fare_amount', ' surcharge': 'surcharge', 

                                               ' mta_tax': 'mta_tax', ' tip_amount': 'tip_amount', ' tolls_amount': 'tolls_amount', 

                                               ' total_amount': 'total_amount'})



print()

print('Trip fare date adjusted column names:')

print(trip_fare_base.columns)

print()

print('Fare data summary:')

#Initial assessment of fare data file

print(trip_fare_base.describe())
#Assess revenue by day over the month of January to produce a simple line chart

fare_daily_avg = trip_fare_base.groupby(trip_fare_base.pickup_datetime.dt.day).agg({'total_amount': np.mean})

#Plot the line chart

sns.set_style("darkgrid")

plt.figure(figsize = (15,8))

plt.title("Average Trip Revenue per Day")

sns.lineplot(data = fare_daily_avg)

plt.xlabel("Day")

plt.ylabel("Revenue")
fare_dayofweek_avg = trip_fare_base.groupby(trip_fare_base.pickup_datetime.dt.dayofweek).agg({'total_amount': np.mean})



sns.set_style("darkgrid")

plt.figure(figsize = (15,8))

plt.title("Average Trip Revenue per Day of Week")

sns.lineplot(data = fare_dayofweek_avg)

plt.xlabel("Day of Week")

plt.ylabel("Revenue")
trips_per_day = trip_data_base.groupby(trip_data_base.pickup_datetime.dt.day).size()

#Plot the line chart

sns.set_style("darkgrid")

plt.figure(figsize = (15,8))

plt.title("Number of Trips per Day")

sns.lineplot(data = trips_per_day)

plt.xlabel("Day")

plt.ylabel("Number of Trips")
average_trips_dayofweek = trip_data_base.groupby(trip_data_base.pickup_datetime.dt.dayofweek).size()

#Plot the line chart

sns.set_style("darkgrid")

plt.figure(figsize = (15,8))

plt.title("Total Trips by day of Week")

sns.lineplot(data = average_trips_dayofweek)

plt.xlabel("Day")

plt.ylabel("Number of Trips")
print(average_trips_dayofweek)
passenger_daily_count = trip_data_base.groupby(trip_data_base.pickup_datetime.dt.day).agg({'passenger_count': np.mean})

#Plot the line chart

sns.set_style("darkgrid")

plt.figure(figsize = (15,8))

plt.title("Daily Average Number of Passengers")

sns.lineplot(data = passenger_daily_count)

plt.xlabel("Day")

plt.ylabel("Average Number of Passengers")
passenger_dayofweek_avg = trip_data_base.groupby(trip_data_base.pickup_datetime.dt.dayofweek).agg({'passenger_count': np.mean})

#Plot the line chart

sns.set_style("darkgrid")

plt.figure(figsize = (15,8))

plt.title("Average Number of Passengers per Day of Week")

sns.lineplot(data = passenger_dayofweek_avg)

plt.xlabel("Day of Week")

plt.ylabel("Average Number of Passengers")
#One month data file is too large to process on Kaggle session so data over a single day of January will be observed (21st Jan)



trip_data_single_day = trip_data_base[trip_data_base['pickup_datetime'].dt.day == 21]

#trip_data_last_full_week  = trip_data_last_full_week_prep[trip_data_last_full_week_prep['pickup_datetime'].dt.day <= 26]



# # #Create geo spatial data file

# pick_up_hotspots = gpd.GeoDataFrame(trip_data_last_full_week, geometry=gpd.points_from_xy(trip_data_last_full_week.pickup_longitude,

#                                                                                           trip_data_last_full_week.pickup_latitude))

# # # Set the coordinate reference system (CRS) to EPSG 4326

# pick_up_hotspots.crs = {'init': 'epsg:3857'}

# pick_up_hotspots.head()
#Show map of New York with coordinates (40.7141667,-74.0063889) as per https://www.travelmath.com/cities/New+York,+NY

# Create a base map

pickup_heatmmap = folium.Map(location=[40.7141667,-74.0063889], tiles='cartodbpositron', zoom_start=12)



# Add a heatmap to the base map

HeatMap(data = trip_data_single_day[['pickup_latitude', 'pickup_longitude']], radius=10).add_to(pickup_heatmmap)



# Display the map

pickup_heatmmap
sns.set_style("darkgrid")

plt.figure(figsize = (15,8))

plt.title("Distribution of Number of Passengers per Trip")

sns.distplot(a=trip_data_base['passenger_count'], kde=False)

plt.xlabel("Number of Passengers")

plt.ylabel("Frequency")
#Create payment type summary dataset

payment_type_summary = trip_fare_base.groupby(trip_fare_base.payment_type).size()

payment_type_summary = payment_type_summary.rename('volume').reset_index()

#print(payment_type_summary)



sns.set_style("darkgrid")

plt.figure(figsize = (15,8))

plt.title("Distribution of Payment Type per Trip")

sns.barplot(x = payment_type_summary['payment_type'], y = payment_type_summary['volume'])

plt.xlabel("Payment Type")

plt.ylabel("Frequency")
#Create required tables for analysis

fare_data_single_day = trip_fare_base[trip_fare_base['pickup_datetime'].dt.day == 21]



#Passenger Data

passengers = trip_data_single_day.loc[:, ['pickup_datetime', 'passenger_count']]

fares = fare_data_single_day.loc[:, ['pickup_datetime', 'fare_amount']]



passengers_fares = pd.merge(passengers, fares, on = ['pickup_datetime'], how='left')

passengers_fares.head()

#sns.jointplot(x = passengers_fares['passenger_count'], y = passengers_fares['fare_amount'], kind="kde")
sns.set_style("darkgrid")

plt.figure(figsize = (15,8))

plt.title("Distribution of Passenger Count vs Fares")

sns.jointplot(x = passengers_fares['passenger_count'], y = passengers_fares['fare_amount'], kind="kde", marginal_kws=dict(bins=10, rug=True))

plt.xlabel("Passenger Count")

plt.ylabel("Fare Amount")
#Create table with working hours per driver per day

driver_hours = trip_data_base.groupby([trip_data_base.index, trip_data_base.hack_license, trip_data_base.pickup_datetime.dt.date]).agg(start_time = ('pickup_datetime', min), 

                                                                                                                  end_time = ('pickup_datetime', max))

driver_hours.head()

driver_hours['number_hours_worked'] = driver_hours['end_time'] - driver_hours['start_time']

driver_hours.head()



#Create table with vendor revenue per day

driver_revenue = trip_fare_base.groupby([trip_fare_base.index, trip_fare_base.hack_license, trip_fare_base.pickup_datetime.dt.date]).agg(total_fare_revenue = ('fare_amount', sum))

driver_revenue.head()



driver_daily_hours_revenue = pd.merge(driver_hours, driver_revenue,  how='left', left_on=[driver_hours.index], 

                                      right_on = [driver_revenue.index], left_index = True)



driver_daily_hours_revenue.reset_index(['pickup_datetime'])

driver_daily_hours_revenue_dayextract = driver_daily_hours_revenue[driver_daily_hours_revenue['start_time'].dt.day == 21]

driver_daily_hours_revenue_dayextract.head()
sns.set_style("darkgrid")

plt.figure(figsize = (15,8))

plt.title("Driver Hours vs Total Fares")

sns.jointplot(x = driver_daily_hours_revenue_dayextract['number_hours_worked'], y = driver_daily_hours_revenue_dayextract['total_fare_revenue'], kind="kde")

plt.xlabel("Driver Hours")

plt.ylabel("Total Fare Revenue")