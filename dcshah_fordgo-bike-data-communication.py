import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os
import glob
import warnings 
%matplotlib inline
#### Reading all files together, ran only once

folder_name = '../input/bay-wheels-2019-data/'
files = glob.glob(os.path.join(folder_name, "*.csv"))
df = pd.concat((pd.read_csv(f) for f in files), ignore_index = True)
print(df.shape)
df.sample(5)
#Saving the master data (File Uploaded)

df.to_csv('fordgo_2019_master.csv', index = False)
ford2019 = pd.read_csv('fordgo_2019_master.csv')
ford2019.sample(10)
ford2019.info()
ford2019.isnull().sum()
ford2019.duplicated().sum()
ford2019.bike_share_for_all_trip.value_counts()
ford2019.user_type.value_counts()
#Copy of the original dataframe
copy_ford = ford2019.copy()
#Drop the rental access method column as 95% column is empty
copy_ford.drop('rental_access_method',axis=1,inplace=True)
#Convert start time and end time to datetime datatype
copy_ford.start_time = pd.to_datetime(copy_ford.start_time)
copy_ford.end_time = pd.to_datetime(copy_ford.end_time)

#Converting the station_id's and bike_id to string 
copy_ford.start_station_id = copy_ford.start_station_id.astype('str')
copy_ford.end_station_id = copy_ford.end_station_id.astype('str')
copy_ford.bike_id = copy_ford.bike_id.astype('str')

#Convert usertype and bike share to category
copy_ford.user_type = copy_ford.user_type.astype('category')
copy_ford.bike_share_for_all_trip = copy_ford.bike_share_for_all_trip.astype('category') 

copy_ford.info()
copy_ford['duration_min'] = copy_ford['duration_sec']/60
copy_ford['duration_min'] = copy_ford['duration_min'].astype(int)
#Getting the start and end date
copy_ford['start_date'] = copy_ford['start_time'].dt.strftime('%Y-%m-%d')
copy_ford['end_date'] = copy_ford['end_time'].dt.strftime('%Y-%m-%d')

#Getting the start and end hour
copy_ford['start_hour'] = copy_ford['start_time'].dt.strftime('%H')
copy_ford['end_hour'] = copy_ford['end_time'].dt.strftime('%H')

#Getting the start and end day
copy_ford['start_day'] = copy_ford['start_time'].dt.strftime('%a')
copy_ford['end_day'] = copy_ford['end_time'].dt.strftime('%a')

#Getting the start and end month
copy_ford['start_month'] = copy_ford['start_time'].dt.strftime('%B')
copy_ford['end_month'] = copy_ford['end_time'].dt.strftime('%B')
copy_ford.head()
#Define function to calculate distance travelled using Haversine Formula
from math import radians, cos, sin, asin, sqrt 
def distance(lat_1, lat_2, lon_1, lon_2): 
      
    # radians which converts from degrees to radians. 
    lon_1 = radians(lon_1) 
    lon_2 = radians(lon_2) 
    lat_1 = radians(lat_1) 
    lat_2 = radians(lat_2) 
       
    # Haversine formula  
    dlon = lon_2 - lon_1  
    dlat = lat_2 - lat_1 
    a = sin(dlat / 2)**2 + cos(lat_1) * cos(lat_2) * sin(dlon / 2)**2
  
    c = 2 * asin(sqrt(a))  
     
    # Radius of earth in kilometers. Use 3956 for miles 
    r = 6371
    dist = c * r
       
    # calculate the result 
    return(dist) 
#Finding the distance travelled
copy_ford['distance_km'] = copy_ford.apply(lambda x: distance(x['start_station_latitude'], x['end_station_latitude'],
                                                              x['start_station_longitude'],x['end_station_longitude']), axis=1)
#KM to Miles
copy_ford['distance_miles'] = copy_ford['distance_km'] * 0.621371
copy_ford.sample(6)
copy_ford.info(null_counts=True)
copy_ford.to_csv('fordgo_2019_master_clean.csv',index=False)
#Basic Color Palette
basic_color = sb.color_palette()[0]
#Trips every hour
plt.figure(figsize=(7,5))
sb.set_style('darkgrid')
sb.countplot(data=copy_ford,x='start_hour',color=basic_color)
plt.ylabel('#Trips')
plt.xlabel('Start Hour of Day');
plt.title('Trips every hour of the day');
#Defining the order of Days
day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
day_cat = pd.api.types.CategoricalDtype(ordered=True, categories=day_order)
copy_ford['start_day'] = copy_ford['start_day'].astype(day_cat)
#Trips every day
plt.figure(figsize=(8,6))
sb.countplot(data=copy_ford,x='start_day',color=basic_color)
plt.ylabel('#Trips');
plt.xlabel('Day of Week');
plt.title('Trips every day of the week')
#Defining the order of Months
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
month_cat = pd.api.types.CategoricalDtype(ordered=True, categories=month_order)
copy_ford['start_month'] = copy_ford['start_month'].astype(month_cat)
#Trips every month
plt.figure(figsize=(8,6))
sb.countplot(data=copy_ford,x='start_month',color=basic_color)
plt.xlabel('Month of Year');
plt.ylabel('#Trips');
plt.title('Trips every Month')
plt.xticks(rotation=15);
copy_ford.duration_sec.describe()
bins = np.arange(0, 3600, 60)

plt.figure(figsize=(8,6))
plt.hist(data = copy_ford, x = 'duration_sec',bins=bins);
plt.title("Trip Duration in Seconds", y=1.05)
plt.ylabel('Number of Bike Trips')
plt.xlabel('Duration (Sec)');
copy_ford.duration_min.describe()
bins = np.arange(0, 50, 1)
ticks = [0, 5, 10, 15, 20, 25, 30,  35, 40, 45,50]
labels = ['{}'.format(v) for v in ticks]

plt.figure(figsize=(8,6))
plt.hist(data = copy_ford, x = 'duration_min',bins= bins);

plt.title("Trip Duration in Minutes", y=1.05)
plt.xlabel('Duration (Min)')
plt.ylabel('Number of Bike Trips')
plt.xticks(ticks, labels);
copy_ford.distance_miles.describe()
bins = np.arange(0, 5, 1.34)
plt.figure(figsize=(8,6))
plt.hist(data = copy_ford, x = 'distance_miles',bins=bins)

plt.title("Trips Distance in Miles")
plt.xlabel('Distance (Miles)')
plt.ylabel('Number of Bike Trips');
plt.figure(figsize=(8,6))
sb.countplot(data=copy_ford, x='user_type', color=basic_color);
plt.xlabel('User Type');
plt.ylabel('Count')
plt.title("Different Types of User");
plt.figure(figsize=(8,6))
sb.countplot(data=copy_ford, x='bike_share_for_all_trip', color=basic_color);
plt.xlabel('Bike Share for All Trip');
plt.ylabel('Count')
plt.title("Bike Share for all trips activation");
plt.figure(figsize = [12, 10])
sb.heatmap(copy_ford.corr(), annot = True, fmt = '.3f', cmap = 'vlag_r', center = 0);
user_type_count = copy_ford.groupby(["start_day", "user_type"]).size().reset_index()

plt.figure(figsize=(12,10))
axis = sb.pointplot(x='start_day', y=0, hue='user_type', scale=.7, data=user_type_count)
plt.title('The weekly bike rides per user type', fontsize=16, y=1.015)
plt.xlabel('Day')
plt.ylabel('Count')
leg = axis.legend()
axis = plt.gca()
plt = copy_ford.groupby('user_type')['distance_miles'].mean().plot(kind='barh', figsize=(10,8));

plt.set_title('Average Distance (Miles) by User Type')
plt.set_xlabel('Average Distance Traveled (Miles)')
plt.set_ylabel('User Type');
data_sub = copy_ford.query('duration_min < 20')

g = sb.catplot(data=data_sub, y='duration_min', col="user_type", kind='box', color = basic_color)

g.set_titles(col_template = '{col_name}')
g.set_axis_labels("", "Duration (Min)")
g.fig.suptitle('Duration (Mins) by User Type');
sb.barplot(data=copy_ford, x='start_day', y='duration_min', color=basic_color);
plt.xlabel('Day of Week');
plt.ylabel('Avg. Trip Duration in Minute')
plt.title('Duration on the days of week');
sb.pointplot(data=copy_ford, x='start_day', y='duration_min', hue='user_type', dodge=0.3,);

plt.xlabel('Day of Week');
plt.ylabel('Avg. Trip Duration in Minute')
plt.legend(title='User Type');
plt.figure(figsize=(9,8))
plt.suptitle('Hourly trips day wise for customers and subscribers', fontsize=14, fontweight='semibold')

# heatmap for customers
plt.subplot(1, 2, 1)
customer = copy_ford.query('user_type == "Customer"').groupby(["start_day","start_hour"])["bike_id"].size().reset_index()
customer = customer.pivot("start_hour", "start_day", "bike_id")
sb.heatmap(customer,cmap='RdPu')

plt.title("Customer", y=1.015)
plt.xlabel('Day')
plt.ylabel('Start Hour')

# heatmap for subscribers
plt.subplot(1, 2, 2)
sub = copy_ford.query('user_type == "Subscriber"').groupby(["start_day","start_hour"])["bike_id"].size().reset_index()
sub = sub.pivot("start_hour", "start_day", "bike_id")
sb.heatmap(sub,cmap='RdPu')

plt.title("Subscriber", y=1.015)
plt.xlabel('Day')
plt.ylabel('');