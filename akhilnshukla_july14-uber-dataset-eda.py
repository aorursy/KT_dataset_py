# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
#Read in the dataset of july uber pickups
uber_data = pd.read_csv("../input/uber-pickups-in-new-york-city/uber-raw-data-jul14.csv")
print(uber_data.head(10))
#Convert the type of the column to datetime
uber_data["Date/Time"] = pd.to_datetime(uber_data["Date/Time"])
#Using a floor can round off the date-time into discrete increments
print(uber_data["Date/Time"].dt.floor('1H').head(10))
#Let's get value counts to see the number of trips at given times
print(uber_data["Date/Time"].dt.floor('1H').value_counts().head(10))
#And sort the data so it is chronological
hourly_data = uber_data["Date/Time"].dt.floor('1H').value_counts()
hourly_data = hourly_data.sort_index()
print(hourly_data.head(10))
import matplotlib.pyplot as plt
#Plotting the data shows some trend components
hourly_data.plot(kind="line")
plt.show()
#We are going to see the average number of trips for each hour/week day combination
#First split the date into the week day, hour and the actual date
hours = uber_data["Date/Time"].dt.hour
week_day = uber_data["Date/Time"].dt.weekday
date = uber_data["Date/Time"].dt.date
weekly_data = pd.concat([week_day, hours, date], axis=1)
weekly_data.columns = ["Week Day", "Hour", "Date"]

print(weekly_data.head(10))
import calendar
#The calendar library can map the integer versions of calendar weekdays to the actual name
#0 -> Monday, 1 -> Tuesday, etc.
print(calendar.day_name[0])
#Map the name
weekly_data["Week Day"] = weekly_data["Week Day"].apply(lambda x: calendar.day_name[x])
print(weekly_data["Week Day"].head(10))
#By grouping by the date, week day, and hour we can aggregate the size (# of entries) on each date
weekly_data = weekly_data.groupby(["Date","Week Day", "Hour"]).size()
print(weekly_data.head(10))
#Reset the index
weekly_data = weekly_data.reset_index()
print(weekly_data.head(10))
#Rename 0, the default column name to be size
weekly_data = weekly_data.rename(columns={0: "Size"})
print(weekly_data.head(10))
#Now we can group by the week day and average to get the mean for each week day/hour
weekly_data = weekly_data.groupby(["Week Day", "Hour"]).mean()["Size"]
print(weekly_data.head(10))
#Unstack takes a level of the index and translates it to be a column
#We pick level=0 because we want the week day name to be the column
weekly_data = weekly_data.unstack(level=0)
print(weekly_data)
#Reindex allows you to re-arrange the columns however you would like
weekly_data = weekly_data.reindex(columns=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
print(weekly_data)
import seaborn as sns
#Plot a heatmap of the data
#Change the color map to blue (default is red)
sns.heatmap(weekly_data, cmap='Blues')
plt.show()
#geopy is a library which finds distance between latitude and longitude
import geopy.distance
#Check to make sure latitude and longitude are in the right order
metro_art_coordinates = (40.7794, -73.9632)
empire_state_building_coordinates = (40.7484, -73.9857)
distance = geopy.distance.distance(metro_art_coordinates, empire_state_building_coordinates)
print(distance)  # gives distance in km
print(distance.mi)  # in miles
#Easy way to convert our latitude and longitude columns to tuples
print(uber_data[["Lat", "Lon"]].apply(lambda x: tuple(x),axis=1))
#Using the geopy version may take too long, so we will use the haversine formula instead
from math import radians, cos, sin, asin, sqrt

def haversine(coordinates1, coordinates2):

    lon1 = coordinates1[1]
    lat1 = coordinates1[0]
    lon2 = coordinates2[1]
    lat2 = coordinates2[0]
    #Change to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    
    # Apply the harversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 3956
    return c * r
print(haversine(metro_art_coordinates, empire_state_building_coordinates))
#Now, we can find the distances to both attractions
uber_data["Distance MM"] = uber_data[["Lat", "Lon"]].apply(lambda x: haversine(metro_art_coordinates,tuple(x)),axis=1)
uber_data["Distance ESB"] = uber_data[["Lat", "Lon"]].apply(lambda x: haversine(empire_state_building_coordinates,tuple(x)),axis=1)
print(uber_data["Distance MM"].head(5))
print(uber_data["Distance ESB"].head(5))
#Summarize the data
print(uber_data[["Distance MM", "Distance ESB"]].describe())
#Using boolean indexing, we can sum to find the count within a specified range
print((uber_data[["Distance MM", "Distance ESB"]] < .25).sum())
import numpy as np
#Distance range takes a start, end (non-inclusive) and step amount
distance_range = np.arange(.1,5.1,.1)
print(distance_range)
#Run our analysis for each distance
distance_data = [(uber_data[["Distance MM", "Distance ESB"]] < dist).sum() for dist in distance_range]
print(distance_data)
#Concat
distance_data = pd.concat(distance_data, axis=1)
print(distance_data)
#Transpose and add in the index
distance_data = distance_data.transpose()
distance_data.index = distance_range
print(distance_data)
#And plot
distance_data.plot(kind="line")
plt.show()
#Folium can let us map geographical data, first get a base map with latitude and longitude
import folium as folium
uber_map = folium.Map(location=[40.7128, -74.0060], zoom_start=12)
uber_map
#Pick the first five latitude/longitude combinations
lat = uber_data["Lat"].values[:5]
lon = uber_data["Lon"].values[:5]

uber_map = folium.Map(location=[40.7128, -74.0060], zoom_start=12)
#Marker let's you drop markers on the map
#You can also add text to the markers with the popup argument
for i in range(len(lat)):
    folium.Marker((lat[i], lon[i]), popup="Rider {}".format(i+1)).add_to(uber_map)
uber_map
from folium.plugins import HeatMap

lat_lon = uber_data[["Lat", "Lon"]].values[:10000]
uber_map = folium.Map(location=[40.7128, -74.0060], zoom_start=12)
#A heatmap can be plotted like so... the radius argument controls the radius of each point within the map
#You can zoom in on this map to see more specific areas, or out to see more general
HeatMap(lat_lon, radius=13).add_to(uber_map)
uber_map
lat_lon = uber_data[["Lat", "Lon"]].values[:10000]
uber_map = folium.Map(location=[40.7128, -74.0060], zoom_start=10)
#A bigger radius (and more zoom) can let us observe drop offs outside of the city that happen often
#Such as the airport
HeatMap(lat_lon, radius=30).add_to(uber_map)
uber_map
#We can also give a weight to either give different values to points, or to make the graphs less dense looking
uber_data["Weight"] = .5
lat_lon = uber_data[["Lat", "Lon", "Weight"]].values[:10000]
uber_map = folium.Map(location=[40.7128, -74.0060], zoom_start=12)
#Now let's increase radius since the weights are less
HeatMap(lat_lon, radius=15).add_to(uber_map)
uber_map
#Let's get the points that are within distance of either point of interest
#There won't be overlap if we use only points that are .25 mile away
i = uber_data[["Distance MM", "Distance ESB"]] < .25
print(i)
#Take data where either one is true
i = i.any(axis=1)
print(i)
#This is our map data
map_data = uber_data[i].copy()
print(map_data)
#Let's draw on a heatmap with the locations within the radius
#Notice that one heatmap is a semi-circle because drop offs can't happen to the left of it
map_data["Weight"] = .1
lat_lon = map_data[["Lat", "Lon", "Weight"]].values
uber_map = folium.Map(location=[40.7728, -74.0060], zoom_start=13)
HeatMap(lat_lon, radius=10).add_to(uber_map)
uber_map
#Let's grab only the date and hour by replacing the other parts with 0
uber_data["Date_Hour"] = uber_data["Date/Time"].apply(lambda x: x.replace(microsecond=0,second=0,minute=0))
print(uber_data["Date_Hour"])
from datetime import datetime
#Take only the first week of data
map_data = uber_data[uber_data["Date/Time"] < datetime(2014,7,8)].copy()
map_data["Weight"] = .5
#Randomly sample 1/3 the values in each group
map_data = map_data.groupby('Date_Hour').apply(lambda x: x[["Lat", "Lon", "Weight"]].sample(int(len(x)/3)).values.tolist())
#Get the index
date_hour_index = [x.strftime("%m/%d/%Y, %H:%M:%S") for x in map_data.index]
#Get the data in list form (each element of this bigger list will be a list of lists with lat/lon/weight)
#Each element of the bigger list is a for a date/hour combo
date_hour_data = map_data.tolist()
from folium.plugins import HeatMapWithTime
uber_map = folium.Map(location=[40.7128, -74.0060], zoom_start=12)
#A heatmap with time can now be out together
hm = HeatMapWithTime(date_hour_data, index=date_hour_index)
hm.add_to(uber_map)
uber_map
#Recall the seasonality we saw before
hourly_data.plot(kind='line')
plt.show()
#What about the hourly trends?
h = hourly_data.groupby(hourly_data.index.hour).mean()
h.plot(kind="line")
plt.show()
#Something else of interest is the difference in hourly trends for weekdays and weekends
#We will index with i for weekdays
i = hourly_data.index.weekday <= 4

h_week = hourly_data.loc[i].groupby(hourly_data.loc[i].index.hour).mean()
h_weekend = hourly_data.loc[~i].groupby(hourly_data.loc[~i].index.hour).mean()
h = pd.concat([h_week, h_weekend], axis=1)
h.columns = ["Weekday", "Weekend"]
print(h)
#And plot to see the difference
h.plot(kind='line')
plt.show()
#We can also divide by the total number of trips for each to normalize and have each be a percent of total trips in a day
(h / h.sum()).plot(kind='line')
plt.show()
#We can finish our assessment of whether or not we see hourly effects by using a t-test to see if each hour
#has a statistically different proportion of rides for weekends vs. weekdays
from scipy.stats import ttest_ind

#The functions takes two samples and returns the t-stat and the p-value denoting the null hypothesis that they are the same
print(ttest_ind([100,105,110], [200,230,210]))
#Now for each day, let's normalize by the total number of rides in the day
hourly_data_pct = hourly_data.groupby(hourly_data.index.date).apply(lambda x: x / x.sum())
print(hourly_data_pct)
#For each group of hours, we will apply a function to test the null hypothesis where the first sample is
#weekdays and the second is weekends found by taking the days of the index
t_stats = hourly_data_pct.groupby(hourly_data_pct.index.hour).apply(lambda x: ttest_ind(x[x.index.weekday<=4], x[x.index.weekday>4])[0])
print(t_stats)
ax = t_stats.plot(kind='bar', color='blue')
ax.axhline(1.96, linestyle='--', color='grey', linewidth=2)
ax.axhline(0, color='black', linewidth=2)
ax.axhline(-1.96, linestyle='--', color='grey', linewidth=2)
plt.xlabel("Hour")
plt.ylabel("T-Statistic")
plt.title("Hourly Differences for Weekend vs. Weekday Uber Rides")
plt.show()