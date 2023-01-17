import pandas as pd

import numpy as np



#Visualization modules

import matplotlib.pyplot as plt

import seaborn as sns



#The matplotlib basemap toolkit is a library for plotting 2D data on maps in Python

from mpl_toolkits.basemap import Basemap

from matplotlib import cm #Colormap



#Animation Modules

from matplotlib.animation import FuncAnimation

import matplotlib.animation as animation



%matplotlib inline
#Load the datasets



df_apr14=pd.read_csv("/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-apr14.csv")

df_may14=pd.read_csv("/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-may14.csv")

df_jun14=pd.read_csv("/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-jun14.csv")

df_jul14=pd.read_csv("/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-jul14.csv")

df_aug14=pd.read_csv("/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-aug14.csv")

df_sep14=pd.read_csv("/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-sep14.csv")



#Merge the dataframes into one



df = df_apr14.append([df_may14,df_jun14,df_jul14,df_aug14,df_sep14], ignore_index=True)
df.head()
df.info()
#Renaming the Date/Time Colomn

df = df.rename(columns={'Date/Time': 'Date_time'})



#Converting the Date_time type into Datetime

df['Date_time'] = pd.to_datetime(df['Date_time'])



#Adding usufull colomns

df['Month'] = df['Date_time'].dt.month_name()

df['Weekday'] = df['Date_time'].dt.day_name()

df['Day'] = df['Date_time'].dt.day

df['Hour'] = df['Date_time'].dt.hour

df['Minute'] = df['Date_time'].dt.minute
df.head()
df.info()
df.describe(include = 'all')
#Defining a function that counts the number of rows

def count_rows(rows):

    return len(rows)
#Creating the hour and day dataframe

df_hour_day = df.groupby('Hour Day'.split()).apply(count_rows).unstack()

df_hour_day.head()
plt.figure(figsize = (12,8))



#Using the seaborn heatmap function 

ax = sns.heatmap(df_hour_day, cmap=cm.YlGnBu, linewidth = .5)

ax.set(title="Trips by Hour and Day");
df_hour_weekday = df.groupby('Hour Weekday'.split(), sort = False).apply(count_rows).unstack()

df_hour_weekday.head()
plt.figure(figsize = (12,8))



ax = sns.heatmap(df_hour_weekday, cmap=cm.YlGnBu, linewidth = .5)

ax.set(title="Trips by Hour and Weekday");
df_day_month = df.groupby('Day Month'.split(), sort = False).apply(count_rows).unstack()

df_day_month.head()
plt.figure(figsize = (12,8))



ax = sns.heatmap(df_day_month, cmap = cm.YlGnBu, linewidth = .5)

ax.set(title="Trips by Day and Month");
#The number of trips the 30th of April

max_april = max(df_day_month['April'])



#The mean number of trips the rest of April

mean_rest_april = df_day_month['April'][0:29].sum() / 29



ratio_april = round(max_april / mean_rest_april)

print('The number of trips the 30th of April is {} times higher than the mean number of trips during the rest of the month'.format(ratio_april))
df_month_weekday = df.groupby('Month Weekday'.split(), sort = False).apply(count_rows).unstack()

df_month_weekday.head()
plt.figure(figsize = (12,8))



ax = sns.heatmap(df_month_weekday, cmap= cm.YlGnBu, linewidth = .5)

ax.set(title="Trips by Month and Weekday");
#Setting up the limits

top, bottom, left, right = 41, 40.55, -74.3, -73.6



#Extracting the Longitude and Latitude of each pickup in our dataset

Longitudes = df['Lon'].values

Latitudes  = df['Lat'].values
df_reduced = df.drop_duplicates(['Lat','Lon'])
ratio_reduction = round((count_rows(df) - count_rows(df_reduced))/count_rows(df) * 100)

print('The dataset has been reduced by {}%'.format(ratio_reduction))
#Extracting the Longitude and Latitude of each pickup in our reduced dataset

Longitudes_reduced = df_reduced['Lon']

Latitudes_reduced  = df_reduced['Lat']
%matplotlib inline



plt.figure(figsize=(16, 12))



plt.plot(Longitudes_reduced, Latitudes_reduced, '.', ms=.8, alpha=.5)



plt.ylim(top=top, bottom=bottom)

plt.xlim(left=left, right=right)





plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.title('New York Uber Pickups from April to September 2014')



plt.show()
plt.figure(figsize=(18, 14))

plt.title('New York Uber Pickups from April to September 2014')



#https://matplotlib.org/basemap/api/basemap_api.html

map = Basemap(projection='merc', urcrnrlat=top, llcrnrlat=bottom, llcrnrlon=left, urcrnrlon=right)

x, y = map(Longitudes, Latitudes)

map.hexbin(x, y, gridsize=1000, bins='log', cmap=cm.inferno)

map.colorbar(location='right', format='%.1f', label='Number of Pickups');