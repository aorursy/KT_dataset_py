# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

from matplotlib import cm

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

%matplotlib inline

DATA_FILE = '../input/uber-pickups-in-new-york-city/uber-raw-data-sep14.csv'

uber_data = pd.read_csv(DATA_FILE)

uber_data.head()
uber_data.info()
uber_data['Date/Time'] = pd.to_datetime(uber_data['Date/Time'], format="%m/%d/%Y %H:%M:%S")

uber_data['DayOfWeekNum'] = uber_data['Date/Time'].dt.dayofweek

uber_data['DayOfWeek'] = uber_data['Date/Time'].dt.weekday_name

uber_data['MonthDayNum'] = uber_data['Date/Time'].dt.day

uber_data['HourOfDay'] = uber_data['Date/Time'].dt.hour
#uber_data.head()

print(uber_data.count)
uber_weekdays = uber_data.pivot_table(index=['DayOfWeekNum','DayOfWeek'],

                                  values='Base',

                                  aggfunc='count')

uber_weekdays.plot(kind='bar', figsize=(8,6))

plt.ylabel('Total Journeys')

plt.title('Journeys by Week Day');
uber_monthdays = uber_data.pivot_table(index=['MonthDayNum'],

                                  values='Base',

                                  aggfunc='count')

uber_monthdays.plot(kind='bar', figsize=(8,6))

plt.ylabel('Total Journeys')

plt.title('Journeys by Month Day - August');
uber_hour = uber_data.pivot_table(index=['HourOfDay'],

                                  values='Base',

                                  aggfunc='count')

uber_hour.plot(kind='bar', figsize=(8,6))

plt.ylabel('Total Journeys')

plt.title('Journeys by Hour - August');
#plot the based using lat,lon coordinate

#isntall basemap-data-hires package;

#In the kernel:

#1) Setting (at the right bottom of the kernel): Turn the internet "ON"

#2) In the kernel:

#conda install basemap-data-hires



west, south, east, north = -74.26, 40.50, -73.70, 40.92



fig = plt.figure(figsize=(14,10))

ax = fig.add_subplot(111)

m = Basemap(projection='merc', llcrnrlat=south, urcrnrlat=north,

            llcrnrlon=west, urcrnrlon=east, lat_ts=south, resolution='i')

x, y = m(uber_data['Lon'].values, uber_data['Lat'].values)

m.hexbin(x, y, gridsize=1000,

         bins='log', cmap=cm.YlOrRd_r);
li = []

for dirname, _, filenames in os.walk('/kaggle/input/uber-pickups-in-new-york-city'):

    for filename in filenames:

        if filename.startswith("uber-raw-data-") and not filename.endswith('-15.csv'):

            file = os.path.join(dirname, filename)

            df = pd.read_csv(file, index_col=None, header=0)

            li.append(df)

dataframe_14 = pd.concat(li, axis=0, ignore_index=True)

dataframe_14.head()

dataframe_14.info()
print(dataframe_14.count)
dataframe_14['Date/Time'] = pd.to_datetime(dataframe_14['Date/Time'], format="%m/%d/%Y %H:%M:%S")

dataframe_14['Month'] = dataframe_14['Date/Time'].dt.month

dataframe_14['DayOfWeekNum'] = dataframe_14['Date/Time'].dt.dayofweek

dataframe_14['DayOfWeek'] = dataframe_14['Date/Time'].dt.weekday_name

dataframe_14['MonthDayNum'] = dataframe_14['Date/Time'].dt.day

dataframe_14['HourOfDay'] = dataframe_14['Date/Time'].dt.hour
dataframe_14.head()
uber_weekdays_14 = dataframe_14.pivot_table(index=['DayOfWeekNum','DayOfWeek'],

                                  values='Base',

                                  aggfunc='count')

uber_weekdays_14.plot(kind='bar', figsize=(8,6))

plt.ylabel('Total Journeys')

plt.title('Journeys by Week Day 2014');
uber_hour_14 = dataframe_14.pivot_table(index=['HourOfDay'],

                                  values='Base',

                                  aggfunc='count')

uber_hour_14.plot(kind='bar', figsize=(8,6))

plt.ylabel('Total Journeys')

plt.title('Journeys by Hour - 2014');
uber_month_14 = dataframe_14.pivot_table(index=['Month'],

                                  values='Base',

                                  aggfunc='count')

uber_month_14.plot(kind='bar', figsize=(8,6))

plt.ylabel('Total Journeys')

plt.title('Journeys by Month - 2014')
#dataframe_14['Weekday'] = np.where(dataframe_14.DayOfWeekNum == 5,'0','1')



dataframe_14['Weekday'] = dataframe_14.apply(lambda x: 0 if x[5] == 5 or x[5] == 6  else 1, axis=1)

dataframe_14.head()

print(dataframe_14[dataframe_14['Weekday']==0])
#plot journey for weekday

uber_hour_14 = dataframe_14[dataframe_14['Weekday']==1].pivot_table(index=['HourOfDay'],

                                  values='Base',

                                  aggfunc='count')

uber_hour_14.plot(kind='bar', figsize=(8,6))

plt.ylabel('Total Journeys')

plt.title('Journeys by Hour - 2014');
west, south, east, north = -74.26, 40.50, -73.70, 40.92

data=dataframe_14[dataframe_14['Weekday']==1]

fig = plt.figure(figsize=(14,10))

ax = fig.add_subplot(111)

m = Basemap(projection='merc', llcrnrlat=south, urcrnrlat=north,

            llcrnrlon=west, urcrnrlon=east, lat_ts=south, resolution='i')

x, y = m( data['Lon'].values, data['Lat'].values)

m.hexbin(x, y, gridsize=1000,

         bins='log', cmap=cm.YlOrRd_r);
#plot journey for weekend

uber_hour_14 = dataframe_14[dataframe_14['Weekday']==0].pivot_table(index=['HourOfDay'],

                                  values='Base',

                                  aggfunc='count')

uber_hour_14.plot(kind='bar', figsize=(8,6))

plt.ylabel('Total Journeys')

plt.title('Journeys by Hour - 2014');
west, south, east, north = -74.26, 40.50, -73.70, 40.92

data=dataframe_14[dataframe_14['Weekday']==0]

fig = plt.figure(figsize=(14,10))

ax = fig.add_subplot(111)

m = Basemap(projection='merc', llcrnrlat=south, urcrnrlat=north,

            llcrnrlon=west, urcrnrlon=east, lat_ts=south, resolution='i')

x, y = m( data['Lon'].values, data['Lat'].values)

m.hexbin(x, y, gridsize=1000,

         bins='log', cmap=cm.YlOrRd_r);
DATA_FILE_TEST = '../input/uber-pickups-in-new-york-city/other-Skyline_B00111.csv'

OTHER_data = pd.read_csv(DATA_FILE_TEST)

OTHER_data.head()