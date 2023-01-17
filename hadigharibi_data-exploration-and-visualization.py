import pandas as pd

#import seabon as sns

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

from matplotlib import cm
%matplotlib inline
DATA_FILE = '../input/uber-raw-data-aug14.csv'

uber_data = pd.read_csv(DATA_FILE)

uber_data.head()
uber_data.info()
uber_data['Date/Time'] = pd.to_datetime(uber_data['Date/Time'], format="%m/%d/%Y %H:%M:%S")

uber_data['DayOfWeekNum'] = uber_data['Date/Time'].dt.dayofweek

uber_data['DayOfWeek'] = uber_data['Date/Time'].dt.weekday_name

uber_data['MonthDayNum'] = uber_data['Date/Time'].dt.day

uber_data['HourOfDay'] = uber_data['Date/Time'].dt.hour
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

plt.title('Journeys by Month Day');
uber_hour = uber_data.pivot_table(index=['HourOfDay'],

                                  values='Base',

                                  aggfunc='count')

uber_hour.plot(kind='bar', figsize=(8,6))

plt.ylabel('Total Journeys')

plt.title('Journeys by Hour');
west, south, east, north = -74.26, 40.50, -73.70, 40.92



fig = plt.figure(figsize=(14,10))

ax = fig.add_subplot(111)

m = Basemap(projection='merc', llcrnrlat=south, urcrnrlat=north,

            llcrnrlon=west, urcrnrlon=east, lat_ts=south, resolution='i')

x, y = m(uber_data['Lon'].values, uber_data['Lat'].values)

m.hexbin(x, y, gridsize=1000,

         bins='log', cmap=cm.YlOrRd_r);