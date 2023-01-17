# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
uber_apr = pd.read_csv('../input/uber-raw-data-apr14.csv')

uber_apr.head()

uber_apr['Date/Time'] = pd.to_datetime(uber_apr['Date/Time'], format = '%m/%d/%Y %H:%M:%S')
uber_apr['Day'] = uber_apr['Date/Time'].dt.day

uber_apr['Weekday'] = uber_apr['Date/Time'].dt.weekday

uber_apr.head()
uber_apr['Hour'] = uber_apr['Date/Time'].dt.hour

uber_apr.head()

uber_apr_hour = uber_apr.pivot_table(index=['Hour'], values='Base', aggfunc='count')

uber_apr_hour.plot(kind='bar', figsize=(10,10))

plt.ylabel('trips')

plt.xlabel('Hour of the day')

uber_apr_day = uber_apr.pivot_table(index=['Day'], values='Base', aggfunc='count')

uber_apr_day.plot(kind='bar', figsize=(10,10))

plt.ylabel('trips')

plt.xlabel('Day of the month')

uber_apr['WeekdayName'] = uber_apr['Date/Time'].dt.weekday_name

uber_apr.head()
uber_apr_dayofweek = uber_apr.pivot_table(index=['WeekdayName'], values='Base', aggfunc='count')

uber_apr_dayofweek.plot(kind='bar', figsize=(10,10))

plt.ylabel('trips')

plt.xlabel('Day')
uber_apr_dayofweek
uber_apr_dayofweek.sort_index()

uber_apr_dayofweek
uber_apr_dayofweek.sort_index()
type(uber_apr_dayofweek)
uber_apr['Lat'].describe()

uber_apr['Lon'].describe()
from mpl_toolkits.basemap import Basemap

from matplotlib import cm

north, south, west, east = uber_apr['Lat'].max(), uber_apr['Lat'].min(), uber_apr['Lon'].min(), uber_apr['Lon'].max()

fig = plt.figure(figsize=(14,10))

ax = fig.add_subplot(111)

m = Basemap(projection='merc', llcrnrlat=south, urcrnrlat=north,

            llcrnrlon=west, urcrnrlon=east, lat_ts=south, resolution='i')

x, y = m(uber_apr['Lon'].values, uber_apr['Lat'].values)

m.hexbin(x, y, gridsize=500,

         bins='log', cmap=cm.YlOrRd_r)



north, south, west, east = uber_apr['Lat'].max(), uber_apr['Lat'].min(), uber_apr['Lon'].min(), uber_apr['Lon'].max()

print(north, south, west, east)
plt.scatter(range(uber_apr.shape[0]), np.sort(uber_apr['Lat'].values))

plt.show()