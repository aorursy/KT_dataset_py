# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
american=pd.read_csv("/kaggle/input/uber-pickups-in-new-york-city/other-American_B01362.csv")

american['DATE']=pd.to_datetime(american['DATE']+ ' ' + american['TIME'])

american.head()
american=american.drop(['Unnamed: 3','Unnamed: 4','Unnamed: 5'],axis=1)
american.Timestamp = pd.to_datetime(american['DATE'],format="%Y/%m/%d %H:%M")

american['Date_only'] = american.Timestamp.dt.date

american['Date'] = american.Timestamp

american['Month'] = american.Timestamp.dt.month

american['Week'] = american.Timestamp.dt.dayofweek

american['DayName'] = american.Timestamp.dt.weekday_name

american['Day'] = american.Timestamp.dt.day

american['Hour'] = american.Timestamp.dt.hour

american['Minutes']=american.Timestamp.dt.minute
american.head()
hourly_trip_data = american.groupby(['Day','Hour','Week','Month'])['DATE'].count()

hourly_trip_data = hourly_trip_data.reset_index()

hourly_trip_data = hourly_trip_data.rename(columns = {'DATE':'ride_count'})

hourly_trip_data.head()
#number of trips per hour 

trips_hour=hourly_trip_data.pivot_table(index=['Hour'],values='ride_count',aggfunc={'ride_count':np.sum},margins=True,margins_name='Total')

trips_hour1=hourly_trip_data.pivot_table(index=['Hour'],values='ride_count',aggfunc={'ride_count':np.sum})
#number of trips per hour 

trips_hour1.plot(kind='bar',figsize=(8,6))

plt.ylabel('Total Journeys')

plt.title('Journeys by Hour')
#the highest, minimum, and the average trips per day

trips_day=hourly_trip_data.pivot_table(index=['Day'],values='ride_count',aggfunc={'ride_count':np.sum})
#highest trips per day per cab service.

print("Highest trips per day:",max(trips_day.ride_count))

#minimum trips per day per cab service.

print("Minimum trips per day:",min(trips_day.ride_count))

#average trips per day per cab service.

print("Average trips per day in American:",np.mean(trips_day.ride_count))
trips_day.plot(kind='bar',figsize=(8,6))

plt.ylabel('Total Journeys')

plt.title('Trips per Month day ')
trips_day.plot(kind='bar',figsize=(8,6))

plt.ylabel('Total Journeys')

plt.title('Trips per Month day ')
# Monthwise distribution

american.groupby('Month')['Date'].count().plot(kind = 'bar')

plt.title('Ride Density by Month')

plt.ylabel("Number of Rides")
#Ride density by weekday

american['weekday'] = False

american.loc[american.Week>=5,'weekday'] = False

american.loc[american.Week<5,'weekday'] = True
american.groupby('weekday')['Date'].count().plot(kind = 'bar', figsize = (3,5));

plt.title('Ride Density by Weekend/Weekday');

plt.ylabel("Number of Rides")
#Ride density by weekend

weekday_names = ['0:Monday, 1:Tuesday, 2:Wednesday, 3:Thursday, 4:Friday, 5:Saturday, 6:Sunday']

print(weekday_names)

american.groupby('Week')['Date'].count().plot(kind = 'bar', figsize = (3,5))

plt.title('Ride Density by Individual Days in Week')

plt.ylabel("Number of Rides")