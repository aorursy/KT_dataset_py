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
df1 = pd.read_csv('/kaggle/input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q1 2018.csv')
df2 = pd.read_csv('/kaggle/input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q2 2018.csv')
df3 = pd.read_csv('/kaggle/input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q3 2018.csv')
df4 = pd.read_csv('/kaggle/input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q4 2018.csv')
df2018 = df1.copy()
df2018 = df2018.append([df2, df3, df4], sort=False)
del df1
del df2
del df3
del df4
df2018.head(10)
# prepare the dataset
df2018['time'] = pd.to_datetime(df2018.trip_start_time)
df2018.index = df2018.time.dt.date
df2018.index.name = 'index'

# cleaning the dataset
df2018 = df2018.dropna(subset=['from_station_name','to_station_name'])
df = df2018.copy()
station = list(set(list(df.from_station_name.values) + list(df.to_station_name.values)))
date = df.index.unique().values
route = df.copy()
route = route[['from_station_name', 'to_station_name']]
twoway = route.copy()
twoway = twoway[twoway.from_station_name == twoway.to_station_name]
oneway = route.copy()
oneway = oneway[oneway.from_station_name != oneway.to_station_name]
twoway_map = twoway.groupby('from_station_name').count().sort_values(by='to_station_name', ascending=False)
print('10 Stations with the highest number of two-way traveller')
twoway_map[:10]
# mapping the number of outgoing bike from each station each day in 2018
outmap = pd.get_dummies(route.from_station_name).groupby('index').sum()
# mapping the number of incoming bike to each station each day in 2018
inmap = pd.get_dummies(route.to_station_name).groupby('index').sum()
outmap.head(5) # number of bikes leaves the station
inmap.head(5) # number of bikes entering the station
print('number of station with enough bike to use next morning, aka number of bikes entering > number of bikes leaving the station')
((inmap - outmap)>=0).sum(axis=1)
print('number of station with less bike to use next morning, or need a crew to return bikes back to station before next morning')
((inmap - outmap)<0).sum(axis=1)
print('Station and the total number of days in 2018 where stations need more bikes to be returned by the crew every night')
((inmap - outmap)<0).sum(axis=0).sort_values(ascending=False)[:20]
bike_minus = inmap - outmap # incoming bikes minus leaving bikes
bike_minus = np.absolute(bike_minus[bike_minus < 0]) # show only minus value
bike_minus.head(10) # number of bikes that required by crew to be returned to each station
print('20 Stations with the highest number of required returned bikes in a day')
np.max(bike_minus, axis=0).sort_values(ascending=False)[:20]
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#import chart_studio.plotly as py

import plotly.express as px
import plotly.graph_objects as go

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
bike_share = df2018
bike_share['trip_start_time'] = pd.to_datetime(bike_share['trip_start_time'])
bike_share['trip_stop_time'] = pd.to_datetime(bike_share['trip_stop_time'])
bike_share['hour_start'] = bike_share['trip_start_time'].apply(lambda time: time.hour)
bike_share['month_start'] = bike_share['trip_start_time'].apply(lambda time: time.month)
bike_share['weekday_start'] = bike_share['trip_start_time'].apply(lambda time: time.dayofweek)
bike_share['hour_stop'] = bike_share['trip_stop_time'].apply(lambda time: time.hour)
bike_share['month_stop'] = bike_share['trip_stop_time'].apply(lambda time: time.month)
bike_share['weekday_stop'] = bike_share['trip_stop_time'].apply(lambda time: time.dayofweek)
mon = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
bike_share['month_start'] = bike_share['month_start'].map(mon)
bike_share['month_stop'] = bike_share['month_stop'].map(mon)
day = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
bike_share['weekday_start'] = bike_share['weekday_start'].map(day)
bike_share['weekday_stop'] = bike_share['weekday_stop'].map(day)
bike_share.head()
plt.figure(figsize=(10,5))
sns.set_style('darkgrid')
sns.countplot(x='user_type',data=bike_share,palette='viridis')
plt.title('Bike Share Toronto Membership 2018')
plt.figure(figsize=(10,5))
sns.set_style('darkgrid')
ridership = sns.countplot(data=bike_share, x='month_start', hue='user_type', palette='coolwarm')
plt.title('Bike Share Toronto Ridership 2018')
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.figure(figsize=(10,5))
sns.set_style('darkgrid')
ridership = sns.countplot(data=bike_share, x='weekday_start', hue='user_type', palette='coolwarm')
plt.title('Bike Share Toronto Ridership 2018')
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.figure(figsize=(10,5))
sns.set_style('darkgrid')
ridership = sns.countplot(data=bike_share, x='hour_start', hue='user_type', palette='coolwarm')
plt.title('Bike Share Toronto Ridership 2018')
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.figure(figsize=(10,5))
sns.set_style('darkgrid')
sns.countplot(y=bike_share[bike_share['user_type']=='Annual Member']['from_station_name'],data=bike_share, palette='coolwarm',order=bike_share[bike_share['user_type']=='Annual Member']['from_station_name'].value_counts().index[:5])
plt.title('Top 5 Departing Stations (Annual Members)')
plt.figure(figsize=(10,5))
sns.set_style('darkgrid')
sns.countplot(y=bike_share[bike_share['user_type']=='Annual Member']['to_station_name'],data=bike_share, palette='coolwarm',order=bike_share[bike_share['user_type']=='Annual Member']['to_station_name'].value_counts().index[:5])
plt.title('Top 5 Arriving Stations (Annual Members)')
plt.figure(figsize=(10,5))
sns.set_style('darkgrid')
sns.countplot(y=bike_share[bike_share['user_type']=='Casual Member']['from_station_name'],data=bike_share, palette='viridis',order=bike_share[bike_share['user_type']=='Casual Member']['from_station_name'].value_counts().index[:5])
plt.title('Top 5 Departing Stations (Casual Members)')
plt.figure(figsize=(10,5))
sns.set_style('darkgrid')
sns.countplot(y=bike_share[bike_share['user_type']=='Casual Member']['to_station_name'],data=bike_share, palette='viridis',order=bike_share[bike_share['user_type']=='Casual Member']['to_station_name'].value_counts().index[:5])
plt.title('Top 5 Arriving Stations (Casual Members)')
daily_activity = bike_share.groupby(by=['weekday_start','hour_start']).count()['user_type'].unstack()
daily_activity.head()
plt.figure(figsize=(12,6))
sns.heatmap(daily_activity,cmap='coolwarm')
