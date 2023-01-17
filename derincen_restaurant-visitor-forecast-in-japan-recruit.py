# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import numpy as np
import pandas as pd
import scipy.stats as stat
import scipy.io as scipio
import matplotlib.pyplot as plt
%matplotlib inline
import datetime
import folium


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

## Load air data
air_reserve = pd.DataFrame(pd.read_csv('../input/air_reserve.csv'))
air_store = pd.DataFrame(pd.read_csv('../input/air_store_info.csv'))
air_visit = pd.DataFrame(pd.read_csv('../input/air_visit_data.csv'))
## Load hpg data
hpg_reserve = pd.DataFrame(pd.read_csv('../input/hpg_reserve.csv'))
hpg_store = pd.DataFrame(pd.read_csv('../input/hpg_store_info.csv'))
## Load other data
store_id =  pd.DataFrame(pd.read_csv('../input/store_id_relation.csv'))
date_info = pd.DataFrame(pd.read_csv('../input/date_info.csv'))
## Convert visit_date column from str to datetime timestamp
air_visit['visit_date'] = air_visit['visit_date'].\
apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))

air_visit.head()
## Check convert
type(air_visit['visit_date'][0])
# both air_store and hpg_store have latitude longitude cols

# Map of AirReg Restaurants
m = folium.Map(location=[20, 0], tiles="Mapbox Bright", zoom_start=2)
 
for i in range(0,len(air_store)):
    folium.Marker([air_store.iloc[i]['latitude'], air_store.iloc[i]['longitude']], \
                  popup=air_store.iloc[i]['air_store_id']).add_to(m)

m.fit_bounds([[50,145],[30, 140]])
# m.save('airREG_latlong.html')
m
# both air_store and hpg_store have latitude longitude cols

m2 = folium.Map(location=[20, 0], tiles="Mapbox Bright", zoom_start=2)
 
## plot HPF restaurants
for i in range(0,len(air_store)):
    folium.Marker([hpg_store.iloc[i]['latitude'], hpg_store.iloc[i]['longitude']], \
                  popup=air_store.iloc[i]['air_store_id']).add_to(m2)

m2.fit_bounds([[50,145],[30, 140]])
# m2.save('HPG_latlong.html')
m2
## Plot the visitor numbers vs days
plt.figure(figsize=(12,6))
plt.hist(air_visit['visitors'], bins=300, color='c', alpha=0.6, edgecolor='black')
plt.rcParams['font.size'] = 14
plt.title('AirREGI Total Daily Visitor Numbers')
plt.xlabel('Total Number of Visitors in the Day')
plt.ylabel('Number of Days')
plt.xlim(-5, 200)
## Plot the HPG visitor numbers per reservation
plt.figure(figsize=(12,6))
plt.hist(hpg_reserve['reserve_visitors'], bins=100, color='r', alpha=0.6, edgecolor='black')
plt.rcParams['font.size'] = 14
plt.title('HPG Visitor Numbers per Table')
plt.xlabel('Number of Visitors in the Group')
plt.ylabel('Number of Groups')
plt.xlim(-5, 60)
## Plot the HPG visitor numbers per reservation
plt.figure(figsize=(12,6))
plt.hist(air_reserve['reserve_visitors'], bins=100, color='g', alpha=0.6, edgecolor='black')
plt.rcParams['font.size'] = 14
plt.title('HPG Visitor Numbers per Table')
plt.xlabel('Number of Visitors in the Group')
plt.ylabel('Number of Groups')
plt.xlim(-5, 60)
# plt.xticks(np.linspace(0, 60, 30))
## Add 'month' and 'year' columns 
air_visit['month'] = air_visit['visit_date'].apply(lambda x: x.month)
air_visit['year'] = air_visit['visit_date'].apply(lambda x: x.year)
## Create different lists for unique dates in 2016 and 2017 
## Will merge later 
rest_2016 = list(set(list(air_visit[air_visit['year']==2016]['air_store_id'])))
rest_2017 = list(set(list(air_visit[air_visit['year']==2017]['air_store_id'])))
print (len(rest_2016))
print (len(rest_2017))
air_visit_day = air_visit.set_index(['year', 'month','visit_date', 'air_store_id'])
avg_visitor_2017 = list(air_visit_day.loc[2017].groupby('visit_date').mean()['visitors'])
## Get descriptives
daily_m_visitors = air_visit.groupby('visit_date').mean()['visitors']
visitor_stats = pd.DataFrame(air_visit.groupby('visit_date').var())
visitor_stats['mean'] = daily_m_visitors
visitor_stats['stdev'] = visitor_stats['visitors'].apply(lambda x: np.sqrt(x))
visitor_stats['stdev1'] = np.add(np.array(visitor_stats['mean']), np.array(visitor_stats['stdev']))
visitor_stats['stdev2'] = np.add(np.array(visitor_stats['mean']), 2*np.array(visitor_stats['stdev']))
visitor_stats['stdev3'] = np.add(np.array(visitor_stats['mean']), 3*np.array(visitor_stats['stdev']))
visitor_stats['stdev1n'] = np.subtract(np.array(visitor_stats['mean']), np.array(visitor_stats['stdev']))
visitor_stats['stdev2n'] = np.subtract(np.array(visitor_stats['mean']), 2*np.array(visitor_stats['stdev']))
visitor_stats['stdev3n'] = np.subtract(np.array(visitor_stats['mean']), 3*np.array(visitor_stats['stdev']))
visitor_stats.head()
plt.figure(figsize=(20,10))
plt.plot(air_visit.groupby('visit_date').mean()['visitors'], linewidth=3, alpha=0.6\
        , color='b')
plt.plot(visitor_stats['stdev1'], color='r', linewidth=2, alpha=0.6)
plt.plot(visitor_stats['stdev2'], color='darkorange', linewidth=2, alpha=0.6)
plt.plot(visitor_stats['stdev3'], color='y', linewidth=2, alpha=0.6)
plt.plot(visitor_stats['stdev1n'], color='r', linewidth=2, alpha=0.6)
plt.plot(visitor_stats['stdev2n'], color='darkorange', linewidth=2, alpha=0.6)
plt.plot(visitor_stats['stdev3n'], color='y', linewidth=2, alpha=0.6)
plt.title('Average Visitor Numbers with Standard Deviations')
air_visit_rest = air_visit.set_index(['air_store_id', 'year', 'month', 'visit_date'])
plt.figure(figsize=(20,10))
## plot the average number of visitors across all restaurants
day_avg = []

for rest in rest_2017:
    day_avg.append(np.average(np.array(air_visit_rest.loc[rest, 2017].groupby('visit_date').sum()['visitors'])))
    ## plot the total number of daily visitors for each restaurant
    plt.plot(np.array(air_visit_rest.loc[rest, 2017].groupby('visit_date').sum()['visitors']), marker='o', markersize=8, markerfacecolor='white',\
            linewidth=1.5, )
    
    plt.title('Daily Visitor Numbers in 2017')
    plt.xlabel('Days')
    plt.ylabel('Visitors')
plt.figure(figsize=(20,10))
## plot the average number of visitors across all restaurants


for rest in rest_2016:
    ## plot the total number of daily visitors for each restaurant
    plt.plot(list(air_visit_rest.loc[rest, 2016]['visitors']), marker='o', markersize=8, markerfacecolor='white',\
            linewidth=1.5)
    plt.title('Daily Visitor Numbers in 2016')
    plt.xlabel('Days')
    plt.ylabel('Visitors')
pd.DataFrame(pd.read_csv('../rrv-weather-data/weather_stations.csv')).head()
