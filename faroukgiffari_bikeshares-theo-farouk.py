# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df1 = pd.read_csv('/kaggle/input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q1 2018.csv')

df2 = pd.read_csv('/kaggle/input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q2 2018.csv')

df3 = pd.read_csv('/kaggle/input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q3 2018.csv')

df4 = pd.read_csv('/kaggle/input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q4 2018.csv')
Data2018 = df1.copy()

Data2018 = Data2018.append([df2, df3, df4], sort=False)

del df1

del df2

del df3

del df4
Data2018.head(10)
# prepare the dataset

Data2018['time'] = pd.to_datetime(Data2018.trip_start_time)

Data2018.index = Data2018.time.dt.date

Data2018.index.name = 'index'

# cleaning the dataset

Data2018 = Data2018.dropna(subset=['from_station_name','to_station_name'])

df = Data2018.copy()


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
outmap.head(10) # number of bikes leaves the station
inmap.head(10) # number of bikes entering the station
print('number of station with enough bike to use next morning, aka number of bikes entering > number of bikes leaving the station')

((inmap - outmap)>=0).sum(axis=1)
print('number of station with less bike to use next morning, or need a crew to return bikes back to station before next morning')

((inmap - outmap)<0).sum(axis=1)
print('Station and the total number of days in 2017 where stations need more bikes to be returned by the crew every night')

((inmap - outmap)<0).sum(axis=0).sort_values(ascending=False)[:20]
bike_minus = inmap - outmap # incoming bikes minus leaving bikes

bike_minus = np.absolute(bike_minus[bike_minus < 0]) # show only minus value
bike_minus.head(10) # number of bikes that required by crew to be returned to each station
print('20 Stations with the highest number of required returned bikes in a day')

np.max(bike_minus, axis=0).sort_values(ascending=False)[:20]