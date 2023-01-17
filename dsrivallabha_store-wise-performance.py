# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import datetime as dt

import seaborn as sns

import matplotlib.pyplot as plt

import folium

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#read airstore data and read store IDs into a list

airstore = pd.read_csv('../input/air_store_info.csv')

print (airstore.head())

airstorelist = list(airstore['air_store_id'])

print (len(airstorelist))
#read the air visit data and read the date range into a list

airvisit = pd.read_csv('../input/air_visit_data.csv')

airvisit['visit_date'] = pd.to_datetime(airvisit['visit_date']).dt.date

print (airvisit.head())

datelist = list(airvisit['visit_date'].unique())

print (len(datelist))
#test sample values at a random store

parstore = airvisit[airvisit['air_store_id'] == 'air_0f0cdeee6c9bf3d7']

parstore.tail()
#create a new dataframe with store id and date combination

from itertools import product

dailystorevisits = pd.DataFrame(list(product(airstorelist, datelist)), columns=['air_store_id', 'visit_date'])

dailystorevisits['visit_date'] = pd.to_datetime(dailystorevisits['visit_date']).dt.date

print (dailystorevisits.head())
#read the number of visitors to each store, every day, by merging with the airvisit data

dailystorevisits = pd.merge(dailystorevisits, airvisit, how='left', on=['visit_date', 'air_store_id'])

dailystorevisits['visitors'] = dailystorevisits['visitors'].fillna(0)

#print (dailystorevisits.head())

dailystorevisits.describe()
import datetime as dt

dailystorevisits = dailystorevisits[dailystorevisits['visit_date'] > dt.date(2016, 8, 1)]

dailystorevisits.describe()
sns.distplot(dailystorevisits['visitors'], bins=200)

plt.xlabel('number of visitors per day')

plt.ylabel('normalized count')

plt.show()
totstoredays = len(dailystorevisits)

print ('total number of store days: ',  totstoredays)



nocuststoredays = len(dailystorevisits[dailystorevisits['visitors'] == 0])

print ('total number of store days with no customers: ',  nocuststoredays)
frac = 1. * nocuststoredays/totstoredays

print ('Percentage of store days with no customers is: ', 100*frac)
swp = dailystorevisits.groupby(dailystorevisits['air_store_id'], as_index=False).mean()

swp.describe()

#swp.head()
plt.hist(swp['visitors'], bins=50, normed=True)

plt.xlabel('average number of visitors per day')

plt.ylabel('normalized count')

plt.show()
print ('Percentage of stores recieving less than 10 visitors on average are:  ' , 100 * len(swp[swp['visitors'] < 10])/len(swp))
dst = pd.merge(swp, airstore, how='left', on='air_store_id')

plt.hist(dst[dst['visitors'] <  10]['air_genre_name'])

plt.xlabel('average number of visitors per day')

plt.ylabel('normalized count')

plt.xticks(rotation='vertical')

plt.show()
#Marking high and low performance stores by number of visitors

hpstores = dst[dst['visitors'] > 30]

lpstores = dst[dst['visitors'] < 10]
map_osm = folium.Map(location=[40, 140], zoom_start=5)

for lng, lat, desc in zip(hpstores['longitude'], hpstores['latitude'], hpstores['air_genre_name']):

    folium.Marker([lat, lng], popup=desc, icon = folium.Icon(color='blue')).add_to(map_osm)

    



for lng, lat, desc in zip(lpstores['longitude'], lpstores['latitude'], lpstores['air_genre_name']):

    folium.Marker([lat, lng], popup=desc, icon = folium.Icon(color='red')).add_to(map_osm)

map_osm