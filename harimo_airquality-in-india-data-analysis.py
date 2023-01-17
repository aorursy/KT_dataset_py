# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
airQuality = pd.read_excel('../input/AirQuality.xlsx')
airQuality.head()
airQuality.info()
airQuality.describe()
airQuality.columns
#List of Countries
print('Num of Unique Countries : ',airQuality['Country'].nunique())
airQuality['Country'].unique()
airQuality.drop(['Country', 'lastupdate'], axis = 1, inplace = True)
airQuality.head()
#List of States
airQuality['State'].unique()
plt.figure(figsize=(17,7), dpi = 100)
sns.countplot(x='State',data=airQuality)
plt.xlabel('State')
plt.tight_layout()
# Grouping by States
by_state = airQuality.groupby('State')
# Mean Pollution
by_state.mean()
plt.figure(figsize=(17,7), dpi = 100)
#by_state.mean().plot()
plt.plot(by_state.mean())
plt.tight_layout()
plt.legend(['Avg', 'Max', 'Min'])
plt.xlabel('State')
plt.title('Avg Pollution by State')
by_state.max()
plt.figure(figsize=(17,7), dpi = 100)
#by_state.mean().plot()
plt.plot(by_state.max()['Avg'])
plt.plot(by_state.max()['Max'])
plt.plot(by_state.max()['Min'])
plt.tight_layout()
plt.legend(['Avg', 'Max', 'Min'])
plt.xlabel('State')
plt.title('Max Pollution by State')
print('List of Cities by State with Max Pollution')
by_state.max()['city']
by_state.min()
plt.figure(figsize=(17,7), dpi = 100)
#by_state.mean().plot()
plt.plot(by_state.max()['Avg'])
plt.plot(by_state.max()['Max'])
plt.plot(by_state.max()['Min'])
plt.tight_layout()
plt.legend(['Avg', 'Max', 'Min'])
plt.xlabel('State')
plt.title('Min Pollution by State')
print('List of Cities by State with Min Pollution')
by_state.min()['city']
sns.countplot(x='Pollutants', data = airQuality)
airQuality['Pollutants'].value_counts()
airQuality['State'].nunique()
state = list(airQuality['State'].unique())
fig, axes = plt.subplots(nrows=5,ncols=4,figsize=(17,20))
i = 0
for st in state:
    airQualityState = airQuality[airQuality['State'] == st]
    plot = sns.countplot(x='Pollutants', data = airQualityState, ax=axes.flatten()[i])
    plot.set_title(st)
    plt.tight_layout()
    i = i + 1
state = list(airQuality['State'].unique())
fig, axes = plt.subplots(nrows=5,ncols=4,figsize=(17,20))
i = 0
for st in state:
    airQualityState = airQuality[airQuality['State'] == st]
    plot = sns.countplot(x='city', data = airQualityState, ax=axes.flatten()[i])
    plot.set_title(st)
    plot.set_xticklabels(plot.get_xticklabels(), rotation = 45)
    plt.tight_layout()
    i = i + 1
list(airQuality['Pollutants'].unique())
pollutant = list(airQuality['Pollutants'].unique())
for poll in pollutant:
    plt.figure(figsize=(17,7), dpi = 100)
    sns.countplot(airQuality[airQuality['Pollutants'] == poll]['State'], data = airQuality)
    plt.tight_layout()
    plt.title(poll)
max_value = list()
avg_value = list()
min_value = list()
pollutant = list(airQuality['Pollutants'].unique())
for poll in pollutant:
    max_value.append(airQuality[airQuality['Pollutants'] == poll]['Max'].max())
    avg_value.append(airQuality[airQuality['Pollutants'] == poll]['Avg'].mean())
    min_value.append(airQuality[airQuality['Pollutants'] == poll]['Min'].min())
fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(17,7))
axes[0].set_ylabel('Max Value')
axes[0].bar(pollutant, max_value)
axes[1].set_ylabel('Avg Value')
axes[1].bar(pollutant, avg_value)
axes[2].set_ylabel('Min Value')
axes[2].bar(pollutant, min_value)
airQuality.columns
airQuality['city'].unique()
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
import matplotlib.cm
#data = {'City': ['Delhi', 'Ahmedabad', 'Bengaluru', 'Aurangabad', 'Mumbai', 'Nagpur', 'Nashik', 'Pune', 'Amritsar', 'Jaipur', 'Chennai', 'Hyderabad', 'Agra', 'Kanpur', 'Lucknow', 'Kolkata'],
#        'Lat':  [28.70, 23.02, 12.58, 19.87, 18.59, 21.08, 19.99, 18.52, 31.63, 26.91, 13.08, 17.38, 27.17, 26.44, 26.84, 22.57],
#        'Long': [77.10, 72.35, 77.34, 75.34, 72.50, 79.05, 73.78, 73.85, 74.87, 75.78, 80.27, 78.48, 78.00, 80.33, 80.94, 88.36]}
data = {
        'City': ['Delhi',  
                 'Amaravati', 'Rajamahendravaram', 'Tirupati', 'Vijayawada', 'Visakhapatnam', 'Gaya', 'Muzaffarpur', 'Patna', 
                 'Ahmedabad', 'Faridabad', 'Gurugram', 'Manesar', 'Panchkula', 'Rohtak', 'Jorapokhar', 'Bengaluru', 
                 'Chikkaballapur', 'Hubballi', 'Thiruvananthapuram', 'Dewas', 'Mandideep', 'Pithampur', 'Satna', 'Singrauli', 
                 'Ujjain', 'Aurangabad', 'Chandrapur', 'Mumbai', 'Nagpur', 'Nashik', 'Pune', 'Solapur', 'Thane', 'Brajrajnagar',
                 'Talcher', 'Amritsar', 'Bathinda', 'Jalandhar', 'Khanna', 'Ludhiana', 'Mandi Gobindgarh', 'Patiala', 'Rupnagar', 
                 'Ajmer', 'Alwar', 'Bhiwadi', 'Jaipur', 'Jodhpur', 'Kota', 'Pali', 'Udaipur', 'Chennai', 'Hyderabad', 'Agra', 
                 'Baghpat', 'Bulandshahr', 'Ghaziabad', 'Greater_Noida', 'Kanpur', 'Lucknow', 'Moradabad', 'Muzaffarnagar', 
                 'Noida', 'Varanasi', 'Asanol', 'Durgapur', 'Haldia', 'Howrah', 'Kolkata', 'Siliguri'],
        'Lat' : [28.70, 
                 20.93, 17.00, 13.62, 16.50, 17.68, 24.79, 26.12, 25.59,
                 23.02, 28.40, 28.45, 28.35, 30.69, 28.89, 23.70, 12.97,
                 13.43, 15.36, 8.52,  22.96, 23.09, 22.61, 24.60, 24.19,
                 23.17, 19.87, 19.97, 19.07, 21.14, 19.99, 18.52, 17.65, 19.21, 21.82,
                 20.95, 31.63, 30.21, 31.32, 30.70, 30.90, 30.66, 30.33, 30.96,
                 26.44, 27.55, 28.20, 26.91, 26.23, 25.21, 25.77, 24.58, 13.08, 17.38, 27.17,
                 28.94, 28.40, 28.66, 28.47, 26.44, 26.84, 28.83, 29.47,
                 28.53, 25.31, 23.67, 23.52, 22.06, 22.59, 22.57, 26.72],
        'Long': [77.10, 
                 77.77, 81.80, 79.41, 80.64, 83.21, 85.00, 85.36, 85.13,
                 72.57, 77.31, 77.02, 76.93, 76.86, 76.60, 86.41, 77.59,
                 77.72, 75.12, 76.93, 76.05, 77.50, 75.67, 80.83, 82.66,
                 75.78, 75.34, 79.30, 72.87, 79.08, 73.78, 73.85, 75.90, 72.97, 83.92,
                 85.21, 74.87, 74.94, 75.57, 76.21, 75.85, 76.30, 76.38, 76.52,
                 74.63, 76.63, 76.84, 75.78, 73.02, 75.86, 73.32, 73.71, 80.27, 78.48, 78.00,
                 77.22, 77.84, 77.45, 77.50, 80.33, 80.94, 78.77, 77.70,
                 77.39, 82.97, 86.95, 87.31, 88.06, 88.26, 88.36, 88.39]
       }
dfr = pd.DataFrame(data, columns = ['City', 'Lat', 'Long'])
print(data['City'])
airQuality.Pollutants.unique()
for c in list(airQuality['city'].unique()):
    dfr.loc[dfr['City'] == c,'PM2.5'] = len(airQuality[(airQuality['city'] == c) & (airQuality['Pollutants'] == 'PM2.5')])
    dfr.loc[dfr['City'] == c,'PM10']  = len(airQuality[(airQuality['city'] == c) & (airQuality['Pollutants'] == 'PM10')])
    dfr.loc[dfr['City'] == c,'NO2']   = len(airQuality[(airQuality['city'] == c) & (airQuality['Pollutants'] == 'NO2')])
    dfr.loc[dfr['City'] == c,'NH3']   = len(airQuality[(airQuality['city'] == c) & (airQuality['Pollutants'] == 'NH3')])
    dfr.loc[dfr['City'] == c,'SO2']   = len(airQuality[(airQuality['city'] == c) & (airQuality['Pollutants'] == 'SO2')])
    dfr.loc[dfr['City'] == c,'CO']    = len(airQuality[(airQuality['city'] == c) & (airQuality['Pollutants'] == 'CO')])
    dfr.loc[dfr['City'] == c,'OZONE'] = len(airQuality[(airQuality['city'] == c) & (airQuality['Pollutants'] == 'OZONE')])
    dfr.loc[dfr['City'] == c,'Avg']   = airQuality[(airQuality['city'] == c)]['Avg'].mean()
    dfr.loc[dfr['City'] == c,'Max']   = airQuality[(airQuality['city'] == c)]['Max'].mean()
    dfr.loc[dfr['City'] == c,'Min']   = airQuality[(airQuality['city'] == c)]['Min'].mean()
dfr.head()
plt.figure(figsize=(20,20))
map = Basemap(projection='aeqd', lat_0 = 20.7, lon_0 = 82.71, width = 5000000, height = 4000000, resolution='l') # set res=h
map.drawmapboundary(fill_color='cyan')
map.etopo()
map.drawcoastlines()
map.drawcountries()
map.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
scale = 1
#for c in list(newDf['City'].unique()):
    ##print(c, ":", newDf[newDf['City'] == c]['Number of Reviews'].sum())
    #dfr.loc[dfr['City'] == c, 'Total Num of Reviews'] = newDf[newDf['City'] == c]['Number of Reviews'].sum()
for i in range(0,len(dfr)):
    x, y = map(dfr.ix[i,'Long'], dfr.ix[i,'Lat'])
    map.plot(x,y,marker='o', color='Red', markersize=5)
plt.title('List of cities')
plt.show()
for p in list(airQuality.Pollutants.unique()):
    plt.figure(figsize=(20,20))
    map = Basemap(projection='aeqd', lat_0 = 20.7, lon_0 = 82.71, width = 5000000, height = 4000000, resolution='l') # set res=h
    map.drawmapboundary(fill_color='cyan')
    map.etopo()
    map.drawcoastlines()
    map.drawcountries()
    map.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
    scale = 2
#for c in list(newDf['City'].unique()):
    ##print(c, ":", newDf[newDf['City'] == c]['Number of Reviews'].sum())
    #dfr.loc[dfr['City'] == c, 'Total Num of Reviews'] = newDf[newDf['City'] == c]['Number of Reviews'].sum()
    for i in range(0,len(dfr)):
        x, y = map(dfr.ix[i,'Long'], dfr.ix[i,'Lat'])
        map.plot(x,y,marker='o', color='Red', alpha = 0.5, markersize=int(dfr.ix[i,p]*scale))
    plt.title('{} Emission Spread'.format(p))
    plt.show()
for p in ['Avg', 'Max', 'Min']:
    plt.figure(figsize=(20,20))
    map = Basemap(projection='aeqd', lat_0 = 20.7, lon_0 = 82.71, width = 5000000, height = 4000000, resolution='l') # set res=h
    map.drawmapboundary(fill_color='cyan')
    map.etopo()
    map.drawcoastlines()
    map.drawcountries()
    map.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
    scale = 0.1
#for c in list(newDf['City'].unique()):
    ##print(c, ":", newDf[newDf['City'] == c]['Number of Reviews'].sum())
    #dfr.loc[dfr['City'] == c, 'Total Num of Reviews'] = newDf[newDf['City'] == c]['Number of Reviews'].sum()
    for i in range(0,len(dfr)):
        x, y = map(dfr.ix[i,'Long'], dfr.ix[i,'Lat'])
        map.plot(x,y,marker='o', color='Red', alpha = 0.5, markersize=int(dfr.ix[i,p]*scale))
    plt.title('{} Pollutant Spread'.format(p))
    plt.show()
