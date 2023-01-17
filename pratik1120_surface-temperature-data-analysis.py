import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# reading in GlobalLandTemperaturesByCountry.csv

gltc = pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv')
gltc.head(2)
#separating zimbabwe data
df = gltc[gltc['Country']=='Zimbabwe']

#dropping rows with NaN values
df.dropna(inplace=True)

# first lets bifurcate the months and year data for the dt
df.loc[:,'dt'] = pd.to_datetime(df['dt'])

df.loc[:,'month'] = [x.month for x in list(df['dt'])]
df.loc[:,'year'] = [x.year for x in list(df['dt'])]
plt.plot(df['dt'], df['AverageTemperature'])
plt.show()
fig = plt.figure(figsize=(10,5))
plt.plot(df.loc[df['year']==2012, 'dt'], df.loc[df['year']==2012,'AverageTemperature'])
plt.title('Temperature at Zimbabwe in 2012')
plt.xlabel('Months')
plt.ylabel('Average Temperature')
plt.show()
#checking highest temperature

gltc[gltc['AverageTemperature']==gltc['AverageTemperature'].max()]
#lets analyse Kuwait daa throughout 2012

df = gltc[gltc['Country']=='Kuwait']
df.dropna(inplace=True)
df.loc[:,'dt'] = pd.to_datetime(df['dt'])
df.loc[:,'month'] = [x.month for x in list(df['dt'])]
df.loc[:,'year'] = [x.year for x in list(df['dt'])]
fig = plt.figure(figsize=(10,5))
plt.plot(df.loc[df['year']==2012, 'dt'], df.loc[df['year']==2012,'AverageTemperature'])
plt.show()
mean_temp = gltc['AverageTemperature'].mean()
#lets plot again
fig = plt.figure(figsize=(10,5))
plt.plot(df.loc[df['year']==2012, 'dt'], df.loc[df['year']==2012,'AverageTemperature'])
plt.axhline(mean_temp)
plt.show()
fig = plt.figure(figsize=(10,5))
years = [2008,2009,2010,2011,2012]
for year in years:
    plt.plot(df.loc[df['year']==year, 'month'], df.loc[df['year']==year,'AverageTemperature'], label=year)
plt.title('Temperature variation in Kuwait in last 5 years')
plt.xlabel('Months')
plt.ylabel('Average Temperature')
plt.legend(loc='upper left')
plt.axhline(mean_temp)
plt.show()
df['year'].unique()
mean_by_year = []
for year in list(df['year'].unique()):
    df1 = df[df['year']==year]
    mean_by_year.append(df1['AverageTemperature'].mean())
    
fig = plt.figure(figsize=(15,10))
plt.bar(list(df['year'].unique()), mean_by_year)
plt.show()
mean_by_year = []
years = list(df['year'].unique()[-10:])
for year in years:
    df1 = df[df['year']==year]
    mean_by_year.append(df1['AverageTemperature'].mean())
    
fig = plt.figure(figsize=(10,5))
barlist = plt.bar(years, mean_by_year)
barlist[0].set_color('r')
barlist[1].set_color('b')
barlist[2].set_color('g')
barlist[3].set_color('r')
barlist[4].set_color('y')
barlist[5].set_color('m')
barlist[6].set_color('c')
barlist[7].set_color('k')
barlist[8].set_color('g')
barlist[9].set_color('y')
plt.xticks(np.arange(2004,2014),labels=years)
plt.title('Temperature in Kuwait in last decade')
plt.show()
gltc['dt'] = pd.to_datetime(gltc['dt'])
df = gltc[gltc['dt'].dt.year==2012]
df = df.sort_values('AverageTemperature', ascending=False)
df.head()
df = gltc[(gltc['dt'].dt.year==2012) & (gltc['dt'].dt.month==7)]
df = df.sort_values('AverageTemperature', ascending=False)
top_countries = list(df['Country'].head())
df.head()
gltc['month'] = gltc['dt'].dt.month
gltc['year'] = gltc['dt'].dt.year
fig = plt.figure(figsize=(10,5))
for country in top_countries:
    plt.plot(gltc.loc[(gltc['year']==2012)&(gltc['Country']==country), 'month'], gltc.loc[(gltc['year']==2012)&(gltc['Country']==country),'AverageTemperature'], label=country)
plt.legend(loc="upper left")
plt.title('Top 5 countries with highest Average Soil Temperature for 2012')
plt.xlabel('Months')
plt.ylabel('Average Temperature')
plt.show()
data = pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCity.csv')
data.tail(10)
data[data['AverageTemperature']==data['AverageTemperature'].max()]
data['dt'] = pd.to_datetime(data['dt'])
data['year'] = data['dt'].dt.year
data['month'] = data['dt'].dt.month
df = data[data['year']==2013]
df.sort_values('AverageTemperature', ascending=False, inplace=True)
df = df.reset_index()
df.head()
cities = df['City'][:5].tolist()
fig = plt.figure(figsize=(10,5))
for city in cities:
    df1 = df[df['City']==city]
    df1.sort_values('month', inplace=True)
    plt.plot(df1['month'], df1['AverageTemperature'], label=city)
plt.legend(loc='upper left')
plt.title('Top 5 cities having highest temperature in 2013')
plt.show()
data['City'].nunique()
def clean_lat_data(lat):
    lat1 = lat.replace('N','')
    lat1 = float(lat1.replace('S',''))
    if lat[-1] == 'S':
        lat1 *= (-1)
    return lat1

def clean_lon_data(lon):
    lon1 = lon.replace('E','')
    lon1 = float(lon1.replace('W',''))
    if lon[-1] == 'W':
        lon1 *= (-1)
    return lon1
df = data[['AverageTemperature','City','Latitude','Longitude']]
cities = list(df['City'].unique())
latitudes = []
longitudes = []
temp = []
for city in cities:
    df1 = df[df['City']==city]
    df1.dropna(inplace=True)
    temp.append(df1['AverageTemperature'].mean())
    lats = df1['Latitude'].tolist()
    lons = df1['Longitude'].tolist()
    lat = clean_lat_data(lats[0])
    lon = clean_lon_data(lons[0])
    latitudes.append(lat)
    longitudes.append(lon)
len(cities), len(latitudes), len(longitudes), len(temp)
df = data[['AverageTemperature','Country','Latitude','Longitude']]
countries = list(df['Country'].unique())
latitudes = []
longitudes = []
temp = []
for country in countries:
    df1 = df[df['Country']==country]
    df1.dropna(inplace=True)
    temp.append(df1['AverageTemperature'].mean())
    lats = df1['Latitude'].tolist()
    lons = df1['Longitude'].tolist()
    lat = clean_lat_data(lats[0])
    lon = clean_lon_data(lons[0])
    latitudes.append(lat)
    longitudes.append(lon)
len(countries), len(latitudes), len(longitudes), len(temp)
new_df = pd.DataFrame({'Country':countries, 'Latitude': latitudes, 'Longitude':longitudes, 'AverageTemperature': temp})
new_df.head()
#summoning the libraries needed
from mpl_toolkits.basemap import Basemap

m = Basemap(projection="merc", llcrnrlat=-40, urcrnrlat=60, llcrnrlon=-50, urcrnrlon=150)

#creating instances
x , y = m(new_df["Longitude"].tolist(),new_df["Latitude"].tolist())
fig = plt.figure(figsize=(10,7))
plt.title("Temperature of Countries")
m.scatter(x, y, s=1, c='red')
m.drawcoastlines()
plt.show()
