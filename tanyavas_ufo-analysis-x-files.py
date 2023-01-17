%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

plt.style.use('ggplot')
data = pd.read_csv('../input/scrubbed.csv', usecols=[0, 1, 2, 4, 9, 10], low_memory=False)
data.head()
data.info()
data[data['state'].isnull()].head()
data = data.dropna(axis=0)
data.info()
data['datetime'] = pd.to_datetime(data['datetime'],errors='coerce')

data['latitude'] = pd.to_numeric(data['latitude'],errors='coerce')

data = data.rename(columns={'longitude ': 'longitude'})
data.info()
data['shape'].value_counts().plot(kind='bar',figsize=(10,6));

plt.xticks(rotation=45)

plt.xlabel('Shapes')

plt.ylabel('Number of observations')

plt.title('Number of observations by shape');
data.insert(1,'year',data['datetime'].dt.year)

data.insert(2,'month',data['datetime'].dt.month)
data['state'].value_counts().plot(kind='bar',figsize=(12,8));

plt.title('Most visited states');
plt.figure(figsize=(12,8))

Map = Basemap(projection='mill', llcrnrlat = 26, urcrnrlat = 50, llcrnrlon = -120, urcrnrlon = -65, 

             resolution = 'h')

Map.drawcoastlines()

Map.drawcountries()

Map.drawstates()

x, y = Map(list(data['longitude'].astype("float")), list(data['latitude'].astype(float)))

Map.plot(x, y, 'go', markersize = 1, alpha = 0.5, color = 'blue');
year1910_1950 = np.arange(1910,1950)

year1950_1970 = np.arange(1950,1970)

year1970_1990 = np.arange(1970,1990)

year1990_2000 = np.arange(1990,2000)

year2000_2014 = np.arange(2000,2015)
data1910_1950 = data[data['year'].isin(year1910_1950)]

data1950_1970 = data[data['year'].isin(year1950_1970)]

data1970_1990 = data[data['year'].isin(year1970_1990)]

data1990_2000 = data[data['year'].isin(year1990_2000)]

data2000_2014 = data[data['year'].isin(year2000_2014)]
plt.figure(figsize=(12,8))

Map = Basemap(projection='mill', llcrnrlat = 26, urcrnrlat = 50, llcrnrlon = -120, urcrnrlon = -65, 

             resolution = 'h')

Map.drawcoastlines()

Map.drawcountries()

Map.drawstates()

x, y = Map(list(data1910_1950['longitude'].astype("float")), list(data1910_1950['latitude'].astype(float)))

Map.plot(x, y, 'go', markersize = 5, alpha = 0.9, color = 'blue');
pd.pivot_table(data1910_1950,index='year',values='state',aggfunc='count').plot(figsize=(10,6));

plt.xticks(year1910_1950)

plt.xticks(rotation = 45)

plt.title('1910 - 1949');
pd.pivot_table(data1950_1970,index='year',values='state',aggfunc='count').plot(figsize=(10,6));

plt.xticks(year1950_1970)

plt.xticks(rotation = 45)

plt.title('1950 - 1969');
pd.pivot_table(data1970_1990,index='year',values='state',aggfunc='count').plot(figsize=(10,6));

plt.xticks(year1970_1990)

plt.xticks(rotation = 45)

plt.title('1970 - 1989');
pd.pivot_table(data1990_2000,index='year',values='state',aggfunc='count').plot(figsize=(10,6));

plt.xticks(year1990_2000)

plt.xticks(rotation = 45)

plt.title('1990 - 1999');
pd.pivot_table(data2000_2014,index='year',values='state',aggfunc='count').plot(figsize=(10,6));

plt.xticks(year2000_2014)

plt.xticks(rotation = 45);

plt.title('2000 - 2014');
pd.pivot_table(data,index='year',values='state',aggfunc='count').plot(figsize=(10,6));

plt.title('1910 - 2014');
pd.pivot_table(data,index=('month'),values='state',aggfunc='count').plot(figsize=(10,6));

plt.xticks(np.arange(1,13));

plt.title('Sightings by month');