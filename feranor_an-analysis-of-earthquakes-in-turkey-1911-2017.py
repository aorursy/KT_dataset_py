#Importing the necessary libraries
import numpy as np
import pandas as pd
import numpy as np
import chardet
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
with open('../input/datasetindex.txt', 'rb') as rawdata:
    result = chardet.detect(rawdata.read(1000000))
    
print(result)
#reading the dataset
data = pd.read_table('../input/datasetindex.txt', sep='\t', encoding='ISO-8859-1', index_col=0)
data.info()
data.sample(5)
col_names = ['earthquake_code', 'date', 'time', 'latitude', 'longitude',
       'depth(km)', 'xM', 'MD', 'ML', 'Mw', 'Ms', 'Mb', 'type', 'location']
data.columns = col_names
data.head(3)
#There was a wrong data of seconds, updated that
sec_err = data['time'].str.extract(r'\:(\d+)\.')[0].apply(lambda x: int(x))
print(sec_err[sec_err>59])
data.loc[13080, 'time'] = '10:03:59.00'
print(sec_err[sec_err>59])
data['date_time'] = pd.to_datetime(data['date'] + ' ' + data['time'])
data = data.drop(['date', 'time'], axis=1)
countries = ['YUNANISTAN', 'GURCISTAN', 'RUSYA', 'IRAN', 'AZERBAYCAN', 'MAKEDONYA', 
             'BULGARISTAN', 'SURIYE', 'IRAK', 'ROMANYA', 'ARNAVUTLUK', 'MISIR', 
             'KIBRIS RUM KESIMI', 'UKRAYNA', 'YUNANiSTAN', 'iRAN', 'BULGARiSTAN',
             'GÜRCiSTAN', 'MISIR', 'SURiYE', 'ISRAIL', 'ONiKi ADALAR YUNANiSTAN',
             'KIBRIS RUM KESiMi']
for country in countries:
    data = data[data.location != country]
    
#delete the points out of Turkey's geolocation
data = data[data.latitude >= 36]
data = data[data.longitude >= 26]
data['city'] = data['location'].str.extract(r'\((.+)\)')
data.city = data.city.fillna(data[data.city.isnull()].location)
data[['date_time', 'city', 'xM']].sort_values('xM', ascending=False).head(10)
data.xM.plot.hist(bins=10)
data['year'] = data.date_time.apply(lambda x: x.year)
data['month'] = data.date_time.apply(lambda x: x.month)
data['weekday'] = data.date_time.apply(lambda x: x.dayofweek)
dataover5 = data[data.xM >= 5]

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
dataover5.month.value_counts().sort_index().plot.bar()
plt.xlabel('Months of the year')
plt.subplot(1,2,2)
dataover5.weekday.value_counts().sort_index().plot.bar()
plt.xlabel('Days of the week')
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
data.month.value_counts().sort_index().plot.bar()
plt.xlabel('Months of the year')
plt.subplot(1,2,2)
data.weekday.value_counts().sort_index().plot.bar()
plt.xlabel('Days of the week')
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(data.year.value_counts().sort_index())
plt.subplot(1,2,2)
plt.plot(data[data.xM >= 5].year.value_counts().sort_index())
data.year.value_counts().sort_index(ascending=False).plot.area()
data[data.xM >= 5].year.value_counts().sort_index(ascending=False).plot.area()
plt.legend(['all earthquakes', '5 plus'])
sel = data[data.xM >= 5]
lon = sel['longitude'].values
lat = sel['latitude'].values
xM = sel['xM'].values

fig = plt.figure(figsize=(12, 6))
m = Basemap(projection='lcc', resolution='l', lat_0=39, lon_0=35, width=1.7E6, height=1E6)
m.bluemarble()
m.drawcoastlines(color='gray')
m.drawcountries(color='red')

m.scatter(lon, lat, latlon=True, c=xM, s=xM*2, cmap='YlOrRd', alpha=0.7)

plt.colorbar()
plt.clim(5, 8)