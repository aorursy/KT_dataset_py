
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
data = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1')
data = data[data['iyear'] > 1995]
data.dropna()
lon = [float(x) for x in data['longitude']]
lat = [float(y) for y in data['latitude']]
plt.figure(figsize=(16,12))

map = Basemap()
map.drawcoastlines()
map.drawcountries()
map.fillcontinents(color="grey", lake_color="white")

x, y = map(lon, lat)
map.plot(x,y,'ro', markersize=1)

plt.title('Geographic Distribution of Terrorist Activities between 1996-2016')

plt.show()
year = data['iyear'].unique()
attacks = data.groupby('iyear').count()['eventid']
plt.figure(figsize=(12,6))
plt.plot(year, attacks)
plt.xlabel('year')
plt.ylabel('number of attacks')
plt.title('Number of Terrorists Activities between 1996-2016')

plt.xticks(year)
plt.axvline(x=2003, color='r')
plt.text(2000.5,5000,'US invasion\n   of Iraq',fontsize=12)

plt.axvline(x=2004, color='r')
plt.text(2004.2,8000,'Al-Qaeda\n formed',fontsize=12)

plt.show()