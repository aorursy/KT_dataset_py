import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from mpl_toolkits.basemap import Basemap



import seaborn as sns

%pylab inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/PakistanDroneAttacks.csv', encoding='latin1')

df = df.dropna(subset=['Latitude', 'Longitude'])

lat, lng = df.Latitude, df.Longitude

lons, lats = lng.unique(), lat.unique()



scale, alpha = 10, 0.5

lllat = lats.min() - 4

lllon = lons.min() - 6

urlat = lats.max() + 6

urlon = lons.max() + 6



print(lllat, lllon, urlat, urlon, 'is our bounding box')
plt.figure(figsize=(10, 15))

mp = Basemap(llcrnrlat=lllat,

             llcrnrlon=lllon,

             urcrnrlat=urlat,

             urcrnrlon=urlon,

             resolution='h'

            )

mp.drawcountries()

mp.drawcoastlines(linewidth=0.5)



mp.scatter(lng, lat, s=df['Injured Min']*scale, alpha=alpha, color='lightgreen', latlon=True)

mp.scatter(lng, lat, s=df['Injured Max']*scale, alpha=alpha, color='darkgreen', latlon=True)

mp.scatter(lng, lat, s=df['Total Died Min']*scale, alpha=alpha, color='pink', latlon=True)

mp.scatter(lng, lat, s=df['Total Died Mix']*scale, alpha=alpha, color='red', latlon=True)
df = df.loc[df.Latitude < 50]



lat, lng = df.Latitude, df.Longitude

lons, lats = lng.unique(), lat.unique()



lllat = lats.min() - 4

lllon = lons.min() - 6

urlat = lats.max() + 3

urlon = lons.max() + 6



print(lllat, lllon, urlat, urlon, 'is our new bounding box')
scale, alpha = 8, 0.5



plt.figure(figsize=(10, 15))

#plt.figure(figsize=(6, 10))

mp = Basemap(llcrnrlat=lllat,

             llcrnrlon=lllon,

             urcrnrlat=urlat,

             urcrnrlon=urlon,

             resolution='h'

            )

mp.drawcountries(linewidth=1)

mp.drawcoastlines(linewidth=1)

mp.drawlsmask(land_color='white', ocean_color='blue')



mp.scatter(lng, lat, s=df['Total Died Min']*scale, alpha=1, color='green', latlon=True)

mp.scatter(lng, lat, s=df['Total Died Mix']*scale, alpha=alpha, color='red', latlon=True)
scale, alpha = 8, 0.5



plt.figure(figsize=(10, 15))

#plt.figure(figsize=(6, 10))

mp = Basemap(llcrnrlat=lllat,

             llcrnrlon=lllon,

             urcrnrlat=urlat,

             urcrnrlon=urlon,

             resolution='h'

            )

mp.drawcountries(linewidth=1)

mp.drawcoastlines(linewidth=1)

mp.drawlsmask(land_color='white', ocean_color='blue')



mp.scatter(lng, lat, s=df['Injured Min']*scale, alpha=1, color='green', latlon=True)

mp.scatter(lng, lat, s=df['Injured Max']*scale, alpha=alpha, color='red', latlon=True)
scale, alpha = 8, 0.5



plt.figure(figsize=(10, 15))

#plt.figure(figsize=(6, 10))

mp = Basemap(llcrnrlat=lllat,

             llcrnrlon=lllon,

             urcrnrlat=urlat,

             urcrnrlon=urlon,

             resolution='h'

            )

mp.drawcountries(linewidth=1)

mp.drawcoastlines(linewidth=1)

mp.drawlsmask(land_color='white', ocean_color='blue')

mp.etopo(alpha=0.5)



mp.scatter(lng, lat, s=df['Total Died Min']*scale, alpha=1, color='green', latlon=True)

mp.scatter(lng, lat, s=df['Total Died Mix']*scale, alpha=alpha, color='red', latlon=True)