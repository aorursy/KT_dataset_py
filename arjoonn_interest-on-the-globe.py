import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from mpl_toolkits.basemap import Basemap



%pylab inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_json('../input/train.json')

df.info()
df.interest_level.unique()
lat, lng = df.latitude, df.longitude

lons, lats = lng.unique(), lat.unique()



scale, alpha = 20, 1

lllat = lats.min() - 4

lllon = lons.min() - 6

urlat = lats.max() + 6

urlon = lons.max() + 6



plt.figure(figsize=(15, 10))



mp = Basemap(llcrnrlat=lllat,

             llcrnrlon=lllon,

             urcrnrlat=urlat,

             urcrnrlon=urlon,

             resolution='h'

            )



mp.drawcountries()

mp.drawcoastlines(linewidth=0.5)





for color, level in zip(['orange', 'red', 'green'], ['medium', 'low', 'high']):

    data = df.loc[df.interest_level == level]

    lng, lat = data.longitude, data.latitude

    mp.scatter(lng, lat, s=scale, alpha=alpha, color=color, latlon=True)
df = df.loc[df.longitude < -60]

lat, lng = df.latitude, df.longitude

lons, lats = lng.unique(), lat.unique()



scale, alpha = 20, 1

lllat = lats.min() - 4

lllon = lons.min() - 6

urlat = lats.max() + 6

urlon = lons.max() + 6



plt.figure(figsize=(15, 10))



mp = Basemap(llcrnrlat=lllat,

             llcrnrlon=lllon,

             urcrnrlat=urlat,

             urcrnrlon=urlon,

             resolution='h'

            )



mp.drawcountries()

mp.drawcoastlines(linewidth=0.5)





for color, level in zip(['orange', 'red', 'green'], ['medium', 'low', 'high']):

    data = df.loc[df.interest_level == level]

    lng, lat = data.longitude, data.latitude

    scales = (lng**0)*scale

    mp.scatter(lng, lat, s=scales, alpha=alpha, color=color, latlon=True)