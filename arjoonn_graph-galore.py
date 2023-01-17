import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from mpl_toolkits.basemap import Basemap



%pylab inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/armories_data - 20161201.csv')

df.info()
lat, lng = df.Lat, df.Long

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



df['Lead present?'] = df['Lead present?'].replace('Unknown', np.nan)

present = df.loc[df['Lead present?'] == 'Yes']

absent = df.loc[df['Lead present?'] == 'No']





mp.scatter(present.Long, present.Lat, color='red', latlon=True)

mp.scatter(absent.Long, absent.Lat, color='green', latlon=True)

plt.title('Was Lead present?')
df['Lead present outside range?'] = df['Lead present outside range?'].replace('Unknown', np.nan)
plt.figure(figsize=(10, 15))

mp = Basemap(llcrnrlat=lllat,

             llcrnrlon=lllon,

             urcrnrlat=urlat,

             urcrnrlon=urlon,

             resolution='h'

            )

mp.drawcountries()

mp.drawcoastlines(linewidth=0.5)



df['Lead present outside range?'] = df['Lead present outside range?'].replace('Unknown', np.nan)

present = df.loc[df['Lead present outside range?'] == 'Yes']

absent = df.loc[df['Lead present outside range?'] == 'No']





mp.scatter(present.Long, present.Lat, color='red', latlon=True)

mp.scatter(absent.Long, absent.Lat, color='green', latlon=True)

plt.title('Lead present outside range?')
plt.figure(figsize=(10, 15))

mp = Basemap(llcrnrlat=lllat,

             llcrnrlon=lllon,

             urcrnrlat=urlat,

             urcrnrlon=urlon,

             resolution='h'

            )

mp.drawcountries()

mp.drawcoastlines(linewidth=0.5)



mp.scatter(df.Long, df.Lat, s=20, alpha=0.1, color='blue', latlon=True)

plt.title('Highest lead level detected (ug)')
plt.figure(figsize=(15, 5))

sns.countplot(df['Inspection year'])
df['LeadLevels'] = df['Highest lead level detected (ug)'].str.replace(',', '').astype(float)



plt.subplots(figsize=(15, 10))

plt.subplot(2, 1, 1)

sns.boxplot(x='Inspection year', y='LeadLevels', data=df)

plt.subplot(2, 1, 2)

data = df.loc[df.LeadLevels < 120000]

sns.boxplot(x='Inspection year', y='LeadLevels', data=data)