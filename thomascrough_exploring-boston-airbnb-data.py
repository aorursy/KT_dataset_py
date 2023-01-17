# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from mpl_toolkits.basemap import Basemap, cm

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

data_source = check_output(["ls", "../input"]).decode("utf8")

print(data_source)



# Any results you write to the current directory are saved as output.
data_source = [str(i) for i in data_source.split()]

##for f in data_source:

##    print(f)

df = pd.concat((pd.read_csv('../input/'+f) for f in data_source))
##df.info()

##df.describe()

len_1 = len(df.columns)



## Remove columns ending in url or beginning with host/review

new_cols = [c for c in df.columns if c[-4:] != '_url' if c[:6] != 'review']



##print(new_cols)

df = df[new_cols]

##print(len_1 - len(df.columns), " Columns removed.")

##print(df.info())

## Adjust price column to be numbers rather than strings

df2 = df.loc[:, ['id', 'price', 'latitude', 'longitude']]

df2 = df2[pd.notnull(df2['price'])]

df2['price'] = df2['price'].str.replace(',', '')

df2['price'] = df2['price'].str.replace('$', '')

df2['price'] = df2['price'].astype(float)









"""

print(df2.head())

print(df2.tail())

"""

df2 = df2.dropna()

print(df2.head())

## I noticed the color mapping was being severely distorted by a few large values in the pricing

## Decided to deal with the problem by removing outliers

df2 = df2[np.abs(df2.price-df2.price.mean())<=(3*df2.price.std())]

print(df2.shape)

print(df2['price'].mean())





##coordinates for plotting 



long_max = df['longitude'].max() + .02

long_min = df['longitude'].min() -.02

mid_long = (df['longitude'].min() + df['longitude'].max())/2



lat_max = df['latitude'].max() + .02

lat_min = df['latitude'].min() - .02

mid_lat = (df['latitude'].min() + df['latitude'].max())/2



## map

m = Basemap(projection='cyl',lat_0=mid_lat,lon_0=mid_long,\

            llcrnrlat=lat_min,urcrnrlat=lat_max,\

            llcrnrlon=long_min,urcrnrlon=long_max,\

            rsphere=6371200.,resolution='h',area_thresh=10)

m.drawcoastlines()

m.drawstates()

m.drawcounties()

m.shadedrelief()



## locations

x, y = m(df2['longitude'], df2['latitude'])

sp = plt.scatter(x, y, c=df2['price'], s=15)







plt.rcParams["figure.figsize"] = [11,7]

cb = plt.colorbar(sp)

cb.set_label('Price($)')

plt.show()

plt.clf()
df3 = df2.sample(frac=.25)



m2 = Basemap(projection='cyl',lat_0=mid_lat,lon_0=mid_long,\

            llcrnrlat=lat_min,urcrnrlat=lat_max,\

            llcrnrlon=long_min,urcrnrlon=long_max,\

            rsphere=6371200.,resolution='h',area_thresh=10)

m2.drawcoastlines()

m2.drawstates()

m2.drawcounties()

m2.shadedrelief()



x1, y1 = m(df3['longitude'], df3['latitude'])

sp1 = plt.scatter(x1, y1, c=df3['price'], s=15)







plt.rcParams["figure.figsize"] = [11,7]

cb1 = plt.colorbar(sp1)

cb1.set_label('Price($)')

plt.show()