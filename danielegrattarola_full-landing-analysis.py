# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.mode.chained_assignment = None 

import seaborn as sns

from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/meteorite-landings.csv')

# Remove rows with null values

clean = data.dropna()

clean.head()
plt.figure(figsize=(9,7))

my_map = Basemap(projection='robin', lat_0=0, lon_0=0)

my_map.drawcoastlines()

my_map.drawcountries()

my_map.fillcontinents(color='gray', alpha=0.2)

my_map.drawmapboundary()

my_map.drawmeridians(np.arange(0, 360, 30))

my_map.drawparallels(np.arange(-90, 90, 30))



x, y = my_map(clean.reclong.values, clean.reclat.values)

my_map.scatter(x, y, marker='.', c='aqua', edgecolor='None', alpha=0.5)

plt.show()
clean.year.hist(bins=np.arange(1800,2014,1),figsize=(8,3))

clean.replace(to_replace='Fell', value='Seen', inplace=True)

df = clean.groupby(['fall', 'year']).size().unstack().T.fillna(0).astype(int)

df = df.reset_index()

df = df[df.year >= 1800][df.year <= 2016]

df.head()
df.plot(x='year', y='Seen', figsize=(8,3))

df.plot(x='year', y='Found', figsize=(8,3))
plt.figure(figsize=(9,7))

my_map = Basemap(projection='robin', lat_0=0, lon_0=0)

my_map.drawcoastlines()

my_map.drawcountries()

my_map.fillcontinents(color='gray', alpha=0.2)

my_map.drawmapboundary()

my_map.drawmeridians(np.arange(0, 360, 30))

my_map.drawparallels(np.arange(-90, 90, 30))



seen = clean[clean.fall == 'Seen']

found = clean[clean.fall == 'Found']

x, y = my_map(seen.reclong.values, seen.reclat.values)

my_map.scatter(x, y, marker='.', c='red', edgecolor='None', alpha=0.5, label='Seen')

x, y = my_map(found.reclong.values, found.reclat.values)

my_map.scatter(x, y, marker='.', c='aqua', edgecolor='None', alpha=0.5, label='Found')

plt.legend(loc=0, frameon=True, fontsize='x-small')

plt.show()