import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 





df = pd.read_csv('/kaggle/input/tree-census/new_york_tree_census_2015.csv')

df2005 = pd.read_csv('/kaggle/input/tree-census/new_york_tree_census_2005.csv')

df1995 = pd.read_csv('/kaggle/input/tree-census/new_york_tree_census_1995.csv')
df.head(10)
df.describe()
df.info()
import seaborn as sns

sns.countplot(x='spc_common', data=df)

#df.drop(df.iloc[:, 1:37], inplace = True, axis = 1)

df = pd.DataFrame(df, columns=['longitude', 'latitude', 'spc_common','health'])

df.info()
d1995 = pd.DataFrame(df1995, columns=['longitude', 'latitude', 'spc_common', 'health'])

d1995.info()

d1995.head(20)

df.plot.scatter('longitude', 'latitude', figsize=(20,20))
plt.figure(figsize=(100,100))

sns.scatterplot(x='longitude', y='latitude', hue='health', data=df)
plt.figure(figsize=(50,50))

sns.scatterplot(x='longitude', y='latitude', hue='spc_common', data=df)
df['spc_common'].value_counts()



    
trees = ['green ash', 'honeylocust', 'Callery pear', 'pin oak', 'white ash']

df.head()
df2 = df[(df['spc_common'] == 'Norway maple')]

df2.plot.scatter('longitude', 'latitude', figsize=(20,20))
df2 = df[(df['spc_common'] == 'white ash')]

df2.plot.scatter('longitude', 'latitude', figsize=(20,20))
d1995.plot.scatter('longitude', 'latitude', figsize=(10,10))
d1995.describe()

nyc_min_lon = -74.05

nyc_max_lon = -73.75



nyc_min_lat = 40.63

nyc_max_lat = 40.85

        

for long in ['longitude', 'longitude']:

    d1995 = d1995[(d1995[long] > nyc_min_lon) & (d1995[long] < nyc_max_lon) ]



for lat in ['latitude', 'latitude']:

    d1995 = d1995[(d1995[lat] > nyc_min_lat) & (d1995[lat] < nyc_max_lat)]



d1995.describe()
d1995.plot.scatter('longitude', 'latitude', figsize=(20,20))
df.plot.scatter('longitude', 'latitude', figsize=(20,20))

import geopandas



nyc = geopandas.read_file(geopandas.datasets.get_path('nybb'))

ax = nyc.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')



for long in df['longitude']:

    nyc.plot


