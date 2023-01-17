import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 





df = pd.read_csv('/kaggle/input/tree-census/new_york_tree_census_2015.csv')

df2005 = pd.read_csv('/kaggle/input/tree-census/new_york_tree_census_2005.csv')

d1995 = pd.read_csv('/kaggle/input/tree-census/new_york_tree_census_1995.csv')
df.head()
df.describe()
import seaborn as sns

import matplotlib.pyplot as plt 
plt.figure(figsize=(100,100))

sns.scatterplot(x='longitude', y='latitude', hue = 'health', data=df)
nyc_min_lon = -74.05

nyc_max_lon = -73.75



nyc_min_lat = 40.63

nyc_max_lat = 40.85

        

for long in ['longitude', 'longitude']:

    d1995 = d1995[(d1995[long] > nyc_min_lon) & (d1995[long] < nyc_max_lon) ]



for lat in ['latitude', 'latitude']:

    d1995 = d1995[(d1995[lat] > nyc_min_lat) & (d1995[lat] < nyc_max_lat)]



d1995.describe()
plt.figure(figsize=(100,100))

sns.scatterplot(x='longitude', y='latitude', hue = 'status', data=d1995)
d1995['spc_common'].value_counts()

d1995.head()
df['spc_common'].value_counts()
trees = ['London planetree', 'honeylocust', 'Callery pear', 'pin oak', 'Norway maple']
# Get top 5 trees to map

    # Scatter Plot

def plot_lat_lon(df, color = None):

    plt.figure(figsize=(100,100))

    sns.scatterplot(x='longitude', y='latitude', hue = color, data=df)

        

#plot_lat_lon(df,'health')
x_diff = df[ (df['spc_common'] == 'honeylocust') | (df['spc_common'] == 'pin oak') | (df['spc_common'] == 'pin oak') ]

plot_lat_lon(x_diff,'spc_common')