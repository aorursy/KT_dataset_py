# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cluster import KMeans

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df.pickup_datetime=pd.to_datetime(df.pickup_datetime)

df['hour'] = df.pickup_datetime.dt.hour



kmeans = KMeans(n_clusters=4, random_state=0).fit(df[['hour']])

kmeans.cluster_centers_



df['hour_cluster'] = kmeans.predict(df[['hour']])
pd.crosstab(df.hour, df.hour_cluster)
import matplotlib.pyplot as plt



fig = plt.figure(figsize=(12,12))

ax = fig.add_subplot(221)

ax.hexbin(df['pickup_longitude'][df.hour_cluster==3],df['pickup_latitude'][df.hour_cluster==3], gridsize=1000, cmap='inferno')

plt.ylim([40.6, 40.9])

plt.xlim([-74.1, -73.7])

plt.title('night')



ax2 = fig.add_subplot(222)

ax2.hexbin(df['pickup_longitude'][df.hour_cluster==0][:171480],df['pickup_latitude'][df.hour_cluster==0][:171480], gridsize=1000, cmap='inferno')

plt.ylim([40.6, 40.9])

plt.xlim([-74.1, -73.7])

plt.title('morning')



ax1 = fig.add_subplot(223)

ax1.hexbin(df['pickup_longitude'][df.hour_cluster==2][:171480],df['pickup_latitude'][df.hour_cluster==2][:171480], gridsize=1000, cmap='inferno')

plt.ylim([40.6, 40.9])

plt.xlim([-74.1, -73.7])

plt.title('afternoon')



ax2 = fig.add_subplot(224)

ax2.hexbin(df['pickup_longitude'][df.hour_cluster==1][:171480],df['pickup_latitude'][df.hour_cluster==1][:171480], gridsize=1000, cmap='inferno')

plt.ylim([40.6, 40.9])

plt.xlim([-74.1, -73.7])

plt.title('evening')
from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt



map = Basemap(llcrnrlon=-74.1,llcrnrlat=40.6,urcrnrlon=-73.7,urcrnrlat=40.90, resolution = 'f')



map.drawmapboundary(fill_color='white')

map.drawcoastlines()



lons = [0] #needed to interpret the latitude and longitude by basemap plot

lats = [0]



x, y = map(lons, lats)



map.scatter(df['pickup_longitude'], df['pickup_latitude'], s=1, color='red')



plt.title('Pickup (red) plotted on a map')

plt.show()
%%time

cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height, x_range=x_range, y_range=y_range)

agg = cvs.points(df, 'dropoff_longitude', 'dropoff_latitude', ds.count('passenger_count'))

img = tr_fns.shade(agg, cmap=["white", 'darkblue'], how='linear')



img