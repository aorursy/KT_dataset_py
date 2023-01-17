# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import seaborn as sns
import matplotlib.pyplot as plt
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv',nrows=10000,parse_dates=["pickup_datetime"])
df.head()
df.pickup_datetime=pd.to_datetime(df.pickup_datetime)
df['hour'] = df.pickup_datetime.dt.hour
df['yday'] = df.pickup_datetime.dt.dayofyear
df['wday'] = df.pickup_datetime.dt.dayofweek
fig, ax = plt.subplots(figsize=(15,8))
sns.distplot(df[df['fare_amount']<80]['fare_amount'],ax=ax)
plt.title("fare amount distribution")
plt.show()
df = df.dropna(how = 'any', axis = 'rows')
fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(12,10))
plt.ylim(40.6, 40.85)
plt.xlim(-74.1,-73.7)
ax.scatter(df['pickup_longitude'],df['pickup_latitude'], s=0.5, alpha=1)
plt.show()

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import numpy as np
import pickle

fromclusters = KMeans(n_clusters=20, random_state=42).fit(df[['pickup_longitude','pickup_latitude']])
toclusters = KMeans(n_clusters=5, random_state=42).fit(df[['dropoff_longitude','dropoff_latitude']])

#fromclusters = DBSCAN().fit(X=df[['dropoff_longitude','dropoff_latitude']])
cx = [c[0] for c in fromclusters.cluster_centers_]
cy = [c[1] for c in fromclusters.cluster_centers_]
fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(12,10))
plt.ylim(40.6, 40.9)
plt.xlim(-74.1,-73.7)

df['cluster'] = fromclusters.fit_predict(df[['pickup_longitude','pickup_latitude']])
#df['dest_cluster'] = destkmeans.predict(df[['dropoff_longitude','dropoff_latitude']])
cm = plt.get_cmap('gist_rainbow')

colors = [cm(2.*i/20) for i in range(20)]
colored = [colors[k] for k in df['cluster']]

#plt.figure(figsize = (10,10))
ax.scatter(df.pickup_longitude,df.pickup_latitude,color=colored,s=0.04,alpha=1)
ax.scatter(cx,cy,color='Black',s=50,alpha=1)
plt.title('Taxi Pickup Clusters')
plt.show()
#plt.ylim(40.6, 40.9)

#ax.scatter(sdf['pickup_longitude'],sdf['pickup_latitude'], s=0.1, alpha=1)
#ax.scatter(cx,cy,s=70,color='Red')
df['hour'].unique()
hours =[]
for i in range(0,23):
    hours.append(df[df['hour']==i])


import math
math.floor(3/2)
import math
fig, ax = plt.subplots(ncols=2, nrows=12,figsize=(12,60))

for i in range(0,23):
    ax[math.floor(i/2)][i%2].set_ylim(40.6, 40.85)
    ax[math.floor(i/2)][i%2].set_xlim(-74.1,-73.7)
    ax[math.floor(i/2)][i%2].set_title(str(i))
    ax[math.floor(i/2)][i%2].scatter(hours[i]['pickup_longitude'],hours[i]['pickup_latitude'], s=0.5, alpha=1)
plt.show()
