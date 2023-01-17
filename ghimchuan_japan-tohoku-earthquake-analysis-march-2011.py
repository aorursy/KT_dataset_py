# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Japan earthquakes 2001 - 2018.csv')
df['time'] = pd.to_datetime(df['time'])
sns.jointplot(x='depth',y='mag',data=df)
df2011March = df[(df['time'].dt.year == 2011) & (df['time'].dt.month == 3)]

sns.jointplot(x='depth',y='mag',data=df2011March)
group2011March = df2011March.groupby(by=[df2011March['time'].dt.day,'mag']).count()['depth'].unstack()
plt.figure(figsize=(12,6))

sns.heatmap(group2011March,cmap='seismic')
f, axes = plt.subplots(2, 1, figsize=(15,15))

sns.boxplot(x=df2011March['time'].dt.day,y='depth',data=df2011March, orient='v', ax=axes[0])

sns.boxplot(x=df2011March['time'].dt.day,y='mag',data=df2011March, orient='v', ax=axes[1])

plt.xlabel('Day')

fig.tight_layout()
sns.jointplot(x='mag',y='depth',data=df2011March[df2011March['time'].dt.day==11],kind='hex',cmap='seismic')
from mpl_toolkits.basemap import Basemap
fig = plt.figure(figsize=(20, 20))

ax = plt.subplot(1,1,1)

my_map = Basemap(resolution='l',

                 llcrnrlon=df['longitude'].min(), llcrnrlat=df['latitude'].min(),

                 urcrnrlon=df['longitude'].max(), urcrnrlat=df['latitude'].max())

my_map.bluemarble(alpha=0.42)

my_map.drawcoastlines(color='#555566', linewidth=1)

dfmag_gt7 = df2011March[df2011March['mag']>=7]

dtmag_57 = df2011March[(df2011March['mag']<7) & (df2011March['mag']>=5)]

dtmag_lt5 = df2011March[df2011March['mag']<5]

ax.scatter(dtmag_lt5['longitude'],dtmag_lt5['latitude'],10, lw=0.5,

                  marker='.', facecolors='green', zorder=10, label = 'mag < 5')

ax.scatter(dtmag_57['longitude'],dtmag_57['latitude'],10, lw=0.5,

                  marker='^', facecolors='blue', zorder=10, label = 'mag >=5 or < 7')

ax.scatter(dfmag_gt7['longitude'],dfmag_gt7['latitude'],100, lw=0.5,

                  marker='*', facecolors='red', zorder=10, label = 'mag >=7')

plt.legend()