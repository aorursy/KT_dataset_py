my_map.drawcoastLines()# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
weather_df = pd.read_csv("/kaggle/input/wsht7.csv")
weather_df.shape
weather_df.head()
weather_df.dropna(subset=['Tm', 'Tx', 'Tn'], inplace=True)

print(weather_df.shape)
from mpl_toolkits.basemap import Basemap

import matplotlib

from PIL import Image

import matplotlib.pyplot as plt

from pylab import rcParams

%matplotlib inline

rcParams['figure.figsize'] = (14,10)
llon=-140

ulon=-50

llat=-40

ulat=75
weather_df=weather_df[(weather_df['Long'] > llon) & (weather_df['Long'] < ulon) & (weather_df['Lat'] > llat) & (weather_df['Lat'] < ulat)]
my_map = Basemap(projection="merc", resolution = 'l', area_thresh=1000.0, llcrnrlon=llon, llcrnlat=llat, urcrnrlon=ulon, urcrnrlat=ulat)

my_map.drawcoastLines()

my_map.drawcountries()

my_map.drawlsmask(land_color='orange', ocean_color="skyblue")

my_map.bluemarble()

xs, ys = my_map(np.asarray(weather_df.Long), np.asarray(weather_df.Lat))

weather_df['xm'] = xs.tolist()

weather_df['ym'] = ys.tolist()

from sklearn.cluster import DBSCAN

import sklearn.utils

from sklearn.preprocessing import StandardScaler

weather_df_clus_tmp = weather_df[['Tm', 'Tx', 'Tn', 'xm', 'ym']]

weather_df_clus_temp = StandardScaler().fit_transform(weather_df_clus_temp)

db = DBSCAN(eps=0.3, min_samples=10).fit(weather_df_clus_temp)

labels = db.labels_

print(labels[500:560])

weather_df["Clus_Db"]=labels

realClusterNum = len(set(labels))-(l if -l in labels else 0)

clusterNum = len(set(labels))
set(labels)