# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns 

from mpl_toolkits.basemap import Basemap

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/az.csv')

print(data.columns)

df=data.loc[data['POSTCODE'] == '85027']

print(df.head())

df_events_sample = df.sample(100)

plt.figure(1, figsize=(12,6))

# Mercator of World



lon_min, lon_max =-130,-60

lat_min, lat_max = 25,50



m1 = Basemap(projection='merc',

             llcrnrlat=lat_min,

             urcrnrlat=lat_max,

             llcrnrlon=lon_min,

             urcrnrlon=lon_max,

             lat_ts=35,

             resolution='c')



m1.fillcontinents(color='#191919',lake_color='#000000') # dark grey land, black lakes

m1.drawmapboundary(fill_color='#000000')                # black background

m1.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders



mxy = m1(df_events_sample["LON"].tolist(), df_events_sample["LAT"].tolist())

m1.scatter(mxy[0], mxy[1], s=3, c="#1292db", lw=0, alpha=1, zorder=5)

plt.title("Geo Tagging Phoenix")

plt.show()