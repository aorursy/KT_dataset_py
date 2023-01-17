# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
%%time
data = pd.read_csv("../input/Accident_Information.csv")
data = data.dropna(subset=["Latitude", "Longitude"])
kde = KernelDensity(bandwidth=1.0, kernel = "gaussian")
kde.fit(data[['Latitude', 'Longitude']])
from mpl_toolkits.basemap import Basemap
from sklearn.datasets.species_distributions import construct_grids
from sklearn.datasets.species_distributions import fetch_species_distributions
%%time
m = Basemap(projection='cyl', resolution="c",
            llcrnrlat=min(data.Latitude), urcrnrlat=max(data.Latitude),
            llcrnrlon=min(data.Longitude), urcrnrlon=max(data.Longitude))
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='#FFEEDD')
m.drawcoastlines(color='gray', zorder=100000)
m.drawcountries(color='gray', zorder=1000000)
m.scatter(x=data.Longitude[1:100000], y=data.Latitude[1:100000], latlon=True, s=0.1)

help(m.scatter)
