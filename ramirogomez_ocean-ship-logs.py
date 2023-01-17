# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


# Input data files are available in the "../input/" directory.
!ls ../input
df = pd.read_csv('../input/CLIWOC15.csv', usecols=['Lat3', 'Lon3', 'Nationality'])
df.dropna(axis=0, how='any', inplace=True)
df.shape
fig = plt.figure(figsize=(20, 14))
markersize = .5
markertype = '.'
markercolor = '#000000'
markeralpha = .4

m = Basemap(projection='mill')

# Avoid border around map.
m.drawmapboundary(fill_color='#ffffff', linewidth=.0)

# Convert locations to x/y coordinates and plot them as dots.
x, y = m(df.Lon3.values, df.Lat3.values)
m.scatter(x, y, markersize, marker=markertype, color=markercolor, alpha=markeralpha)
plt.show()