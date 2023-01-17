import pandas as pd

import numpy as np

import os

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.basemap import Basemap

from matplotlib.collections import PolyCollection

%matplotlib inline
listings = pd.read_csv('/kaggle/input/boston/listings.csv')
listings.head()
listings.columns
listings.price = listings.price.apply(lambda x:x.split('.')[0]).replace('[^0-9]','', regex=True).apply(lambda x:int(x))
fig = plt.figure(figsize=(25,25))

m = Basemap(projection='merc', llcrnrlat=42.23, urcrnrlat=42.4, llcrnrlon=71.18, urcrnrlon=70.99,)

m.drawcounties()



num_colors = 20

values = listings.price

cm = plt.get_camp('coolwarm')

scheme = [cm(i/num_colors) for i in range(num_colors)]

bins = np.linspace(values.min(), values.max(), num_colors)

listings['bin'] = np.digitize(values, bins) - 1

cmap = mpl.colors.ListedColormap(scheme)



color = [schema[listings[(listings.latitude==x)&(listings.longitude==y)]['bin'].values]

        for x,y in zip(listings.latitude, listings.longitude)]

x,y = m(listings.longitude.values, listings.latitude.values)

scat = m.scatter(x,y, s=listings.price, color = color, cmap = cmap, alpha=0.8)



# Draw color legend, [left, top, width, height]

ax_legend = fig.add_axes([0.21, 0.12, 0.6, 0.02])

cb = mpl.colorbar.ColorbarBase(ax_legend, cmap=cmap, ticks=bins, boundaries=bins, orientation='horizontal')

cb.ax.set_xticklabels([str(round(i, 1)) for i in bins])



plt.show()
plt.figure(figsize=(12,6))

sns.boxplot(x='neighbourhood_cleansed', y='price', data = listings)

xt = plt.xticks(rotation=90)
sns.violinplot('neighbourhood_cleansed', 'price', data = listings)

xt = plt.xticks(rotation = 90)
sns.factorplot('neighbourhood_cleansed', 'price', data = listings, color = 'm', estimator = np.median, size = 4.5, aspect = 1.35)

xt. plt.xticks(rotation = 90)
plt.figure(figsize=(10,10))

sns.heatmap(listings.groupby(['neighbourhood_cleansed', 'bedrooms']).price.mean().unstack(),annot=True, fmt='.0f')
plt.figure(figsize=(10,10))

sns.heatmap(listings.groupby(['city', 'bedrooms']).price.mean().unstack(),annot=True, fmt='.0f')
plt.figure(figsize=(10,10))

sns.heatmap(listings.groupby(['property_type','bedrooms']).price.mean().unstack(), annot=True, fmt='.0f')
plt.figure(figsize=(10,10))

sns.heatmap(listings.groupby(['beds','bedrooms']).price.mean().unstack(), annot=True, fmt='.0f')