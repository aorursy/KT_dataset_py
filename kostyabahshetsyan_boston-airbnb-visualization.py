import pandas as pd

import os

import numpy as np



import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

%matplotlib inline



from mpl_toolkits.mplot3d import Axes3D

from matplotlib.collections import PolyCollection
listings = pd.read_csv('../input/listings.csv')
listings.head()
# clean price data

listings.price = listings.price.apply(lambda x: x.split('.')[0]).replace('[^0-9]', '', regex=True).apply(lambda x: int(x)) 
fig = plt.figure(figsize=(25,25))



m = Basemap(projection='merc', llcrnrlat=42.23, urcrnrlat=42.4, llcrnrlon=-71.18, urcrnrlon=-70.99,)



m.drawcounties()



num_colors = 20

values = listings.price

cm = plt.get_cmap('coolwarm')

scheme = [cm(i / num_colors) for i in range(num_colors)]

bins = np.linspace(values.min(), values.max(), num_colors)

listings['bin'] = np.digitize(values, bins) - 1

cmap = mpl.colors.ListedColormap(scheme)



color = [scheme[listings[(listings.latitude==x)&(listings.longitude==y)]['bin'].values] 

             for x,y in zip(listings.latitude, listings.longitude)]



x,y = m(listings.longitude.values, listings.latitude.values)

scat = m.scatter(x,y, s = listings.price, color = color, cmap=cmap, alpha=0.8)





# Draw color legend.

                        #[left, top, width, height]

ax_legend = fig.add_axes([0.21, 0.12, 0.6, 0.02])

cb = mpl.colorbar.ColorbarBase(ax_legend, cmap=cmap, ticks=bins, boundaries=bins, orientation='horizontal')

cb.ax.set_xticklabels([str(round(i, 1)) for i in bins])





plt.show()
m = Basemap(llcrnrlon=-71.18,llcrnrlat=42.23,urcrnrlon=-70.99,urcrnrlat=42.4)



fig = plt.figure(figsize = [20, 15])

ax = Axes3D(fig)



ax.set_axis_off()

ax.azim = 270

ax.elev = 60

ax.dist = 6



ax.add_collection3d(m.drawcountries(linewidth=0.35, color = 'black'))

ax.add_collection3d(m.drawcounties())



polys = []

for polygon in m.landpolygons:

    polys.append(polygon.get_coords())



lc = PolyCollection(polys, edgecolor='black',

                    facecolor='steelblue', closed=True)

ax.add_collection3d(lc)



x,y = m(listings.longitude.values, listings.latitude.values)

ax.bar3d(x, y, np.zeros(len(x)), 0, 0, listings.price.values, color=color, alpha=0.7)



plt.show()


plt.figure(figsize = (12, 6))

sns.boxplot(x = 'neighbourhood_cleansed', y = 'price',  data = listings)

xt = plt.xticks(rotation=90)
sns.violinplot('neighbourhood_cleansed', 'price', data = listings)

xt = plt.xticks(rotation=90)
sns.factorplot('neighbourhood_cleansed', 'price', data = listings, color = 'm', \

               estimator = np.median, size = 4.5,  aspect=1.35)

xt = plt.xticks(rotation=90)
plt.figure(figsize=(10,10))

sns.heatmap(listings.groupby([

        'neighbourhood_cleansed', 'bedrooms']).price.mean().unstack(),annot=True, fmt=".0f")
plt.figure(figsize=(10,10))

sns.heatmap(listings.groupby([

        'city', 'bedrooms']).price.mean().unstack(),annot=True, fmt=".0f")
plt.figure(figsize=(10,10))

sns.heatmap(listings.groupby(['property_type', 'bedrooms']).price.mean().unstack(), annot=True, fmt=".0f")
plt.figure(figsize=(10,10))

sns.heatmap(listings.groupby(['beds', 'bedrooms']).price.mean().unstack(), annot=True, fmt=".0f")