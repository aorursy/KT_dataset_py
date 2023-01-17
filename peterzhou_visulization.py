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
listings = pd.read_csv('../input/listings_detail.csv')
listings.head()
# clean price data
listings.price = listings.price.apply(lambda x: x.split('.')[0]).replace('[^0-9]', '', regex=True).apply(lambda x: int(x)) 
#price distribution
#truncated at 5 in order to see a more meanningful plot
fig = plt.figure(figsize=(30,30))

m = Basemap(projection='merc', llcrnrlat=40.47, urcrnrlat=40.97, llcrnrlon=-74.41, urcrnrlon=-73.61,)

m.drawcounties()

num_colors = 20

cm = plt.get_cmap("YlOrRd")
scheme = [cm(i / num_colors) for i in range(num_colors)]

####preprocessing#####
values = listings.price.apply(lambda x: 0 if x <= 1 else np.log(x))
values.fillna(0)
bins = np.linspace(values.min(), values.max(), num_colors)
######################

listings['bin'] = np.digitize(values, bins) - 1
cmap = mpl.colors.ListedColormap(scheme)

color = [scheme[listings[(listings.latitude==x)&(listings.longitude==y)]['bin'].values[0]] 
             for x,y in zip(listings.latitude, listings.longitude)]

x,y = m(listings.longitude.values, listings.latitude.values)
scat = m.scatter(x,y, color = color, cmap=cmap, s=1, alpha=0.8)


# Draw color legend.
                        #[left, top, width, height]
ax_legend = fig.add_axes([0.21, 0.12, 0.6, 0.02])
cb = mpl.colorbar.ColorbarBase(ax_legend, cmap=cmap, ticks=bins, boundaries=bins, orientation='horizontal')
cb.ax.set_xticklabels([str(round(i, 1)) for i in bins])

plt.title("NYC airbnb listing log(price)")
plt.show()
#reviews per month
#truncated at 5 in order to see a more meanningful plot

fig = plt.figure(figsize=(30,30))

m = Basemap(projection='merc', llcrnrlat=40.47, urcrnrlat=40.97, llcrnrlon=-74.41, urcrnrlon=-73.61,)

m.drawcounties()

num_colors = 20

cm = plt.get_cmap('coolwarm')
scheme = [cm(i / num_colors) for i in range(num_colors)]

####preprocessing#####
values = listings.reviews_per_month
values = values.fillna(0)
bins = np.linspace(values.min(), 5, num_colors)
######################

listings['bin'] = np.digitize(values, bins) - 1
cmap = mpl.colors.ListedColormap(scheme)

color = [scheme[listings[(listings.latitude==x)&(listings.longitude==y)]['bin'].values[0]] 
             for x,y in zip(listings.latitude, listings.longitude)]

x,y = m(listings.longitude.values, listings.latitude.values)
scat = m.scatter(x,y, color = color, s=10*values, cmap=cmap, alpha=0.8)


# Draw color legend.
                        #[left, top, width, height]
ax_legend = fig.add_axes([0.21, 0.12, 0.6, 0.02])
cb = mpl.colorbar.ColorbarBase(ax_legend, cmap=cmap, ticks=bins, boundaries=bins, orientation='horizontal')
cb.ax.set_xticklabels([str(round(i, 1)) for i in bins])

plt.title("NYC airbnb listing popularity")
plt.show()