# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from mpl_toolkits.basemap import Basemap

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
kiva_loans_data = pd.read_csv("../input/kiva_loans.csv")
kiva_mpi_locations_data = pd.read_csv("../input/kiva_mpi_region_locations.csv")
loan_theme_ids_data = pd.read_csv("../input/loan_theme_ids.csv")
loan_themes_by_region_data = pd.read_csv("../input/loan_themes_by_region.csv")
# Find total funding amounts per sector in each region
region_totals = kiva_loans_data.groupby(['country', 'region', 'sector']).sum().copy().reset_index()

# Join region totals dataframe with locations and filter only data we want
region_totals = region_totals.merge(kiva_mpi_locations_data, on=['country', 'region'])

# Finally, select only the columns relevant to our analysis
region_totals = region_totals[['country', 'region', 'sector', 'funded_amount', 'lon', 'lat']]
region_totals.head()
# Scale down the regional funding totals to obtain reasonable values to plot on a map
# Each regional total will be plotted as a bubble. The bubble for the largest funding total will have a size of '50'
scaled_funding = region_totals['funded_amount'] / region_totals['funded_amount'].max()
region_totals.loc[:,'funded_scaled'] = scaled_funding.values

# Sort the data by total regional funding from greatest to least.
# This is to make sure that smaller bubbles aren't obscured by larger ones by plotting the largest first then the smallest over them.
region_totals = region_totals.sort_values(['funded_scaled'], ascending=[0])
region_totals.shape
# There are too many sectors to plot all of them:
print(region_totals['sector'].unique())

#Let's choose a few of them to plot and specify a color for each
sectors = ['Food', 'Retail', 'Agriculture', 'Education', 'Clothing']
sector_totals = region_totals.loc[region_totals['sector'].isin(sectors)]
print(sector_totals.shape)
# Finally, plot the map using Basemap
plt.figure(figsize=(24, 8))
m = Basemap(projection='robin', resolution='c', lat_0=0, lon_0=-0,)
m.drawcoastlines()
m.drawcountries()

x, y = m(sector_totals['lon'].values, sector_totals['lat'].values)
l = [sectors.index(s) for s in sector_totals['sector'].values]
colormap = cm.Set1
plt.scatter(x, y, c=colormap(l), label=l, s=1E4*sector_totals['funded_scaled'].values)

legend_elements = []
for i in range(0, len(sectors)):
    legend_elements.append(Line2D([0], [0], 
                                  marker='o', 
                                  label=sectors[i],
                                  color='w',
                                  markerfacecolor=colormap(i), 
                                  markersize=10))

plt.legend(handles=legend_elements, loc='upper left')
"""
On first glance, the map has a couple of nice features. It gives a good overview of not only how a sector's funding is disbursed
throughout the world, but also how a region's funding breaks down by sector. That being said, there are 324 bubbles on the map, 
so most of them are too small for us to get a good look at them. For further investigation, it might be interesting to zoom in 
on an area of interest to get a better view.

Below is a zoomed-in plot of Central America using the same dataset as above. Note that the basemap projection has been switched 
to mercator, and the markersize has been increased dramatically.
"""
fig = plt.figure(figsize=(24, 8))
m = Basemap(projection='merc', lat_0=14, lon_0=-88,
    resolution = 'l', area_thresh = 10,
    llcrnrlon=-95, llcrnrlat=11,
    urcrnrlon=-81, urcrnrlat=17)
m.drawcoastlines()
m.drawcountries()

x, y = m(sector_totals['lon'].values, sector_totals['lat'].values)
plt.scatter(x, y, c=colormap(l), label=l, s=5E4*sector_totals['funded_scaled'].values)
plt.legend(handles=legend_elements, loc='upper left')