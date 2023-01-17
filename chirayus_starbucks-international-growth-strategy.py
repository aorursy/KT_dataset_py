# We will mainly explore through visualizations

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

import matplotlib.patches as mpatches



%matplotlib inline
# Reading-in the data

starbucks = pd.read_csv('../input/directory.csv')

starbucks.columns
starbucks.shape[0]
starbucks.isnull().sum()
starbucks['City'].value_counts().head(10)
starbucks['Ownership Type'].value_counts()
sns.set(style="whitegrid", context="talk")

sns.countplot(x='Ownership Type', data=starbucks, palette="BuGn_d")
f, ax = plt.subplots(figsize=(18,12))

map = Basemap(projection='mill', 

              llcrnrlat = -80,

              urcrnrlat = 80,

              llcrnrlon = -180,

              urcrnrlon = 180,

              resolution = 'h')



# Plot coastlines and country borders.

map.drawcoastlines()

map.drawcountries()



# Fill continents and color wet areas.

map.drawmapboundary(fill_color='lightskyblue')

map.fillcontinents(color='beige',

                   lake_color='lightskyblue')



# Color coding the store by ownership type.

markerCode = {'Company Owned': 'green', 

              'Licensed': 'yellow',

              'Franchise': 'tomato',

              'Joint Venture': 'mediumpurple'}

starbucks['ownerColorCode'] = starbucks['Ownership Type'].map(markerCode)





lons, lats = map(list(starbucks["Longitude"].astype(float)),

                 list(starbucks["Latitude"].astype(float)))

colors = list(starbucks['ownerColorCode'])



# Loop through each location to plot the individual stores.

for lon, lat, owner in zip(lons, lats, colors):

    x, y = lon, lat

    marker_string = owner

    map.plot(x, y, 'bo', alpha = 0.6, color = marker_string)



# Drop the color-code column after use.

starbucks.drop(['ownerColorCode'], axis=1, inplace=True)



# Hack together a legend

legend_handles = [mpatches.Patch(color = color_code, label = owner) for owner, color_code in markerCode.items()]

ax.legend(loc='lower left',

          handles = legend_handles)



plt.title('Starbucks Stores and Ownership')

plt.show()
# A long list of countries I think are in Europe.

europe =['BE', 'EL', 'LT', 'PT', 'BG', 'ES', 'LU', 'RO', 'CZ', 'FR', 'HU', 'SI', 'DK', 'GB', 'GR', 'HR', 'MT', 'SK', 'DE', 'IT', 'NL', 'FI', 'EE', 'CY', 'AT', 'SE', 'IE', 'LV', 'PL', 'UK', 'IS', 'NO', 'LI', 'CH', 'TR']

european_stores = starbucks[starbucks['Country'].isin(europe)]



f, ax = plt.subplots(figsize=(18,12))



# Where I think Europe is...

map = Basemap(projection='mill', 

              llcrnrlat = 30,

              urcrnrlat = 70,

              llcrnrlon = -20,

              urcrnrlon = 40,

              resolution = 'l')



# Plot coastlines and country borders.

map.drawcoastlines()

map.drawcountries()



# Fill continents and color wet areas.

map.drawmapboundary(fill_color='lightskyblue')

map.fillcontinents(color='beige',

                   lake_color='lightskyblue')



# Color coding the store by ownership type.

markerCode = {'Company Owned': 'green', 

              'Licensed': 'yellow',

              'Franchise': 'tomato',

              'Joint Venture': 'mediumpurple'}

european_stores['ownerColorCode'] = european_stores['Ownership Type'].map(markerCode)





lons, lats = map(list(european_stores["Longitude"].astype(float)),

                 list(european_stores["Latitude"].astype(float)))

colors = list(european_stores['ownerColorCode'])



# Loop through each location to plot the individual stores.

for lon, lat, owner in zip(lons, lats, colors):

    x, y = lon, lat

    marker_string = owner

    map.plot(x, y, 'bo', alpha = 0.6, color = marker_string)



# Drop the color-code column after use.

european_stores.drop(['ownerColorCode'], axis=1, inplace=True)



# Hack together a legend

legend_handles = [mpatches.Patch(color = color_code, label = owner) for owner, color_code in markerCode.items()]

ax.legend(loc='lower left',

          handles = legend_handles)



plt.title('Starbucks Stores and Ownership in Europe')

plt.show()