#invite people for the Kaggle party

import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
# time to fix up some of the scrubbed data

# Data types for each feature

ufo_data = pd.read_csv('../input/scrubbed.csv', low_memory=False)

ufo_data['datetime'] = pd.to_datetime(ufo_data['datetime'], errors='coerce')

ufo_data.insert(1, 'year', ufo_data['datetime'].dt.year)

ufo_data.insert(2, 'month', ufo_data['datetime'].dt.month)

ufo_data.insert(3, 'day', ufo_data['datetime'].dt.day)

ufo_data['year'] = ufo_data['year'].fillna(0).astype(int)

ufo_data['month'] = ufo_data['month'].fillna(0).astype(int)

ufo_data['day'] = ufo_data['day'].fillna(0).astype(int)

ufo_data['city'] = ufo_data['city'].str.title()

ufo_data['state'] = ufo_data['state'].str.upper()

ufo_data['latitude'] = pd.to_numeric(ufo_data['latitude'], errors='coerce')

ufo_data = ufo_data.rename(columns={'longitude ':'longitude'})
#for fun, lets make a subset for Ontario.

ontario_ufo_data = ufo_data[ufo_data['state'].str.contains('ON') == True]

ontario_ufo_data


# az_ufo_years = az_ufo_data[az_ufo_data.year != 0]

# fl_ufo_years = fl_ufo_data[fl_ufo_data.year != 0]

# ny_ufo_years = ny_ufo_data[ny_ufo_data.year != 0]

ontario_ufo_years = ontario_ufo_data[ontario_ufo_data['year'] != 0]





# a = az_ufo_years['year'].groupby(az_ufo_years['year']).count()

# b = fl_ufo_years['year'].groupby(fl_ufo_years['year']).count()

# e = ny_ufo_years['year'].groupby(ny_ufo_years['year']).count()

f = ontario_ufo_years['year'].groupby(ontario_ufo_years['year']).count()



fig, ax = plt.subplots()

# ax.plot(a, label='AZ')

# ax.plot(b, label='FL')

# ax.plot(e, label='NY')

ax.plot(f, label='ON')



legend = ax.legend(loc='upper left', shadow=True)
ontario_ufo_data.describe()




plt.figure(figsize=(12,8))

ON = Basemap(projection='mill', llcrnrlat = 40, urcrnrlat = 54, llcrnrlon = -94, urcrnrlon = -72, \

             resolution = 'h')



# ON.drawcoastlines()

# ON.drawcountries()

ON.drawstates()

# ON.drawmapboundary(fill_color='aqua')

# ON.fillcontinents(color='grey',lake_color='aqua')

x, y = ON(list(ontario_ufo_data["longitude"].astype("float")), list(ontario_ufo_data["latitude"].astype(float)))

ON.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "green")



plt.title('UFO Sightings in Ontario')

ON.bluemarble()

plt.show()