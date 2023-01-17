from pylab import rcParams



%matplotlib inline

rcParams['figure.figsize'] = (10,8)
import numpy as np

from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

import csv

import seaborn as sns

sns.set(color_codes=True)
lats, lons, magnitudes = [], [], []
def convert_lat(lat_str):

    return float(lat_str[:-1]) * (1.0 if 'N' in lat_str[-1] else -1.0)

def convert_lon(lon_str):

    return float(lon_str[:-1]) * (1.0 if 'E' in lon_str[-1] else -1.0)
filename = '../input/cneos_fireball_data.csv'

with open(filename) as f:

    # Create a csv reader object.

    reader = csv.reader(f)

    

    # Ignore the header row.

    next(reader)

    

    # Store the latitudes and longitudes in the appropriate lists.

    for row in reader:

        lat = row[1]

        lon = row[2]

        if lat:

            lats.append(convert_lat(lat))

            lons.append(convert_lon(lon))

            magnitudes.append(float(row[8]))
impact_map = Basemap(projection='robin', 

                     lat_0=0, 

                     lon_0=-100,

                     resolution='l', 

                     area_thresh=1000.0)

 

impact_map.drawcoastlines()

impact_map.drawcountries()

impact_map.fillcontinents(color='coral')

impact_map.drawmapboundary()



impact_map.drawmeridians(np.arange(0, 360, 30))

impact_map.drawparallels(np.arange(-90, 90, 30))



x,y = impact_map(lons, lats)

impact_map.plot(x, y, 'ko', markersize=10)
sns.distplot(magnitudes);
scaled_mags = [i/1e12 for i in magnitudes]