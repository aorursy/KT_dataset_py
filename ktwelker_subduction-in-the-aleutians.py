import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

from mpl_toolkits.basemap import Basemap

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')



from subprocess import check_output

from pandas import Series

print(check_output(["ls", "../input"]).decode("utf8"))





volcanoes = pd.read_csv("../input/volcanic-eruptions/database.csv")

earthquakes = pd.read_csv("../input/earthquake-database/database.csv")
#Defining Latitude and Longitude limits (change to your favorite part of the world!)

llcrnrlat = 45

urcrnrlat = 70

llcrnrlon = -180

urcrnrlon = -125



#Cutting out the data from other parts of the world.

alaska_volcanoes = volcanoes[(volcanoes.Longitude > llcrnrlon) & 

                             (volcanoes.Longitude < urcrnrlon) & 

                             (volcanoes.Latitude > llcrnrlat) & 

                             (volcanoes.Latitude < urcrnrlat)]



alaska_earthquakes = earthquakes[(earthquakes.Longitude > llcrnrlon) & 

                             (earthquakes.Longitude < urcrnrlon) & 

                             (earthquakes.Latitude > llcrnrlat) & 

                             (earthquakes.Latitude < urcrnrlat)]
m = Basemap(projection='mill',llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,lat_ts=20,resolution='l')

fig = plt.figure(figsize=(12,10))





longitudes_vol = alaska_volcanoes.Longitude.tolist()

latitudes_vol = alaska_volcanoes.Latitude.tolist()



longitudes_eq = alaska_earthquakes.Longitude.tolist()

latitudes_eq = alaska_earthquakes.Latitude.tolist()



x,y = m(longitudes_vol,latitudes_vol)

a,b= m(longitudes_eq,latitudes_eq)



plt.title("Alaskan Volcanoes (red) Earthquakes (green)")

m.plot(x, y, "o", markersize = 5, color = 'red')

m.plot(a, b, "o", markersize = 3, color = 'green')



m.drawcoastlines()

m.drawcountries()

m.fillcontinents(color='coral',lake_color='aqua')

m.drawmapboundary()

m.drawcountries()

plt.show()
llcrnrlat = 48

urcrnrlat = 61

llcrnrlon = -180

urcrnrlon = -150



#A closer look at the Aleutian Islands

aleutian_volcanoes = volcanoes[(volcanoes.Longitude > llcrnrlon) & 

                             (volcanoes.Longitude < urcrnrlon) & 

                             (volcanoes.Latitude > llcrnrlat) & 

                             (volcanoes.Latitude < urcrnrlat)]



aleutian_earthquakes = earthquakes[(earthquakes.Longitude > llcrnrlon) & 

                             (earthquakes.Longitude < urcrnrlon) & 

                             (earthquakes.Latitude > llcrnrlat) & 

                             (earthquakes.Latitude < urcrnrlat)]
m = Basemap(projection='mill',llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,lat_ts=20,resolution='l')

fig = plt.figure(figsize=(12,10))





longitudes_vol = aleutian_volcanoes.Longitude.tolist()

latitudes_vol = aleutian_volcanoes.Latitude.tolist()



longitudes_eq = aleutian_earthquakes.Longitude.tolist()

latitudes_eq = aleutian_earthquakes.Latitude.tolist()



x,y = m(longitudes_vol,latitudes_vol)

a,b= m(longitudes_eq,latitudes_eq)



plt.title("Aleutian Volcanoes (red) Earthquakes (green)")

m.plot(x, y, "o", markersize = 5, color = 'red')

m.plot(a, b, "o", markersize = 3, color = 'green')



m.drawcoastlines()

m.drawcountries()

m.fillcontinents(color='coral',lake_color='aqua')

m.drawmapboundary()

m.drawcountries()

plt.show()
from geopy.distance import great_circle
volcano_locations = aleutian_volcanoes[['Latitude', 'Longitude']]



def d_to_volcano(row):

    '''

    Given a row of the earthquakes DataFrame, returns the distance to the nearest aleutian volcano.

    

    There has got to be a better (faster) way to do this.

    '''

    min_dist = 1e4

    for i, c in volcano_locations.iterrows():

        b = great_circle((row['Latitude'], row['Longitude']), c.tolist()).kilometers

        if b < min_dist:

            min_dist = b

    return min_dist



aleutian_earthquakes['Nearest Volcano'] = aleutian_earthquakes.apply(d_to_volcano, axis=1)
plt.plot(aleutian_earthquakes['Nearest Volcano'], - aleutian_earthquakes.Depth, '.')

plt.xlabel('Distance to nearest volcano (km)')

plt.ylabel('Earthquake Depth (km)')

plt.title('Earthquake depth vs. distance to nearest volcano')

plt.show()
llcrnrlat = 48

urcrnrlat = 61

llcrnrlon = -180

urcrnrlon = -150



#Where are the deep earthquakes?



deep_earthquakes = aleutian_earthquakes[(aleutian_earthquakes.Longitude > llcrnrlon) & 

                             (aleutian_earthquakes.Longitude < urcrnrlon) & 

                             (aleutian_earthquakes.Latitude > llcrnrlat) & 

                             (aleutian_earthquakes.Latitude < urcrnrlat) & 

                                  (aleutian_earthquakes.Depth > 75)]



shallow_earthquakes = aleutian_earthquakes[(aleutian_earthquakes.Longitude > llcrnrlon) & 

                             (aleutian_earthquakes.Longitude < urcrnrlon) & 

                             (aleutian_earthquakes.Latitude > llcrnrlat) & 

                             (aleutian_earthquakes.Latitude < urcrnrlat) & 

                                  (aleutian_earthquakes.Depth < 75)]
m = Basemap(projection='mill',llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,lat_ts=20,resolution='l')

fig = plt.figure(figsize=(12,10))





longitudes_vol = aleutian_volcanoes.Longitude.tolist()

latitudes_vol = aleutian_volcanoes.Latitude.tolist()



longitudes_eq = deep_earthquakes.Longitude.tolist()

latitudes_eq = deep_earthquakes.Latitude.tolist()



longitudes_s_eq = shallow_earthquakes.Longitude.tolist()

latitudes_s_eq = shallow_earthquakes.Latitude.tolist()



x,y = m(longitudes_vol,latitudes_vol)

a,b= m(longitudes_eq,latitudes_eq)

s,t = m(longitudes_s_eq, latitudes_s_eq)



plt.title("Aleutian Volcanoes (red) Earthquakes (green = deep, yellow = shallow)")

m.plot(x, y, "o", markersize = 5, color = 'red')

m.plot(a, b, "o", markersize = 3, color = 'green')

m.plot(s, t, "o", markersize = 3, color = 'yellow')





m.drawcoastlines()

m.drawcountries()

m.fillcontinents(color='coral',lake_color='aqua')

m.drawmapboundary()

m.drawcountries()

plt.show()
from scipy.optimize import curve_fit
def y(x, m, b):

    return m * x + b



popt, pcov = curve_fit(y, deep_earthquakes['Nearest Volcano'].tolist(), deep_earthquakes['Depth'].tolist())

print(popt)
plt.plot(deep_earthquakes['Nearest Volcano'], - deep_earthquakes.Depth, 'g.')

dist = np.arange(0, 120, 2)

dep = y(dist, - popt[0], - popt[1])

plt.plot(dist, dep)

plt.ylabel('Earthquake Depth (km)')

plt.xlabel('Distance to nearest volcano (km)')

plt.show()
print('Subduction angle:', np.arctan(popt[0]) * 180 / np.pi, 'degrees')

print('Depth of Pacific plate under volcanos:', popt[1], 'km')