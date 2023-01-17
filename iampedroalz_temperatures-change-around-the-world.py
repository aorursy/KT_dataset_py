# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

import seaborn as sns

import os 

%matplotlib inline

print(os.listdir("../input"))

# Reading the input file and storing it as dataframe #

file_name = "../input/global-historical-climatology-network/ghcn-m-v1.csv"

df = pd.read_csv(file_name, na_values=[-9999])

df.fillna(0, inplace=True)   # filling the NA values with 0

df.shape
co2 = pd.read_csv("../input/co2-emissions/co2.csv",skiprows = 3)

co2.head()
# helper function #

def getTemperature(year, month):

    temp_df = df.ix[df.year==year]

    temp_df = temp_df.ix[temp_df.month==month]

    return np.array(temp_df.iloc[:,3:]) / 100.



lons = np.array([-180, -175, -170, -165, -160, -155, -150, -145, -140, -135, -130, -125, -120, -115, -110, -105, -100, -95, -90, -85, -80, -75, -70, -65, -60, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180])

lats = np.array([90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0, -5, -10, -15, -20, -25, -30, -35, -40, -45, -50, -55, -60, -65, -70, -75, -80, -85, -90])

lons, lats = np.meshgrid(lons, lats)
# get the temperature for Jan 1994 #

temperature = getTemperature(1994, 1)

temperature[temperature<-4] = -4

temperature[temperature>4] = 4



# create figure, axes instances.

fig = plt.figure(figsize=(12,8))

ax = fig.add_axes([0.05,0.05,0.9,0.9])



# create a base map with 

m = Basemap(projection='gall',

              llcrnrlon = -180,              # lower-left corner longitude

              llcrnrlat = -90,               # lower-left corner latitude

              urcrnrlon = 180,               # upper-right corner longitude

              urcrnrlat = 90,               # upper-right corner latitude

              resolution = 'l',

              area_thresh = 1000000.0

              )

m.drawcoastlines()

m.drawcountries()



# plot sst, then ice with pcolor

im = m.pcolormesh(lons,lats,temperature,shading='flat',cmap=plt.cm.seismic,latlon=True)



# add colorbar

cb = m.colorbar(im,"bottom", size="5%", pad="5%")

# add a title.

ax.set_title('Temperature anomaly in Jan 1994')

plt.show()
import matplotlib.animation as animation



years = range(1990, 2017, 1)

# get the temperature for Jan 1880#

temperature = getTemperature(1990, 7)

temperature[temperature<-5] = -5

temperature[temperature>5] = 5

# create figure, axes instances.

fig = plt.figure(figsize=(12,8))

ax = fig.add_axes([0.05,0.05,0.9,0.9])

# create a base map with 

m = Basemap(projection='gall',

              llcrnrlon = -180,              # lower-left corner longitude

              llcrnrlat = -60,               # lower-left corner latitude

              urcrnrlon = 180,               # upper-right corner longitude

              urcrnrlat = 90,               # upper-right corner latitude

              resolution = 'l',

              area_thresh = 1000000.0

              )

m.drawcoastlines()

m.drawcountries()

# plot sst, then ice with pcolor

im = m.pcolormesh(lons,lats,temperature,shading='flat',cmap=plt.cm.seismic,latlon=True)

# add colorbar

cb = m.colorbar(im,"bottom", size="5%", pad="5%")

# add a title.

ax.set_title('Temperature anomaly in July 1880')

ax.set_xlabel("Temperature differences (ºC)")





def updatefig(ind):

    year = years[ind]

    temperature = getTemperature(year, 7)

    temperature[temperature<-5] = -5

    temperature[temperature>5] = 5

    m.pcolormesh(lons,lats,temperature,shading='flat',cmap=plt.cm.seismic,latlon=True)

    ax.set_title('Temperature anomaly in July '+str(year))

    ax.set_xlabel("Temperature differences (ºC)")

    return im,



ani = animation.FuncAnimation(fig, updatefig, frames=len(years))

ani.save('lb.gif', fps=0.33, writer='imagemagick')
co2.info()