# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/ghiajmer"))

# Any results you write to the current directory are saved as output.
import sys,getopt
#from netCDF4 import Dataset  # use scipy instead
from scipy.io import netcdf #### <--- This is the library to import.
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta
#import coltbls as coltbls
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid import make_axes_locatable
import matplotlib.axes as maxes

# Reading the File
nc = netcdf.netcdf_file("../input/pmenergy/after_correcting_bias0-24.nc",'r')
#Look at the variables available
print(nc.variables)

#Look at the dimensions
nc.dimensions
nc
nc.dimensions
nc.variables
nc.variables['BCSMASS'][:,:,:]
ghiajmr=pd.read_csv("../input/ghiajmer/Ajmeer_2016.csv")
ghi=np.array(ghiajmr.V4)
ghid=ghi.reshape(366,1440)
d
lats = nc.variables['lat'][:]  # extract/copy the data
print(lats)
lons = nc.variables['lon'][:]
print(lons)
T2MAjm16=T2Majm[140208:148992].reshape(366,24)
print(T2MAjm16)
from numpy import savetxt
lats = nc.variables['lat'][:]  # extract/copy the data
lons = nc.variables['lon'][:]
time = nc.variables['time'][:]
T2M = nc.variables['BCSMASS'][:,:,:]
type(T2M)
T2M.shape
T2M1 = T2M[0,:,:]
# Near Kolkata Lattitude 22.5 and longitude 88.125
# Ajmer 26.4499° N, 74.6399° E lat 43
T2M2kol = T2M[:,32,36]
T2M2Del = T2M[:,14,47]
T2Majm = T2M[:,10,43]
T2MAjm16=T2Majm[140208:148992].reshape(366,24)
T2MAjm16
T2MAjm16.shape
from numpy import savetxt
savetxt('data.csv', T2MAjm16, delimiter=',')
t=T2Majm[140208:148992]
savetxt('data1.csv', t, delimiter=',')
t
# Analyzing T2M2
import matplotlib.pyplot as plt
yr2000= T2M2kol[:8784]
yr2000D= T2M2Del[:8784]

d1= T2M2kol[:24]
d2= T2M2kol[24:48]
d3= T2M2kol[48:72]
d4= T2M2kol[72:96]
d5= T2M2kol[96:120]


t1= pd.Series(d1)
t2= pd.Series(d2)
t3= pd.Series(d3)
t4= pd.Series(d4)
t5= pd.Series(d5)

plt.figure(figsize=(10,5))
ax = plt.subplot(111)
plt.xlabel('Hours')
plt.ylabel('PM2.5')
plt.title('Frist 5 Day wise PM')
ax.plot(t1, label='D1')
ax.plot(t2, label='D2')
ax.plot(t3, label='D3')
ax.plot(t4, label='D4')
ax.plot(t5, label='D5')

ax.legend()
plt.grid(True)
plt.show()
T2M2kol[:]
# Hourly  Average of first year
T2M2d=yr2000.reshape(366,24)
T2M2de=yr2000D.reshape(366,24)

ha= np.mean(T2M2d, axis=0)
ha1= np.mean(T2M2de, axis=0)
s1 = pd.Series(ha)
s2 = pd.Series(ha1)

plt.figure(figsize=(10,5))
ax = plt.subplot(111)
plt.xlabel('Hours')
plt.ylabel('PM2.5')
plt.title('Delhi and Kolkata hour wise pollution in 2000')
ax.plot(s1, label='Kolkata')
ax.plot(s2, label='Delhi')

ax.legend()
plt.grid(True)
plt.show()
da= np.mean(T2M2d[:364,:], axis=1)
da1= np.mean(T2M2de[:364,:], axis=1)

# Reshaping at a weekly level
daw=da.reshape(52,7)
daw1=da1.reshape(52,7)

qa= np.mean(daw, axis=0)
qa1= np.mean(daw1, axis=0)
qa
qa1
s1 = pd.Series(qa)
s2 = pd.Series(qa1)
plt.figure(figsize=(10,5))
ax = plt.subplot(111)
plt.title('Weekday wise Average of the year 2000 in Kolkata and Delhi')
plt.xlabel('Days')
plt.ylabel('PM2.5 levels')
ax.plot(s1, label='Kolkata')
ax.plot(s2, label='Delhi')

ax.legend()
plt.grid(True)
plt.show()
# Daily  Average of first year
qaw= np.mean(daw, axis=1)
qaw1= np.mean(daw1, axis=1)

s1 = pd.Series(qaw)
s2 = pd.Series(qaw1)

plt.figure(figsize=(10,5))
ax = plt.subplot(111)
plt.title('Week wise Average of the year 2000 near Kolkata and Delhi')
plt.xlabel('Weeks')
plt.ylabel('PM2.5 levels')
ax.plot(s1, label='Kolkata')
ax.plot(s2, label='Delhi')

ax.legend()
plt.grid(True)
plt.show()

#========= Start Plotting Data ===============================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap

map = Basemap(resolution='l', projection='eck4', lat_0=0, lon_0=0)

lon, lat = np.meshgrid(lons, lats)
xi, yi = map(lon, lat)

# Plot Data
cs = map.pcolor(xi,yi,np.squeeze(T2M1), vmin=np.min(T2M1), vmax=np.max(T2M1), cmap=cm.jet)
cs.set_edgecolor('face')

# Add Grid Lines
map.drawparallels(np.arange(-90., 90., 15.), labels=[1,0,0,0], fontsize=5)
map.drawmeridians(np.arange(-180., 180., 30.), labels=[0,0,0,1], fontsize=4)

# Add Coastlines, States, and Country Boundaries
map.drawcoastlines()
map.drawstates()
map.drawcountries()

# Add Colorbar
cbar = map.colorbar(cs, location='bottom', pad="10%")
cbar.set_label('K')
cbar.ax.tick_params(labelsize=10)
#print(T2M[:1,:1,:1])

print(lats)
yr2016= T2M2kol[140208:148992]
yr2012=T2M2kol[105148:113932]
yr2008=T2M2kol[70088:78872]
yr2004=T2M2kol[35028:43812]
T2M2d1=yr2016.reshape(366,24)
T2M2d2=yr2012.reshape(366,24)
T2M2d3=yr2008.reshape(366,24)
T2M2d4=yr2004.reshape(366,24)
ha1= np.mean(T2M2d1, axis=0)
ha2= np.mean(T2M2d2, axis=0)
ha3= np.mean(T2M2d3, axis=0)
ha4= np.mean(T2M2d4, axis=0)
d = {'yr2000' : pd.Series(ha),'yr2004' : pd.Series(ha4),'yr2008' : pd.Series(ha3),'yr2012' : pd.Series(ha2),'yr2016' : pd.Series(ha1)} 
  
# creates Dataframe. 
plt.figure(figsize=(10,5))
df = pd.DataFrame(d) 
ax = plt.gca()
plt.title('Hourly Average of PM2.5')
plt.xlabel('Hours')
plt.ylabel('PM2.5 levels')
df.plot(kind='line',y='yr2000',ax=ax)
df.plot(kind='line',y='yr2004', color='red', ax=ax)
df.plot(kind='line',y='yr2008', color='blue', ax=ax)
df.plot(kind='line',y='yr2012', color='orange', ax=ax)
df.plot(kind='line',y='yr2016', color='green', ax=ax)

ax.legend()
plt.grid(True)
plt.show()


# Special Days
yr2017D= T2M2kol[148992:157752]
yr2017D1= T2M2Del[148992:157752]
T2M2de=yr2017D.reshape(365,24)
T2M2de1=yr2017D1.reshape(365,24)

da1= np.mean(T2M2de[:364,:], axis=1)
da2= np.mean(T2M2de1[:364,:], axis=1)

daw1=da1.reshape(52,7)
daw2=da2.reshape(52,7)

qaw1= np.mean(daw1, axis=1)
qaw2= np.mean(daw2, axis=1)

s1 = pd.Series(qaw1)
s2 = pd.Series(qaw2)

plt.figure(figsize=(10,5))
ax = plt.subplot(111)
plt.title('Week wise Average of the year 2017 Delhi')
plt.xlabel('Weeks')
plt.ylabel('PM2.5 levels')
ax.plot(s1, label='Kolkata')
ax.plot(s2, label='Delhi')

ax.legend()
plt.grid(True)
plt.show()
T2M2Del.shape
T2M2DelF=T2M2Del.reshape(18,8764)
T2M2KolF=T2M2kol.reshape(18,8764)
T2M2DelFs=T2M2DelF.mean(axis=1)
T2M2KolFs=T2M2KolF.mean(axis=1)
s1 = pd.Series(T2M2DelFs)
s2 = pd.Series(T2M2KolFs)

plt.figure(figsize=(10,5))
ax = plt.subplot(111)
plt.title('Fortnightly Average from year 2000 to 2017 near Kolkata and Delhi')
plt.xlabel('Fortnight')
plt.ylabel('PM2.5 levels')
ax.plot(s1, label='Delhi')
ax.plot(s2, label='Kolkata')

ax.legend()
plt.grid(True)
plt.show()
s1 = pd.Series(ghid.mean(axis=1))
s2 = pd.Series(T2MAjm16.mean(axis=1))
ghid.shape
plt.figure(figsize=(10,5))
ax = plt.subplot(111)
plt.title('Fortnightly Average from year 2000 to 2017 near Kolkata and Delhi')
plt.xlabel('Fortnight')
plt.ylabel('PM2.5 levels')
ax.plot(s1, label='Delhi')
ax.plot(s2, label='Kolkata')

ax.legend()
plt.grid(True)
plt.show()
