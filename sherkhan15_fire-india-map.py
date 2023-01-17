import os

import pandas as pd

import numpy as np

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import matplotlib.pyplot as plt

import seaborn as sns

import contextily as ctx

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

import seaborn as sns

import contextily as ctx

from mpl_toolkits.basemap import Basemap
df1=pd.read_csv('/kaggle/input/indian-wildfire-nasa-dataset-8-years/fire_nrt_M6_107977.csv')
df1.head()
df1.dtypes
lat = df1['latitude'].values

lon = df1['longitude'].values

brg = df1['brightness'].values



fig = plt.figure(figsize = (10, 10))

m = Basemap(projection = 'lcc', resolution='c', lat_0 =21.7679, lon_0 = 78.8718,width=5E6, height=4E6)

m.shadedrelief()

m.drawcoastlines(color='gray')

m.drawcountries(color='gray')

m.drawstates(color='gray')



m.scatter(lon, lat, c= brg,latlon=True,cmap='Reds', alpha=0.6)

plt.colorbar(label=r'$Brightness$')
df1_night=df1.loc[df1['daynight'].isin(['N'])]

df1_day=df1.loc[df1['daynight'].isin(['D'])]
lat_d = df1_day['latitude'].values

lon_d = df1_day['longitude'].values

brg_d = df1_day['brightness'].values



lat_n = df1_night['latitude'].values

lon_n = df1_night['longitude'].values

brg_n = df1_night['brightness'].values



fig = plt.figure(figsize = (10, 10))

m = Basemap(projection = 'lcc', resolution='c', lat_0 =21.7679, lon_0 = 78.8718,width=5E6, height=4E6)

m.shadedrelief()

m.drawcoastlines(color='gray')

m.drawcountries(color='gray')

m.drawstates(color='gray')



m.scatter(lon_d, lat_d, c= brg_d, latlon=True,cmap='Reds', alpha=0.6)

plt.colorbar(label='Daytime  Brightness')



m.scatter(lon_n, lat_n, c= np.array(brg_n),latlon=True,cmap='Blues', alpha=0.6)

plt.colorbar(label='Nighttime  Brightness')
df1_hot=df1[df1['brightness']>450]
lat_hot=df1_hot['latitude'].values

lon_hot=df1_hot['longitude'].values

brg_hot=c=df1_hot['brightness'].values





fig = plt.figure(figsize = (10, 10))

m = Basemap(projection = 'lcc', resolution='c', lat_0 =21.7679, lon_0 = 78.8718,width=5E6, height=4E6)

m.shadedrelief()

m.drawcoastlines(color='gray')

m.drawcountries(color='gray')

m.drawstates(color='gray')



m.scatter(lon_hot,lat_hot,c=brg_hot, latlon=True,cmap='Reds', alpha=0.6)

plt.colorbar(label='Daytime  Brightness')
import matplotlib

from matplotlib.animation import FuncAnimation

from matplotlib import animation, rc



time=df1['acq_date'].values



#Putting basemap as a frame

fig = plt.figure(figsize=(10, 10))



m = Basemap(projection = 'lcc', resolution='c', lat_0 =21.7679, lon_0 = 78.8718,width=5E6, height=4E6)

m.shadedrelief()

m.drawcoastlines(color='gray')

m.drawcountries(color='gray')

m.drawstates(color='gray')



#Getting unique data values as we have multiple rows assoicated with each date

uniq_time=np.unique(time)



#showing the start date

date_text = plt.text(-170, 80, uniq_time[0],fontsize=15)



#very first data to show-brigtness data sets that were obatined on the first acquisition date

data=df1[df1['acq_date'].str.contains(uniq_time[0])]

cmap = plt.get_cmap('Reds')

xs, ys = data['longitude'].values, data['latitude'].values

scat=m.scatter(xs,ys,c=data['brightness'].values,cmap=cmap, latlon=True, alpha=0.6)

plt.colorbar(label='Fire Brightness')



#We will get numbers starting from 0 to the size of the dataframe spaced by "10" as it will take very long to generate animation for all data points.

#Basically we will look at the datasets with a 10-day interval.

empty_index=[]

for i in range(1,len(uniq_time),10):

    empty_index.append(i)    

    

def update(i):

    current_date = uniq_time[i]

    data=df1[df1['acq_date'].str.contains(uniq_time[i])]

    xs, ys = m(data['longitude'].values, data['latitude'].values)

    X=np.c_[xs,ys]

    scat.set_offsets(X)

    date_text.set_text(current_date)

    

ani = matplotlib.animation.FuncAnimation(fig, update, interval=50,frames=empty_index)



#trying to diplay animation with HTML

from IPython.display import HTML

import warnings

warnings.filterwarnings('ignore')



#Exporting the animation to show up correctly on Kaggle kernel. However, this creates an additional unwanted figure at the bottom.

#Let's ignore for this time



import io

import base64



filename = 'animation.gif'



ani.save('animation.gif', writer='imagemagick', fps=1)



video = io.open(filename, 'r+b').read()

encoded = base64.b64encode(video)

HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))