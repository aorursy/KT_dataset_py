# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        #print(os.path.join(dirname, filename))

        continue



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
!ls ../input/birdsong-recognition
train = pd.read_csv('/kaggle/input/birdsong-recognition/train.csv')

test = pd.read_csv('/kaggle/input/birdsong-recognition/test.csv')

sub = pd.read_csv('/kaggle/input/birdsong-recognition/sample_submission.csv')



print("Shape train: ", train.shape)

print("Shape  test: ", test.shape)

print("Shape   sub: ", sub.shape)
train.head(3)
train.columns
df = train.copy()

print("Long min:", df.longitude.min())

print("Long min:", df.longitude.max())
fig, axs = plt.subplots(2,4, figsize=(23,12))



df['rating'].value_counts().plot(kind='bar', legend=True, ax=axs[0,0])

df['playback_used'].value_counts().plot(kind='bar', legend=True, ax=axs[0,1])

df['speed'].value_counts().plot(kind='bar', legend=True, ax=axs[0,2])

df['channels'].value_counts().plot(kind='bar', legend=True, ax=axs[0,3])



df['pitch'].value_counts().plot(kind='bar', legend=True, ax=axs[1,0])

df['length'].value_counts().plot(kind='bar', legend=True, ax=axs[1,1])

df['bird_seen'].value_counts().plot(kind='bar', legend=True, ax=axs[1,2])

df['license'].value_counts().plot(kind='bar', legend=True, ax=axs[1,3])



plt.savefig('data_exploration.png',dpi=300)



plt.show()
df['ebird_code'].unique(), len(df['ebird_code'].unique())
list_dirs = []

for dirs in next(os.walk('../input/birdsong-recognition/train_audio/'))[1]:

    list_dirs.append(dirs)

#list_dirs



array_dirs = np.array(list_dirs)



print('List:\n', list_dirs,'\n')

print('Total: {} directores'.format(len(array_dirs)))
#!ls ../input/birdsong-recognition/train_audio/

#!ls ../input/birdsong-recognition/train_audio/astfly/



# For sound within notebook

import IPython.display as ipd  

ipd.Audio('../input/birdsong-recognition/train_audio/astfly/XC109920.mp3')
fig = plt.figure(figsize=(12,6))



df['duration'].hist(bins=360)



plt.xlim(-10,600)
dflong =  df[df['duration'] > 600]

dflong['duration'].hist()
countries = df['country'].value_counts()

countries
fig = plt.figure(figsize=(23,6))



countries.plot.bar()



plt.savefig('countries.png',dpi=200)

plt.show()
df.replace(['Not specified'], [0], inplace=True)

df_longitude= df['longitude'].astype(float)
df_longitude.hist(bins=360)
df_latitude= df['latitude'].astype(float)

df_latitude.hist(bins=180)
#!pip install basemap
%matplotlib inline

from mpl_toolkits.basemap import Basemap



import matplotlib.gridspec as gridspec

from itertools import chain





def draw_map(m, scale):

    # draw a shaded-relief image

    m.shadedrelief(scale=scale)

    

    # lats and longs are returned as a dictionary

    lats = m.drawparallels(np.linspace(-90, 90, 13))

    lons = m.drawmeridians(np.linspace(-180, 180, 13))



    # keys contain the plt.Line2D instances

    lat_lines = chain(*(tup[1][0] for tup in lats.items()))

    lon_lines = chain(*(tup[1][0] for tup in lons.items()))

    all_lines = chain(lat_lines, lon_lines)

    

    # cycle through these lines and set the desired style

    for line in all_lines:

        line.set(linestyle='--', alpha=0.5, color='w')



    

fig = plt.figure(figsize=(15, 10), edgecolor='w')

gs = gridspec.GridSpec(4,100) # 5 rows, 3 column



## ---------------------------------------------------------------------------------------------------------------

## World Map

ax1 = plt.subplot(gs[:2, 1:99]) 

m = Basemap(ax=ax1,projection='cyl', resolution='l', llcrnrlat=-90, urcrnrlat=90,llcrnrlon=-180, urcrnrlon=180, ) # 'mill'; None, 'c'

draw_map(m, scale=0.5)

## ---------------------------------------------------------------------------------------------------------------



## ---------------------------------------------------------------------------------------------------------------

## Our data

ax2 = plt.subplot(gs[2, 19:81]) #Third row, span all columns by :

ax2.hist(df_longitude, 360, histtype='bar', orientation='vertical', color='blue',alpha=0.5)

plt.xticks([-180, -150, -120, -90, -60, -30, 0, 30, 60 , 90, 120, 150, 180], [-180, -150, -120, -90, -60, -30, 0, 30, 60 , 90, 120, 150, 180],rotation=0)

plt.yscale('symlog')

plt.ylim(1,2000)

plt.grid(True)

## ---------------------------------------------------------------------------------------------------------------



#plt.tight_layout()

plt.savefig('world_plus_distribution.png',dpi=200)



plt.show()



## For world map, can be cutted via

#lons, lats = np.meshgrid(m.drawmeridians(np.linspace(-180, 180, 13)),m.drawparallels(np.linspace(-90, 90, 13)))

#x, y = m(lons, lats)

#plt.xlim(-90, 90)

#plt.ylim(-45, 45)
lon = df_longitude

lat = df_latitude
duration = df['duration'].values

rating = df['rating'].values



# Draw the map background

fig = plt.figure(figsize=(23, 23))

m = Basemap(projection='cyl', resolution='l', lat_0=90, lon_0=0) # high resolution basemap-data-hires is needed, no internet is used in this competiton

m.shadedrelief()

m.drawcoastlines(color='gray')

m.drawcountries(color='gray')

m.drawstates(color='gray')



# Scatter city data, with color reflecting rating and size reflecting duration

m.scatter(lon, lat, latlon=True, c=rating, s=np.log10(duration+2)**3, cmap='Reds', alpha=0.5) # +2 to avoid log10 error, xx2 to increase visibility



# Create colorbar and legend

#plt.colorbar(label=r'rating')

##plt.clim(3, 7)



# create an axes on the right side of ax. The width of cax will be 5%

# of ax and the padding between cax and ax will be fixed at 0.05 inch.

from mpl_toolkits.axes_grid1 import make_axes_locatable

ax = plt.gca()

divider = make_axes_locatable(ax)

cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(cax=cax)



# Legend with dummy points

for a in [100, 500, 1000, 2000]:

    plt.scatter([], [], c='k', alpha=0.5, s=np.log10(a+2)**3,label=str(a) + ' sec')

    

plt.legend(scatterpoints=1, frameon=False, labelspacing=1, loc='lower left');



plt.savefig('geo_duration_rating.png',dpi=300)
fig = plt.figure(figsize=(15, 9))

plt.scatter(x=df['longitude'].astype(float), y=df['latitude'].astype(float))



plt.xticks([-180, -150, -120, -90, -60, -30, 0, 30, 60 , 90, 120, 150, 180], [-180, -150, -120, -90, -60, -30, 0, 30, 60 , 90, 120, 150, 180],rotation=0)

plt.ylim(-90,90)

plt.grid(True)



plt.savefig('birds_location.png',dpi=300)



plt.show()
from shapely.geometry import Point

import geopandas as gpd

from geopandas import GeoDataFrame



#fig = plt.figure(figsize=(15, 9))



geometry = [Point(xy) for xy in zip(df['longitude'].astype(float), df['latitude'].astype(float))]

gdf = GeoDataFrame(df, geometry=geometry)  



#this is a simple map that goes with geopandas

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

gdf.plot(ax=world.plot(figsize=(23, 16)), marker='o', color='red', markersize=10);



plt.savefig('birds_location_world.png',dpi=300)



plt.show()
df['elevation']
df['elevation'].value_counts().unique
elev_list = list(df['elevation'])

elev_list,len(elev_list)
df['elevation'].head(5)
import re



elev_list_num = []



for i in range(len(elev_list)):

    taken_num = re.findall(r"[-+]?\d*\.\d+|\d+", elev_list[i])

    print(taken_num)

    elev_list_num.append(taken_num)

    

elev_list_num
len(elev_list_num)
elev_list_num[449]
elev_list_num2 = []

for i in range(len(elev_list_num)):

    if elev_list_num[i] == []:

        elev_list_num2.append(0.000)

    else:

        elev_list_num2.append(float(elev_list_num[i][0]))
len(elev_list_num2)
df_elevation = pd.DataFrame(elev_list_num2)

df_elevation.head(5)
df_elevation.rename(columns={0: 'elevation'}, inplace=True)

df_elevation.head(5)
fig = plt.figure(figsize=(23, 8))



plt.ylabel('Counts')

plt.xlabel('Elevation value');



degrees = 90

plt.xticks(rotation=degrees)



plt.hist(elev_list_num, density=False, bins=30)  # density=True



plt.savefig('birds_elevation.png',dpi=300)



plt.show()
df['longitude'].shape,df['longitude'].shape,df_elevation['elevation'].shape
df_elevation['elevation'].max()
fig = plt.figure(figsize=(20,20))

ax = fig.add_subplot(111, projection='3d')



ax.scatter((1)*df['latitude'].astype(float),(1)*df['longitude'].astype(float),df_elevation['elevation']/1000, s = 0.5, color = 'r')



ax.set_xlim(-90,90)

ax.set_ylim(-180,180)

ax.set_zlim(0,5)



ax.set_xlabel('lat')

ax.set_ylabel('lon')

ax.set_zlabel('elev [km]')

ax.set_title('Birds on Sky :) ',fontsize=18)



ax.view_init(90, 90)



plt.savefig('birds_3D_elevation_topview.png',dpi=300)



plt.show()
fig = plt.figure(figsize=(20,20))

ax = fig.add_subplot(111, projection='3d')



ax.scatter((1)*df['latitude'].astype(float),(-1)*df['longitude'].astype(float),df_elevation['elevation']/1000, s = 0.5, color = 'r')



ax.set_xlim(-90,90)

ax.set_ylim(-180,180)

ax.set_zlim(0,5)



ax.set_xlabel('lat')

ax.set_ylabel('lon')

ax.set_zlabel('elev [km]')

ax.set_title('Birds on Sky :) ',fontsize=18)



ax.view_init(40, 135)



plt.savefig('birds_3D_elevation.png',dpi=300)



plt.show()
import seaborn as sns
sns.set(rc={'figure.figsize':(10,10)})



sns_plot = sns.jointplot(x='latitude', y='longitude', data=df, kind='kde')



sns_plot.savefig('birds_2D_world.png', dpi=300)
from IPython.display import Image

Image(filename='../working/birds_2D_world.png') 
fig = plt.figure(figsize=(23, 6), edgecolor='w')

df['date'].value_counts().sort_index().plot(c='green', linewidth=1)

plt.savefig('date.png',dpi=200)
fig = plt.figure(figsize=(12, 10), edgecolor='w')



time_series = df['date'].value_counts().reset_index()

time_series.columns = ['date', 'count']



time_series.plot(kind='kde')

time_series.plot(kind='hist')



#plt.savefig('date2.png',dpi=200)

plt.show()
sub.to_csv('submission.csv', index=False)