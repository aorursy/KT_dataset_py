!pip install rioxarray

!pip install earthpy
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python





#We import all the necessary libraries in one place



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import folium

from folium import plugins

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster

import os

import numpy as np

import matplotlib.pyplot as plt

import earthpy as et

import earthpy.spatial as es

import earthpy.plot as ep

import rasterio as rio

from rasterio.plot import show

from rasterio.plot import show_hist

import rioxarray

import geopandas as gpd

from rasterio.plot import show

import fiona

%matplotlib inline





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
border=gpd.read_file('/kaggle/input/malawiadminboundaries/MWI_adm1.shp')

flood_data_df1=pd.read_csv('/kaggle/input/malawi-flood-data/Train_data_Malawi_flood_prediction.csv')

flood_data_df=flood_data_df1[flood_data_df1['target_2015']!=0]

flood_data1 = gpd.GeoDataFrame(flood_data_df, geometry=gpd.points_from_xy(flood_data_df.X, flood_data_df.Y))

flood_data2=gpd.read_file('/kaggle/input/malawifloods2015/malawi_mozambique_flooding.shp')

lgnd_kwds = {'title': 'Malawi Districts',

               'loc': 'upper left', 'bbox_to_anchor': (1, 1.03), 'ncol': 2}

ax = border.plot(figsize=(10,15), linestyle=':', cmap='tab20c',edgecolor='black',column='NAME_1',legend=True,legend_kwds =lgnd_kwds)

flood_data1.plot(markersize=0.25,alpha=0.25,ax=ax,color='red')

flood_data2.plot(markersize=0.25,alpha=0.25,ax=ax,color='red')

_=ax.set_xlabel('Longitude')

_=ax.set_ylabel('Latitude')

_=ax.set_title('Districts of Malawi and extent of flooding')
flood_data1.crs={'init': 'epsg:4326'}

join=gpd.sjoin(border,flood_data1)

join.to_file("join.shp")

ax = join.plot(figsize=(10,15), linestyle=':', cmap='tab20c',edgecolor='black',column='NAME_1',legend=True,legend_kwds =lgnd_kwds)

flood_data1.plot(markersize=0.50,alpha=1.0,ax=ax,color='red',legend_kwds =lgnd_kwds)

_=ax.set_xlabel('Longitude')

_=ax.set_ylabel('Latitude')

_=ax.set_title('Districts of Malawi affected by floods')
affected_districts=join['NAME_1'].unique()

print('Districts affected by flood',len(affected_districts))

print('*'*50)

print('Name of affected districts\n', affected_districts)
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 16))

pop=gpd.read_file('/kaggle/input/malawipopdensity/Malawi_Population.shp')

pop.plot(figsize=(12,12),column='Population',legend=True,ax=ax1)

flood_data1.plot(markersize=0.25,alpha=0.25,color='red',ax=ax1)

flood_data2.plot(markersize=0.25,alpha=0.25,color='red',ax=ax1)

_=ax1.set_xlabel('Longitude')

_=ax1.set_ylabel('Latitude')

_=ax1.set_title('Population & flood extent')

pop.plot(figsize=(12,12),column='Pop_Den',legend=True,ax=ax2)

flood_data1.plot(markersize=0.25,alpha=0.25,color='red',ax=ax2)

flood_data2.plot(markersize=0.25,alpha=0.25,color='red',ax=ax2)

_=ax2.set_xlabel('Longitude')

_=ax2.set_ylabel('Latitude')

_=ax2.set_title('Population density & flood extent')
join1=gpd.sjoin(pop,flood_data1)

ax = join1.plot(figsize=(10,15), linestyle=':', cmap='tab20c',edgecolor='black',column='Population',legend=True)

flood_data1.plot(markersize=0.50,alpha=1.0,ax=ax,color='red')

_=ax.set_xlabel('Longitude')

_=ax.set_ylabel('Latitude')

_=ax.set_title('Population of Malawi affected by floods')
df=pd.DataFrame(join1.groupby(['ADM2_NAME'])['target_2015','Population'].mean().reset_index())

df.columns=['District','Flood_pct','Population']

df
health_sites=gpd.read_file('/kaggle/input/malawihealthsites/healthsites.shp')

lgnd_kwds = {'title': 'Malawi Districts',

               'loc': 'upper left', 'bbox_to_anchor': (1, 1.03), 'ncol': 2}

ax = border.plot(figsize=(10,15), linestyle=':', cmap='tab20c',edgecolor='black',column='NAME_1',legend=True,legend_kwds =lgnd_kwds)

flood_data1.plot(markersize=0.25,alpha=0.25,ax=ax,color='red')

flood_data2.plot(markersize=0.25,alpha=0.25,ax=ax,color='red')

health_sites.plot(markersize=1.0,alpha=1.0,ax=ax,color='blue')

_=ax.set_xlabel('Longitude')

_=ax.set_ylabel('Latitude')

_=ax.set_title('Districts of Malawi, extent of flooding and healthsites')
#Lets do some analysis on the number of health sites impacted by floods. 



print('Total number of health sites are:',len(health_sites))

# District wise plot

health_districts=gpd.sjoin(border,health_sites)

df=pd.DataFrame(health_districts.groupby(['NAME_1'])['amenity'].count().reset_index(name = "Group_Count"))

df.columns=['District','Number_of_healthcare_facilities']

df['affected']=df['District'].apply(lambda x:1 if x in affected_districts else 0)

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,6))

_=df[df['affected']==0].plot.bar(x='District',y='Number_of_healthcare_facilities',ax=ax1,color='blue',legend=False)

_=df[df['affected']==1].plot.bar(x='District',y='Number_of_healthcare_facilities',ax=ax2,color='red',legend=False)

_=ax1.set_xlabel('Districts')

_=ax1.set_ylabel('Number_of_healthcare_facilities')

_=ax1.set_title('Unaffected Districts of Malawi & number of healthcare facilities in them')

_=ax2.set_xlabel('Districts')

_=ax2.set_ylabel('Number_of_healthcare_facilities')

_=ax2.set_title('Affected Districts of Malawi & number of healthcare facilities in them')
print('Number of healthcare facilities in affected districts:',np.sum(df.loc[df['affected']==1,'Number_of_healthcare_facilities']))

print('Number of healthcare facilities in unaffected districts:',np.sum(df.loc[df['affected']==0,'Number_of_healthcare_facilities']))
lc='/kaggle/input/malawisentinel/Malawi_Sentinel2_LULC2016.tif'

with rio.open(lc) as src:

    lu_lc = src.read()

    lu_lc[lu_lc < 0.0] = 0.0



#Plot the data

ep.plot_bands(

    lu_lc,

    cmap="gist_earth_r",

    title="Land cover of Malawi",

    figsize=(10, 10),

)

plt.show()
with fiona.open("/kaggle/working/join.shp", "r") as shapefile:

    geoms = [feature["geometry"] for feature in shapefile]



with rio.open(lc) as src:

    out_image, out_transform = rio.mask.mask(src, geoms, crop=True)

    out_meta = src.meta

out_image[out_image < 0] = 0

ep.plot_bands(

    out_image,

    cmap="gist_earth_r",

    title="Land cover of Malawi districts affected by flood",

    figsize=(10, 10),

)

plt.show()
#Load the digital elevation model which is stored as a geo-tiff file

dtm = "/kaggle/input/demmalawi/Malawi SRTM DEM 30meters.tiff"

with rio.open(dtm) as src:

    elevation = src.read(1)

    # Set masked values to np.nan

    elevation[elevation < 0.0] = 0.0



# Plot the data

ep.plot_bands(

    elevation,

    cmap="gist_earth",

    title="DTM Without Hillshade",

    figsize=(25, 25),

)

plt.show()
hillshade = es.hillshade(elevation)

ep.plot_bands(

    hillshade, cbar=False, title="Hillshade made from DEM", figsize=(25, 25),

)

plt.show()
fig, ax = plt.subplots(figsize=(100, 50))

ep.plot_bands(

    elevation,

    ax=ax,

    cmap="terrain",

    title="Digital Elevation Model (DEM)\n overlayed on top of a hillshade",

)

ax.imshow(hillshade, cmap="Greys", alpha=0.5)

plt.show()
with fiona.open("/kaggle/working/join.shp", "r") as shapefile:

    geoms = [feature["geometry"] for feature in shapefile]



with rio.open(dtm) as src:

    dtm_image, dtm_transform = rio.mask.mask(src, geoms, crop=True)

    dtm_meta = src.meta

dtm_image[dtm_image < 0] = 0

ep.plot_bands(

    dtm_image,

    cmap="gist_earth",

    title="Elevation map of Malawi districts affected by flood",

    figsize=(10, 10),

)

plt.show()
m = folium.Map(location=[-15.30,34], tiles='Stamen Terrain', zoom_start=8)

style1 = {'fillColor': 'blue', 'color': 'black'}

_=folium.GeoJson(border.geometry,style_function=lambda x:style1).add_to(m)



def color_producer(val):

    if val <= 0.5:

        return 'forestgreen'

    elif val <=0.75:

        return 'orange'

    else:

        return 'darkred'



# Add a bubble map to the base map

for i in range(0,len(flood_data1)):

    Circle(

        location=[flood_data1.iloc[i]['Y'], flood_data1.iloc[i]['X']],

        radius=20,

        color=color_producer(flood_data1.iloc[i]['target_2015'])).add_to(m)



m