# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

#Analysing datetime

import datetime as dt

from datetime import datetime

#plotting geographical data

import folium

import rasterio as rio

#file system mgmt

import os

#supress warnings

import warnings

warnings.filterwarnings('ignore')
#Total number of power plants in puerto rico

global_power_plants = pd.read_csv('../input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')

global_power_plants.head().T
#number of different power plants

global_power_plants.shape
#lets check the different kinds of power plants based on primary fuel used

sns.barplot(x=global_power_plants['primary_fuel'].value_counts().index, 

            y=global_power_plants['primary_fuel'].value_counts())

plt.ylabel('count')
global_power_plants['commissioning_year'].value_counts()
#The different sources of data

fig = plt.gcf()

fig.set_size_inches(10,6)

colors = ['dodgerblue', 'plum', '#F0A30A','#8c564b','orange','green','yellow']

global_power_plants['source'].value_counts(ascending=True).plot(kind='barh',color=colors,linewidth=2,edgecolor='black')
#Who  is the ownner the power plants



fig = plt.gcf()

fig.set_size_inches(10, 6)

colors = ['dodgerblue', 'plum', '#F0A30A','#8c564b','orange','green','yellow'] 

global_power_plants['owner'].value_counts(ascending=True).plot(kind='barh',color=colors)

# Total capacity of all the plants

total_capacity_mw = global_power_plants['capacity_mw'].sum()

print('Total Installed Capacity: '+'{:.2f}'.format(total_capacity_mw) + ' MW')
capacity = (global_power_plants.groupby(['primary_fuel'])['capacity_mw'].sum()).to_frame()

capacity = capacity.sort_values('capacity_mw',ascending=False)

capacity['percentage_of_total'] = (capacity['capacity_mw']/total_capacity_mw)*100

capacity
fig = plt.gcf()

fig.set_size_inches(10, 6)

colors = ['dodgerblue', 'plum', '#F0A30A','#8c564b','orange','green','yellow'] 

capacity['percentage_of_total'].plot(kind='bar',color=colors)
# Total generation of all the plants

total_gen_mw = global_power_plants['estimated_generation_gwh'].sum()

print('Total Generatation: '+'{:.2f}'.format(total_gen_mw) + ' GW')
generation = (global_power_plants.groupby(['primary_fuel'])['estimated_generation_gwh'].sum()).to_frame()

generation = generation.sort_values('estimated_generation_gwh',ascending=False)

generation['percentage_of_total'] = (generation['estimated_generation_gwh']/total_gen_mw)*100

generation
# Code source: https://www.kaggle.com/paultimothymooney/overview-of-the-eie-analytics-challenge

from folium import plugins      

def plot_points_on_map(dataframe,begin_index,end_index,latitude_column,latitude_value,longitude_column,longitude_value,zoom):

    df = dataframe[begin_index:end_index]

    location = [latitude_value,longitude_value]

    plot = folium.Map(location=location,zoom_start=zoom,tiles = 'Stamen Terrain')

    

    for i in range(0,len(df)):

        popup = folium.Popup(str(df.primary_fuel[i:i+1]))

        folium.Marker([df[latitude_column].iloc[i],

                       df[longitude_column].iloc[i]],

                       popup=popup,icon=folium.Icon(color='white',icon_color='red',icon ='bolt',prefix='fa',)).add_to(plot)

    return(plot)



def overlay_image_on_puerto_rico(file_name,band_layer,lat,lon,zoom):

    band = rio.open(file_name).read(band_layer)

    m = folium.Map([lat, lon], zoom_start=zoom)

    folium.raster_layers.ImageOverlay(

        image=band,

        bounds = [[18.6,-67.3,],[17.9,-65.2]],

        colormap=lambda x: (1, 0, 0, x),

    ).add_to(m)

    return m



def split_column_into_new_columns(dataframe,column_to_split,new_column_one,begin_column_one,end_column_one):

    for i in range(0, len(dataframe)):

        dataframe.loc[i, new_column_one] = dataframe.loc[i, column_to_split][begin_column_one:end_column_one]

    return dataframe

    
global_power_plants = split_column_into_new_columns(global_power_plants,'.geo','latitude',50,66)

global_power_plants = split_column_into_new_columns(global_power_plants,'.geo','longitude',31,48)

global_power_plants['latitude'] = global_power_plants['latitude'].astype(float)

a = np.array(global_power_plants['latitude'].values.tolist()) 

global_power_plants['latitude'] = np.where(a < 10, a+10, a).tolist() 



lat=18.200178; lon=-66.664513 # Puerto Rico's co-ordinates

plot_points_on_map(global_power_plants,0,425,'latitude',lat,'longitude',lon,9)
image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180701T161259_20180707T175356.tif'



# Opening the file

raster = rio.open(image)



# All Metadata for the whole raster dataset

raster.meta
from rasterio.plot import show

show(raster)
# Plotting the red channel.

show((raster, 4), cmap='Reds')
# Calculating the dimensions of the image on earth in metres

sat_data = raster



width_in_projected_units = sat_data.bounds.right - sat_data.bounds.left

height_in_projected_units = sat_data.bounds.top - sat_data.bounds.bottom

print("Width: {}, Height: {}".format(width_in_projected_units, height_in_projected_units))

print("Rows: {}, Columns: {}".format(sat_data.height, sat_data.width))
# Upper left pixel

row_min = 0

col_min = 0

# Lower right pixel.  Rows and columns are zero indexing.

row_max = sat_data.height - 1

col_max = sat_data.width - 1

# Transform coordinates with the dataset's affine transformation.

topleft = sat_data.transform * (row_min, col_min)

botright = sat_data.transform * (row_max, col_max)

print("Top left corner coordinates: {}".format(topleft))

print("Bottom right corner coordinates: {}".format(botright))
print(sat_data.count)



# sequence of band indexes

print(sat_data.indexes)
# Load the 12 bands into 2d arrays

b01, b02, b03, b04,b05,b06,b07,b08, b09,b10, b11, b12 = sat_data.read()
# Displaying the second band.



fig = plt.imshow(b02)

plt.show()
fig = plt.imshow(b03)

fig.set_cmap('gist_earth')

plt.show()
fig = plt.imshow(b04)

fig.set_cmap('inferno')

plt.colorbar()

plt.show()
# Displaying the infrared band.



fig = plt.imshow(b08)

fig.set_cmap('winter')

plt.colorbar()

plt.show()