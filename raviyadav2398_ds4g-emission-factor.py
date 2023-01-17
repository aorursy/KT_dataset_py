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

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import datetime as dt

from datetime import datetime

# color pallette

cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801'

import folium 

from folium import plugins

import plotly.express as px

import rasterio as rio

import warnings

warnings.filterwarnings('ignore')
power_plant=pd.read_csv('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')

display(power_plant.shape)

display(power_plant.columns)

display(power_plant['system:index'].is_unique)

display(power_plant.head())
power_plant.set_index('system:index',inplace=True)

power_plant.head().T
display("Database Country")

display(power_plant['country'].value_counts())

display("ID of Global Power Plant Database")

display(power_plant['gppd_idnr'].value_counts())

display("Stockholder of power plant")

display(power_plant['owner'].value_counts())

display("Source to generated electricity")

display(power_plant['source'].value_counts())

display("Types of primary fuel  used :")

display(power_plant['primary_fuel'].value_counts())
power_plant['name'].value_counts()
#plot primary fuel used

sns.countplot(power_plant['primary_fuel'])
#Year of establishment of plant

power_plant['commissioning_year'].value_counts()
#contribution of Stock Holder

power_plant['owner'].value_counts(ascending=True).plot(kind='barh',title='Contribution of stock holders ')
#types of source

power_plant['source'].value_counts(ascending=True).plot(kind='barh',title='Types of source')
temp=power_plant.groupby('commissioning_year')['estimated_generation_gwh','capacity_mw'].sum().reset_index()

temp=temp[temp['commissioning_year']==max(temp['commissioning_year'])].reset_index(drop=True)

tm=temp.melt(id_vars="commissioning_year",value_vars=["estimated_generation_gwh","capacity_mw"])

temp.head()

fig=px.treemap(tm,path=["variable"],values="value",height=225,width=1200,color_discrete_sequence=[act,rec])

fig.data[0].textinfo='label+text+value'

fig.show()
#Estimated generation growth from commissioning year

temp=power_plant.groupby('commissioning_year')['estimated_generation_gwh','capacity_mw'].sum().reset_index()

temp=temp.melt(id_vars="commissioning_year",value_vars=["estimated_generation_gwh","capacity_mw"],var_name='Year',value_name='Count')

temp.head()

fig=px.area(temp,x='commissioning_year',y='Count',color='Year',height=600,title='Production  over time',color_discrete_sequence=[rec,dth])

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
full_grouped=power_plant.groupby(['source','primary_fuel',])['capacity_mw','estimated_generation_gwh'].sum().reset_index()

temp_1=full_grouped.sort_values(by='estimated_generation_gwh',ascending=False)

temp_1=temp_1.reset_index(drop=True)

temp_1.style.background_gradient(cmap='Blues')
#total electricity generation in gigwatthour for one year

total_gen=power_plant['estimated_generation_gwh'].sum()

print('Total Generation :'+'{:.3f}'.format(total_gen)+'GW')
# percentage of total generation in gigawatthour

generation = (power_plant.groupby(['primary_fuel'])['estimated_generation_gwh'].sum()).to_frame()

generation = generation.sort_values('estimated_generation_gwh',ascending=False)

generation['percentage_of_total'] = (generation['estimated_generation_gwh']/total_gen)*100

generation
fig = plt.gcf()

fig.set_size_inches(10, 6)

colors = ['dodgerblue', 'plum', '#F0A30A','#8c564b','orange','green','yellow'] 

generation['percentage_of_total'].plot(kind='bar',color=colors)


generation = (power_plant.groupby(['source'])['estimated_generation_gwh'].sum()).to_frame()

generation = generation.sort_values('estimated_generation_gwh',ascending=False)

generation['percentage_of_total'] = (generation['estimated_generation_gwh']/total_gen)*100

generation
#total production capcity 

total_cap=power_plant['capacity_mw'].sum()

print('Total Capcity :'+'{:.3f}'.format(total_cap)+'MW')
capcity = (power_plant.groupby(['primary_fuel'])['capacity_mw'].sum()).to_frame()

capcity = capcity.sort_values('capacity_mw',ascending=False)

capcity['percentage_of_total'] = (capcity['capacity_mw']/total_cap)*100

capcity
fig = plt.gcf()

fig.set_size_inches(10, 6)

colors = ['dodgerblue', 'plum', '#F0A30A','#8c564b','orange','green','yellow'] 

capcity['percentage_of_total'].plot(kind='bar',color=colors)
capcity = (power_plant.groupby(['source'])['capacity_mw'].sum()).to_frame()

capcity = capcity.sort_values('capacity_mw',ascending=False)

capcity['percentage_of_total'] = (capcity['capacity_mw']/total_cap)*100

capcity
def plot_points_on_map(dataframe,begin_index,end_index,latitude_column,latitude_value,longitude_column,longitude_value,zoom):

    df = dataframe[begin_index:end_index]

    location = [latitude_value,longitude_value]

    plot = folium.Map(location=location,zoom_start=zoom)

    for i in range(0,len(df)):

        popup = folium.Popup(str(df.primary_fuel[i:i+1]))

        folium.Marker([df[latitude_column].iloc[i],df[longitude_column].iloc[i]],popup=popup).add_to(plot)

    return(plot)



def overlay_image_on_puerto_rico(file_name,band_layer):

    band = rio.open(file_name).read(band_layer)

    m = folium.Map([lat, lon], zoom_start=8)

    folium.raster_layers.ImageOverlay(

        image=band,

        bounds = [[18.6,-67.3,],[17.9,-65.2]],

        colormap=lambda x: (1, 0, 0, x),

    ).add_to(m)

    return m



def plot_scaled(file_name):

    vmin, vmax = np.nanpercentile(file_name, (5,95))  # 5-95% stretch

    img_plt = plt.imshow(file_name, cmap='gray', vmin=vmin, vmax=vmax)

    plt.show()



def split_column_into_new_columns(dataframe,column_to_split,new_column_one,begin_column_one,end_column_one):

    for i in range(0, len(dataframe)):

        dataframe.loc[i, new_column_one] = dataframe.loc[i, column_to_split][begin_column_one:end_column_one]

    return dataframe

power_plant = pd.read_csv('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')

power_plant = split_column_into_new_columns(power_plant,'.geo','latitude',50,66)

power_plant = split_column_into_new_columns(power_plant,'.geo','longitude',31,48)

power_plant['latitude'] = power_plant['latitude'].astype(float)

a = np.array(power_plant['latitude'].values.tolist()) # 18 instead of 8

power_plant['latitude'] = np.where(a < 10, a+10, a).tolist() 

lat=18.200178; lon=-66.664513

plot_points_on_map(power_plant,0,425,'latitude',lat,'longitude',lon,9)
image='/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gldas/gldas_20181210_0600.tif'

image_band=rio.open(image).read(1)

plot_scaled(image_band)

overlay_image_on_puerto_rico(image,band_layer=1)

image='/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/gfs_2019051106.tif'

image_band=rio.open(image).read(1)

plot_scaled(image_band)

overlay_image_on_puerto_rico(image,band_layer=1)

image='/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/gfs_2019031218.tif'

image_band=rio.open(image).read(1)

plot_scaled(image_band)

overlay_image_on_puerto_rico(image,band_layer=1)
# checck random image:

image='../input/inputravi/l3-ne43h01-094-059-01feb2013-band2.tif'

image_band=rio.open(image).read(1)

plot_scaled(image_band)

overlay_image_on_puerto_rico(image,band_layer=1)
from kaggle_secrets import UserSecretsClient

from google.oauth2.credentials import Credentials

import ee

import folium



def add_ee_layer(self, ee_image_object, vis_params, name):

  # https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/ee-api-colab-setup.ipynb

  map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)

  folium.raster_layers.TileLayer(

    tiles = map_id_dict['tile_fetcher'].url_format,

    attr = 'Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',

    name = name,

    overlay = True,

    control = True

  ).add_to(self)



def plot_ee_data_on_map(dataset,column,begin_date,end_date,minimum_value,maximum_value,latitude,longitude,zoom):

    # https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/ee-api-colab-setup.ipynb

    folium.Map.add_ee_layer = add_ee_layer

    vis_params = {

      'min': minimum_value,

      'max': maximum_value,

      'palette': ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']}

    my_map = folium.Map(location=[latitude,longitude], zoom_start=zoom, height=500)

    s5p = ee.ImageCollection(dataset).filterDate(

        begin_date, end_date)

    my_map.add_ee_layer(s5p.first().select(column), vis_params, 'Color')

    my_map.add_child(folium.LayerControl())

    display(my_map)
