# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt 

import missingno as msno

import rasterio as rio

import folium

import tifffile as tiff 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv'

df = pd.read_csv(data)
def split_latnlog_into_new_columns_fromgeo(dataframe,column_to_split,new_column_one,begin_column_one,end_column_one):

    for i in range(0, len(dataframe)):

        dataframe.loc[i, new_column_one] = dataframe.loc[i, column_to_split][begin_column_one:end_column_one]

    return dataframe
power_plants = split_latnlog_into_new_columns_fromgeo(df,'.geo','latitude',50,66)

power_plants = split_latnlog_into_new_columns_fromgeo(df,'.geo','longitude',31,48)
power_plants.head()
plt.figure(figsize=(8,8))

sns.catplot('primary_fuel', data= power_plants, kind='count', alpha=0.7, height=6, aspect= 3.5)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = power_plants['primary_fuel'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of primary_fule', fontsize = 20, color = 'black')

plt.show()
plt.figure(figsize=(8,8))

sns.catplot('source', data= power_plants, kind='count', alpha=0.7, height=6, aspect= 3.5)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = power_plants['source'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of source', fontsize = 20, color = 'black')

plt.show()
plt.figure(figsize=(8,8))

sns.catplot('owner', data= power_plants, kind='count', alpha=0.7, height=6, aspect= 3.5)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = power_plants['owner'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of owner ', fontsize = 20, color = 'black')

plt.show()
from scipy import stats



sns.distplot(power_plants['estimated_generation_gwh'] , fit=stats.norm);



# Get the fitted parameters used by the function

(mu, sigma) = stats.norm.fit(power_plants['estimated_generation_gwh'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('estimated_generation_gwh distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(power_plants['estimated_generation_gwh'], plot=plt)

plt.show()
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

power_plants["estimated_generation_gwh"] = np.log1p(power_plants["estimated_generation_gwh"])



#Check the new distribution 

sns.distplot(power_plants['estimated_generation_gwh'] , fit=stats.norm);



# Get the fitted parameters used by the function

(mu, sigma) =stats.norm.fit(power_plants['estimated_generation_gwh'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('estimated_generation_gwh distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(power_plants['estimated_generation_gwh'], plot=plt)

plt.show()
plt.figure(figsize=(8,8))

sns.catplot(x="primary_fuel",

            y="capacity_mw",

            data=power_plants,

            jitter=False,

           )

plt.show()
missing_df = df.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name','missing_count']

missing_df = missing_df.ix[missing_df['missing_count']>0]

missing_df = missing_df.sort_values(by='missing_count')



ind = np.arange(missing_df.shape[0])

width = 0.5

fig,ax = plt.subplots(figsize=(12,18))

rects = ax.barh(ind,missing_df.missing_count.values,color='blue')

ax.set_yticks(ind)

ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_title("Number of missing values in each column")

plt.show()
generation_gwh_years = ["generation_gwh_2013", 

                        "generation_gwh_2014", 

                        "generation_gwh_2015", 

                        "generation_gwh_2016",

                        "generation_gwh_2017"]



power_plants.loc[:, generation_gwh_years].sum()
power_plants_df = power_plants.sort_values('capacity_mw',ascending=False).reset_index()

power_plants_df=power_plants_df[['name','latitude','longitude','primary_fuel','owner','capacity_mw','estimated_generation_gwh',]]

power_plants_df.head()
world_d= dict(

   name=list(power_plants['country']),

    lat=list(power_plants['latitude']),

   lon=list(power_plants['longitude']),

   estimated_generation_gwh =list(power_plants['estimated_generation_gwh'])

)

world_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in world_d.items() ]))

world_data = world_data.fillna(method='ffill') 





# create map and display it

world_map = folium.Map(location=[18, -66], zoom_start=9)



for lat, lon, value, name in zip(world_data['lat'], world_data['lon'], world_data['estimated_generation_gwh'], world_data['name']):

    folium.CircleMarker([lat, lon],

                        radius=10,

                        popup = ('<strong>country</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>estimated_generation_gwh</strong>: ' + str(value) + '<br>'),

                        color='red',

                        

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(world_map)

world_map
def plot_tif_img_on_map(file_name,lat,lon,zoom):

    wor_map = folium.Map([lat, lon], zoom_start=zoom)

    folium.raster_layers.ImageOverlay(

        image=file_name,

        bounds = [[18.6,-67.3,],[17.9,-65.2]],

        colormap=lambda x: (1, 0, 0, x),

    ).add_to(wor_map)

    return wor_map
image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180708T172237_20180714T190743.tif'

band = rio.open(image).read(7)

print(band.shape)

vmin, vmax = np.nanpercentile(band, (5,95))  # 5-95% stretch

img_plt = plt.imshow(band, cmap='Oranges', vmin=vmin, vmax=vmax)

plt.show()

latitude=18.1429005246921; longitude=-65.4440010699994

plot_tif_img_on_map(band,latitude,longitude,9)
image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180707T174140_20180713T191854.tif'

band = rio.open(image).read(7)

print(band.shape)

vmin, vmax = np.nanpercentile(band, (5,95))  # 5-95% stretch

img_plt = plt.imshow(band, cmap='Oranges', vmin=vmin, vmax=vmax)

plt.show()

latitude=18.1429005246921; longitude=-65.4440010699994

plot_tif_img_on_map(band,latitude,longitude,9)
image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/gfs_2018070106.tif'

band = rio.open(image).read(6)

vmin, vmax = np.nanpercentile(band, (5,95))  # 5-95% stretch

img_plt = plt.imshow(band, cmap='Oranges', vmin=vmin, vmax=vmax)

plt.show()

latitude=18.1429005246921; longitude=-65.4440010699994

plot_tif_img_on_map(band,latitude,longitude,9)
from rasterio.plot import show

image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gldas/gldas_20180701_0300.tif'



#load the image

band = rio.open(image)

show(band)

#All Metadata for the whole raster dataset

band.meta
image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/gfs_2018070106.tif'



#load the image

band = rio.open(image)

show(band)

#All Metadata for the whole raster dataset

band.meta
image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180718T173458_20180724T193016.tif'



#load the image

band = rio.open(image)

show(band)

#All Metadata for the whole raster dataset

band.meta
quantity_of_electricity_generated = power_plants_df['estimated_generation_gwh'][29:30].values

print('Quanity of Electricity Generated: ', quantity_of_electricity_generated)
image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180718T173458_20180724T193016.tif'



average_no2_emission = [np.average(tiff.imread(image))]

print('Average NO2 emissions value: ', average_no2_emission)
simplified_emissions_factor = float(average_no2_emission/quantity_of_electricity_generated)

print('Simplified emissions factor (S.E.F.) for a single power plant on the island of Vieques =  \n\n', simplified_emissions_factor, 'S.E.F. units')