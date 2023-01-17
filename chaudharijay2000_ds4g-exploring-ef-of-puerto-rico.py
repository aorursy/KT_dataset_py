## Importing necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Analysing datetime

import datetime as dt

from datetime import datetime 



# Plotting geographical data

import folium

import rasterio as rio



# File system manangement

import os



# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')
# Checking out number of global power plants in Puerto Rico

gpp_df = pd.read_csv('../input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')
gpp_df.head()
# Transposing the table of global power plants

gpp_df.head().T
# Looking the shape of dataframe

gpp_df.shape
import pandas_profiling

pandas_profiling.ProfileReport(gpp_df)
#to indicate if any value is missing. Any missing values?

gpp_df.isnull().values.any()
# Total missing values for each feature

gpp_df.isnull().sum()
# replacing numerical variables missing values by 0

global_power_plants = gpp_df.fillna(0)
global_power_plants.isnull().sum()
# List the global power plants which are used primary fuel.

sns.barplot(x=global_power_plants['primary_fuel'].value_counts().index,y=global_power_plants['primary_fuel'].value_counts())

plt.title("Global power plants which are used Primary fuel")

plt.ylabel('Count')
print(global_power_plants['commissioning_year'].value_counts(normalize=True))
fig = plt.gcf()

fig.set_size_inches(8,5)

colors = ["aquamarine", "plum", "orchid", "fuchsia", "goldenrod", "lavender", "olive", "lime", "turquoise"]

global_power_plants['owner'].value_counts(ascending=True).plot(kind='bar', color = colors)
# Overall capacity of the plants

overall_capacity = global_power_plants['capacity_mw'].sum()

print('Overall Capacity: '+'{:.4f}'.format(overall_capacity) + ' MW')
capacity = (global_power_plants.groupby(['primary_fuel'])['capacity_mw'].sum()).to_frame()

capacity = capacity.sort_values('capacity_mw',ascending=False)

capacity['percentage_of_total'] = (capacity['capacity_mw']/overall_capacity)*100

capacity['percentage_of_total'].plot(kind='bar',color=['orange', 'yellow', 'black', 'orange','cyan','blue'])

capacity
# Overall estimation of the plants

est_power_gen_gwh = global_power_plants['estimated_generation_gwh'].sum()

print('Total Estimated Capacity in Year: '+'{:.4f}'.format(est_power_gen_gwh) + ' MW')
generation = (global_power_plants.groupby(['primary_fuel'])['estimated_generation_gwh'].sum()).to_frame()

generation = generation.sort_values('estimated_generation_gwh',ascending=False)

generation['percentage_of_total'] = (generation['estimated_generation_gwh']/est_power_gen_gwh)*100

generation['percentage_of_total'].plot(kind='bar',color=['orange', 'yellow', 'black', 'orange','cyan','blue'])

generation
def split_column_into_new_columns(dataframe,column_to_split,new_column_one,begin_column_one,end_column_one):

    for i in range(0, len(dataframe)):

        dataframe.loc[i, new_column_one] = dataframe.loc[i, column_to_split][begin_column_one:end_column_one]

    return dataframe
def plot_points_on_map(dataframe,begin_index,end_index,latitude_column,latitude_value,longitude_column,longitude_value,zoom):

    df = dataframe[begin_index:end_index]

    location = [latitude_value,longitude_value]

    plot = folium.Map(location=location,zoom_start=zoom, tiles= 'Stamen Terrain')

    color={ 'Hydro' : 'lightblue', 'Solar' : 'orange', 'Oil' : 'darkblue', 'Coal' : 'black', 'Gas' : 'lightgray', 'Wind' : 'green' }

    for i in range(0,len(df)):

        popup = folium.Popup(str(df.primary_fuel[i:i+1]))        

        folium.Marker([df[latitude_column].iloc[i],

                       df[longitude_column].iloc[i]],

                       popup=popup,icon=folium.Icon(icon_color='red',icon ='bolt',prefix='fa',color=color[df.primary_fuel.iloc[i]])).add_to(plot)

        

    return(plot)
global_power_plants = split_column_into_new_columns(global_power_plants,'.geo','latitude',50,66)

global_power_plants = split_column_into_new_columns(global_power_plants,'.geo','longitude',31,48)

global_power_plants['latitude'] = global_power_plants['latitude'].astype(float)

a = np.array(global_power_plants['latitude'].values.tolist()) 

global_power_plants['latitude'] = np.where(a < 10, a+10, a).tolist() 



lat=18.200178; lon=-66.664513 # Puerto Rico's co-ordinates

plot_points_on_map(global_power_plants,0,425,'latitude',lat,'longitude',lon,9)
global_power_plants_df = global_power_plants.sort_values('capacity_mw',ascending=False).reset_index()

global_power_plants_df[['name','latitude','longitude','primary_fuel','capacity_mw','estimated_generation_gwh']]
power_plants_df = global_power_plants.sort_values('capacity_mw',ascending=False).reset_index()
global_power_plants["capacity_factor"] = global_power_plants["estimated_generation_gwh"]/(global_power_plants["capacity_mw"]*24*365/1000)
global_power_plants[['name','owner','latitude','longitude','primary_fuel','capacity_mw','estimated_generation_gwh', 'capacity_factor']]
global_power_plants['prod_hrs_per_day']=global_power_plants.estimated_generation_gwh*1000/(global_power_plants.capacity_mw*365)
global_power_plants['prod_hrs_per_hour']=global_power_plants.estimated_generation_gwh*1000/(global_power_plants.capacity_mw*365)*24
global_power_plants['prod_hrs_per_minutes']=global_power_plants.estimated_generation_gwh*1000/(global_power_plants.capacity_mw*365)*24*60
global_power_plants[['name','owner','latitude','longitude','primary_fuel','capacity_mw','estimated_generation_gwh', 'capacity_factor', 'prod_hrs_per_day', 'prod_hrs_per_hour', 'prod_hrs_per_minutes']]
global_power_plants[['name','latitude','longitude','primary_fuel','capacity_mw','estimated_generation_gwh', 'capacity_factor', 'prod_hrs_per_day','prod_hrs_per_hour','prod_hrs_per_minutes']].groupby(['primary_fuel']).head(2)
global_power_plants.groupby(['primary_fuel']).agg({'capacity_mw' : ['nunique', 'sum', 'mean', 'max', 'min']}).reset_index()
global_power_plants.groupby(['primary_fuel']).agg({'estimated_generation_gwh': ['nunique', 'sum', 'mean', 'max', 'min']}).reset_index()
global_power_plants[['name','latitude','longitude','primary_fuel','capacity_mw','estimated_generation_gwh', 'owner']][power_plants_df['owner']=='PREPA'].sort_values('estimated_generation_gwh', ascending = False).groupby(['primary_fuel','capacity_mw']).head()
from ast import literal_eval



def get_lon_from_geo(str_):

    dict_ = literal_eval(str_)

    coordinates = dict_['coordinates']

    lon = coordinates[0]

    return lon



def get_lat_from_geo(str_):

    dict_ = literal_eval(str_)

    coordinates = dict_['coordinates']

    lat = coordinates[1]

    return lat
global_power_plants['lon'] = global_power_plants['.geo'].map(get_lon_from_geo)

global_power_plants['lat'] = global_power_plants['.geo'].map(get_lat_from_geo)

global_power_plants.drop(columns=['.geo'], inplace=True)
# https://stackoverflow.com/questions/9835762/how-do-i-find-the-duplicates-in-a-list-and-create-another-list-with-them

features = global_power_plants[['name','latitude','longitude','primary_fuel','capacity_mw','estimated_generation_gwh','gppd_idnr','name','owner','primary_fuel','source','url','wepp_id']].columns.values

unique_max = []

for feature in features:

    values = global_power_plants[feature].value_counts()

    unique_max.append([feature, values.max(), values.idxmax()])
np.transpose((pd.DataFrame(unique_max, columns=['Feature', 'Max duplicates', 'Value'])).sort_values(by = 'Max duplicates', ascending=False).head(50))
global_power_plants["gppd_idnr"].unique()
global_power_plants["name"].unique()
to_drop = ["generation_gwh_2013", "generation_gwh_2014", "generation_gwh_2015", "generation_gwh_2016","generation_gwh_2017", 

           "other_fuel1","other_fuel2","other_fuel3",

           "geolocation_source","year_of_capacity_data",

           "country", "country_long"]

power_plants_df = global_power_plants.drop(to_drop, axis=1)
def overlay_image_on_puerto_rico(file_name,band_layer,lat,lon,zoom):

    band = rio.open(file_name).read(band_layer)

    m = folium.Map([lat, lon], zoom_start=zoom)

    folium.raster_layers.ImageOverlay(

        image=band,

        bounds = [[18.6,-67.3,],[17.9,-65.2]],

        colormap=lambda x: (1, 0, 0, x),

    ).add_to(m)

    return m
import rasterio as rio

import folium

import tifffile as tiff



image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180701T161259_20180707T175356.tif'

latitude=18.1429005246921; longitude=-65.4440010699994

overlay_image_on_puerto_rico(image,band_layer=7,lat=latitude,lon=longitude,zoom=8)
average_no2_emission = [np.average(tiff.imread(image))]

print('Average NO2 emissions value: ', average_no2_emission)
image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180706T161914_20180712T200737.tif'

overlay_image_on_puerto_rico(image,band_layer=7,lat=latitude,lon=longitude,zoom=8)
# Likewise you might want to limit the data to only the region of interest

average_no2_emission = [np.average(tiff.imread(image))]

print('Average NO2 emissions value: ', average_no2_emission)
image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180704T165720_20180710T184641.tif'

overlay_image_on_puerto_rico(image,band_layer=7,lat=latitude,lon=longitude,zoom=8)
average_no2_emission = [np.average(tiff.imread(image))]

print('Average NO2 emissions value: ', average_no2_emission)
tiff.imread(image).shape
# only consider pollute fuel types

fossil_fuels = ['Coal', 'Oil', 'Gas']

fossil_fuel_df = global_power_plants[global_power_plants['primary_fuel'].isin(fossil_fuels)]

# sum the electricity generation

fossil_fuel_sum = fossil_fuel_df['estimated_generation_gwh'].sum()
# sum the pollution of the last satellite picture

sum_No2_emission = np.sum(tiff.imread(image)[:, :, 0 : 4])

# consider 14% of pollution is made from power plants electricity

sum_No2_emission_oe = sum_No2_emission * 0.14
# use the simplified emission factor formula

factor = sum_No2_emission_oe / fossil_fuel_sum

print(f'Simplified emissions factor for Puerto Rico is {factor} mol * h / m^2 * gw')
import glob



no2_path = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/*'

no2_pictures_path = glob.glob(no2_path)

len(no2_pictures_path)

print('We have {} pictures of the Copernicus Sentinel'.format(len(no2_pictures_path)))
from tqdm import tqdm_notebook as tqdm





# https://www.kaggle.com/ragnar123/exploratory-data-analysis-and-factor-model-idea

# this function will help us extract the no2 emission data in a tabular way

def read_s5p_no2_pictures_data(only_no2_emissions = True):

    s5p_no2_pictures = []

    for num, i in tqdm(enumerate(no2_pictures_path), total = 387):

        temp_s5p_no2_pictures = {'start_date': [], 'end_date': [], 'data': []}

        temp_s5p_no2_pictures['start_date'] = no2_pictures_path[num][76:84]

        temp_s5p_no2_pictures['end_date'] = no2_pictures_path[num][92:100]

        # only no2 emissions

        if only_no2_emissions:

            temp_s5p_no2_pictures['data'] = tiff.imread(i)[:, :, 0 : 4]

            temp_s5p_no2_pictures['no2_emission_sum'] = np.sum(tiff.imread(i)[:, :, 0 : 4])

            temp_s5p_no2_pictures['no2_emission_mean'] = np.average(tiff.imread(i)[:, :, 0 : 4])

            temp_s5p_no2_pictures['no2_emission_std'] = np.std(tiff.imread(i)[:, :, 0 : 4])

            temp_s5p_no2_pictures['no2_emission_max'] = np.max(tiff.imread(i)[:, :, 0 : 4])

            temp_s5p_no2_pictures['no2_emission_min'] = np.min(tiff.imread(i)[:, :, 0 : 4])

            s5p_no2_pictures.append(temp_s5p_no2_pictures)

        # all Copernicus data

        else:

            temp_s5p_no2_pictures['data'] = tiff.imread(i)

            s5p_no2_pictures.append(temp_s5p_no2_pictures)

    s5p_no2_pictures = pd.DataFrame(s5p_no2_pictures)

    s5p_no2_pictures['start_date'] = pd.to_datetime(s5p_no2_pictures['start_date'])

    s5p_no2_pictures['end_date'] = pd.to_datetime(s5p_no2_pictures['end_date'])

    s5p_no2_pictures.sort_values('start_date', inplace = True)

    s5p_no2_pictures.reset_index(drop = True, inplace = True)

    return s5p_no2_pictures



s5p_no2_pictures_df = read_s5p_no2_pictures_data()
s5p_no2_pictures_stats = s5p_no2_pictures_df[[col for col in s5p_no2_pictures_df.columns if col not in ['data']]]

s5p_no2_pictures_data = s5p_no2_pictures_df[['data']]

del s5p_no2_pictures_df

s5p_no2_pictures_stats.head()
def check_arrays(df, row = 1):

    band1 = pd.DataFrame(s5p_no2_pictures_data['data'][row][:, :, 0])

    band2 = pd.DataFrame(s5p_no2_pictures_data['data'][row][:, :, 1])

    band3 = pd.DataFrame(s5p_no2_pictures_data['data'][row][:, :, 2])

    band4 = pd.DataFrame(s5p_no2_pictures_data['data'][row][:, :, 3])

    

    def check_nan(df):

        df_nan = df.isnull().values.sum()

        return df_nan

    

    band1_nan = check_nan(band1)

    band2_nan = check_nan(band2)

    band3_nan = check_nan(band3)

    band4_nan = check_nan(band4)

    

    print('From row {} we have {} nan values for band1'.format(row, band1_nan))

    print('From row {} we have {} nan values for band2'.format(row, band2_nan))

    print('From row {} we have {} nan values for band3'.format(row, band3_nan))

    print('From row {} we have {} nan values for band4'.format(row, band4_nan))



    return band1, band2, band3, band4



band1, band2, band3, band4 = check_arrays(s5p_no2_pictures_data, row = 4)
# this function ignore nan values from the images

def read_s5p_no2_pictures_data_ignore_nan(only_no2_emissions = True):

    s5p_no2_pictures = []

    for num, i in tqdm(enumerate(no2_pictures_path), total = 387):

        temp_s5p_no2_pictures = {'start_date': [], 'end_date': [], 'data': []}

        temp_s5p_no2_pictures['start_date'] = no2_pictures_path[num][76:84]

        temp_s5p_no2_pictures['end_date'] = no2_pictures_path[num][92:100]

        # only no2 emissions

        if only_no2_emissions:

            temp_s5p_no2_pictures['data'] = tiff.imread(i)[:, :, 0 : 4]

            temp_s5p_no2_pictures['no2_emission_sum'] = np.nansum(tiff.imread(i)[:, :, 0 : 4])

            temp_s5p_no2_pictures['no2_emission_mean'] = np.nanmean(tiff.imread(i)[:, :, 0 : 4])

            temp_s5p_no2_pictures['no2_emission_std'] = np.nanstd(tiff.imread(i)[:, :, 0 : 4])

            temp_s5p_no2_pictures['no2_emission_max'] = np.nanmax(tiff.imread(i)[:, :, 0 : 4])

            temp_s5p_no2_pictures['no2_emission_min'] = np.nanmin(tiff.imread(i)[:, :, 0 : 4])

            s5p_no2_pictures.append(temp_s5p_no2_pictures)

        # all Copernicus data

        else:

            temp_s5p_no2_pictures['data'] = tiff.imread(i)

            s5p_no2_pictures.append(temp_s5p_no2_pictures)

    s5p_no2_pictures = pd.DataFrame(s5p_no2_pictures)

    s5p_no2_pictures['start_date'] = pd.to_datetime(s5p_no2_pictures['start_date'])

    s5p_no2_pictures['end_date'] = pd.to_datetime(s5p_no2_pictures['end_date'])

    s5p_no2_pictures.sort_values('start_date', inplace = True)

    s5p_no2_pictures.reset_index(drop = True, inplace = True)

    return s5p_no2_pictures



s5p_no2_pictures_df_ig_nan = read_s5p_no2_pictures_data_ignore_nan()
# https://stackoverflow.com/questions/14940743/selecting-excluding-sets-of-columns-in-pandas

s5p_no2_pictures_stats_ig_nan = s5p_no2_pictures_df_ig_nan[[col for col in s5p_no2_pictures_df_ig_nan.columns if col not in ['data']]]

del s5p_no2_pictures_df_ig_nan

s5p_no2_pictures_stats_ig_nan.head()
# Using plotly.express

import plotly.express as px

fig = px.line(s5p_no2_pictures_stats_ig_nan, x='start_date', y='no2_emission_sum')

fig.show()
import matplotlib.pyplot as plt

from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected = True)

import plotly.graph_objs as go



def line_plot_check_nan(df1, df2, x, y, title, width, height):

    

    trace1 = go.Scatter(

        x = df1[x],

        y = df1[y],

        mode='lines',

        name='with_nans',

        marker = dict(

            color = '#1E90FF', 

        ), 

    )

    

    df3 = df2.dropna()

    trace2 = go.Scatter(

        x = df3[x],

        y = df3[y],

        mode='markers',

        name='no_nans',

        marker = dict(

            color = 'red', 

        ), 

    )

    

    layout = go.Layout(

        title = go.layout.Title(

            text = title,

            x = 0.5

        ),

        font = dict(size = 14),

        width = width,

        height = height,

    )

    

    data = [trace1, trace2]

    fig = go.Figure(data = data, layout = layout)

    py.iplot(fig, filename = 'line_plot')

line_plot_check_nan(s5p_no2_pictures_stats_ig_nan, s5p_no2_pictures_stats, 'start_date', 'no2_emission_sum', 'NO2 emission by date', 1400, 600)
n_duplicates_dates = s5p_no2_pictures_stats_ig_nan.shape[0] - s5p_no2_pictures_stats_ig_nan.drop_duplicates(subset = ['start_date', 'end_date']).shape[0]

print(f'We have {n_duplicates_dates} duplicate days')
# this function will help us extract the no2 emission data in a tabular way

def read_s5p_no2_pictures_data_fill(only_no2_emissions = True):

    s5p_no2_pictures = []

    for num, i in tqdm(enumerate(no2_pictures_path), total = 387):

        temp_s5p_no2_pictures = {'start_date': [], 'end_date': [], 'data': []}

        temp_s5p_no2_pictures['start_date'] = no2_pictures_path[num][76:84]

        temp_s5p_no2_pictures['end_date'] = no2_pictures_path[num][92:100]

        # only no2 emissions

        if only_no2_emissions:

            image = tiff.imread(i)[:, :, 0 : 4]

            band1 = pd.DataFrame(image[: ,: , 0]).interpolate()

            band1.fillna(band1.mean(), inplace = True)

            band2 = pd.DataFrame(image[: ,: , 1]).interpolate()

            band2.fillna(band2.mean(), inplace = True)

            band3 = pd.DataFrame(image[: ,: , 2]).interpolate()

            band3.fillna(band3.mean(), inplace = True)

            band4 = pd.DataFrame(image[: ,: , 3]).interpolate()

            band4.fillna(band4.mean(), inplace = True)

            image = np.dstack((band1, band2, band3, band4))

            temp_s5p_no2_pictures['data'] = image

            temp_s5p_no2_pictures['no2_emission_sum'] = np.sum(image)

            temp_s5p_no2_pictures['no2_emission_mean'] = np.average(image)

            temp_s5p_no2_pictures['no2_emission_std'] = np.std(image)

            temp_s5p_no2_pictures['no2_emission_max'] = np.max(image)

            temp_s5p_no2_pictures['no2_emission_min'] = np.min(image)

            s5p_no2_pictures.append(temp_s5p_no2_pictures)

        # all Copernicus data

        else:

            temp_s5p_no2_pictures['data'] = tiff.imread(i)

            s5p_no2_pictures.append(temp_s5p_no2_pictures)

    s5p_no2_pictures = pd.DataFrame(s5p_no2_pictures)

    s5p_no2_pictures['start_date'] = pd.to_datetime(s5p_no2_pictures['start_date'])

    s5p_no2_pictures['end_date'] = pd.to_datetime(s5p_no2_pictures['end_date'])

    s5p_no2_pictures.sort_values('start_date', inplace = True)

    s5p_no2_pictures.reset_index(drop = True, inplace = True)

    return s5p_no2_pictures



s5p_no2_pictures_df_fill = read_s5p_no2_pictures_data_fill()
s5p_no2_pictures_stats_fill = s5p_no2_pictures_df_fill[[col for col in s5p_no2_pictures_df_fill.columns if col not in ['data']]]

del s5p_no2_pictures_df_fill

s5p_no2_pictures_stats_fill.head()
# drop nan values and check again for duplicate columns

s5p_no2_pictures_stats_fill = s5p_no2_pictures_stats_fill[s5p_no2_pictures_stats_fill['start_date']!='2019-04-15'].dropna()

# drop 2019-04-15 (probably an outlier or a rare event that can affect our factor calculation)

duplicate_columns = s5p_no2_pictures_stats_fill.shape[0] - s5p_no2_pictures_stats_fill.drop_duplicates(subset = ['start_date', 'end_date']).shape[0]

print(f'We have {duplicate_columns} duplicate columns')

print('We have {} days of data'.format(s5p_no2_pictures_stats_fill['start_date'].nunique()))
def line_plot(df, x, y, title, width, height):

    trace = go.Scatter(

        x = df[x],

        y = df[y],

        mode='lines',

        name='lines',

        marker = dict(

            color = '#1E90FF', 

        ), 

    )

    

    layout = go.Layout(

        title = go.layout.Title(

            text = title,

            x = 0.5

        ),

        font = dict(size = 14),

        width = width,

        height = height,

    )

    

    data = [trace]

    fig = go.Figure(data = data, layout = layout)

    py.iplot(fig, filename = 'line_plot')
line_plot(s5p_no2_pictures_stats_fill, 'start_date', 'no2_emission_sum', 'NO2 emission by date', 1400, 800)
# get the mean NO2 emission between 2018/07/01 and 2019/06/29

sum_no2_emission = s5p_no2_pictures_stats_fill['no2_emission_sum'].mean()

# consider 14% of pollution is made from power plants electricity

sum_no2_emission_oe = sum_no2_emission * 0.14

# use the simplified emission factor formula (sum of estimated generation from Caol, Oil and Gas plants)

factor = sum_no2_emission_oe / fossil_fuel_sum

print(f'Simplified emissions factor for Puerto Rico is {factor} mol * h / m^2 * gw')
weather_path_data = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/*'

weather_pictures_path = glob.glob(weather_path_data)

len(weather_pictures_path)

print('We have {} pictures of the global forecast system'.format(len(weather_pictures_path)))
tiff.imread(weather_pictures_path[0]).shape
# this function will help us extract weather pictures in a tabular way

def read_weather_data():

    weather_pictures = []

    for num, i in tqdm(enumerate(weather_pictures_path), total = len(weather_pictures_path)):

        temp_weather_pictures = {'date': [], 'temperature_2m_above_ground': [], 'specific_humidity_2m_above_ground': [], 'relative_humidity_2m_above_ground': [], 

                                 'u_component_of_wind_10m_above_ground': [], 'v_component_of_wind_10m_above_ground': [], 'total_precipitation_surface': []}

        temp_weather_pictures['date'] = weather_pictures_path[num][68:-6]

        temp_weather_pictures['date'] = weather_pictures_path[num][68:-6]

        image = tiff.imread(i)

        temp_weather_pictures['temperature_2m_above_ground'] = image[ : , : , 0]

        temp_weather_pictures['specific_humidity_2m_above_ground'] = image[ : , : , 1]

        temp_weather_pictures['relative_humidity_2m_above_ground'] = image[ : , : , 2]

        temp_weather_pictures['u_component_of_wind_10m_above_ground'] = image[ : , : , 3]

        temp_weather_pictures['v_component_of_wind_10m_above_ground'] = image[ : , : , 4]

        temp_weather_pictures['total_precipitation_surface'] = image[ : , : , 5]

        temp_weather_pictures['temperature_2m_above_ground_mean'] = np.average(image[ : , : , 0])

        temp_weather_pictures['specific_humidity_2m_above_ground_mean'] = np.average(image[ : , : , 1])

        temp_weather_pictures['relative_humidity_2m_above_ground_mean'] = np.average(image[ : , : , 2])

        temp_weather_pictures['u_component_of_wind_10m_above_ground_mean'] = np.average(image[ : , : , 3])

        temp_weather_pictures['v_component_of_wind_10m_above_ground_mean'] = np.average(image[ : , : , 4])

        temp_weather_pictures['total_precipitation_surface_mean'] = np.average(image[ : , : , 5])

        

        weather_pictures.append(temp_weather_pictures)

    

    weather_pictures = pd.DataFrame(weather_pictures)

    weather_pictures['date'] = pd.to_datetime(weather_pictures['date'], infer_datetime_format  = True)

    weather_pictures.sort_values('date', inplace = True)

    weather_pictures.reset_index(drop = True, inplace = True)

    return weather_pictures



weather_pictures_df = read_weather_data()
weather_pictures_df.head()
# check missing values

image_columns = ['temperature_2m_above_ground', 'specific_humidity_2m_above_ground', 'relative_humidity_2m_above_ground', 

               'u_component_of_wind_10m_above_ground', 'v_component_of_wind_10m_above_ground', 'total_precipitation_surface']

weather_pictures_df[[col for col in weather_pictures_df.columns if col not in image_columns]].isnull().sum()
weather_pictures_df_stats = weather_pictures_df[[col for col in weather_pictures_df.columns if col not in image_columns]]

n_duplicates = weather_pictures_df_stats.shape[0] - weather_pictures_df_stats['date'].nunique()

print(f'We have {n_duplicates} observations that belongs to a date with one or more records')
weather_pictures_df_stats = weather_pictures_df_stats.groupby('date').mean().reset_index()

print('We have data for {} days'.format(weather_pictures_df_stats['date'].nunique()))

print('Our data start on {} and finish in {}'.format(weather_pictures_df_stats['date'].min(), weather_pictures_df_stats['date'].max()))

line_plot(weather_pictures_df_stats, 'date', 'temperature_2m_above_ground_mean', 'Temperature by Date', 1400, 800)
no2_weather = s5p_no2_pictures_stats_fill[['start_date', 'no2_emission_sum']].merge(weather_pictures_df_stats, left_on = 'start_date', right_on = 'date', how = 'left')

no2_tem_corr = no2_weather[['no2_emission_sum', 'temperature_2m_above_ground_mean']].corr().loc['no2_emission_sum', 'temperature_2m_above_ground_mean']

print(f'NO2 and temeprature have a correlation of: {no2_tem_corr}')
no2_weather.columns
line_plot(weather_pictures_df_stats, 'date', 'specific_humidity_2m_above_ground_mean', 'Specific Humidity by Date', 1400, 800)
line_plot(weather_pictures_df_stats, 'date', 'relative_humidity_2m_above_ground_mean', 'Relative Humidity by Date', 1400, 800)
line_plot(weather_pictures_df_stats, 'date', 'u_component_of_wind_10m_above_ground_mean', 'U Component of Wind by Date', 1400, 800)
line_plot(weather_pictures_df_stats, 'date', 'v_component_of_wind_10m_above_ground_mean', 'V Component of Wind by Date', 1400, 800)
line_plot(weather_pictures_df_stats, 'date', 'total_precipitation_surface_mean', 'Total Precipitation Surface by Date', 1400, 800)
plt.figure(figsize = (14, 8))

sns.heatmap(no2_weather.corr(), annot = True, cmap = 'coolwarm')
reg_dataset = no2_weather[['date', 'temperature_2m_above_ground_mean', 'specific_humidity_2m_above_ground_mean', 'relative_humidity_2m_above_ground_mean', 'u_component_of_wind_10m_above_ground_mean', 

                               'v_component_of_wind_10m_above_ground_mean', 'total_precipitation_surface_mean', 'no2_emission_sum']]

reg_dataset['month'] = reg_dataset['date'].dt.month

reg_dataset['dayofweek'] = reg_dataset['date'].dt.dayofweek



reg_dataset = pd.get_dummies(reg_dataset, columns = ['dayofweek'])

reg_dataset['no2_emission_sum_t1'] = reg_dataset['no2_emission_sum'].shift(1)

reg_dataset['no2_emission_sum_t2'] = reg_dataset['no2_emission_sum'].shift(2)

reg_dataset['no2_emission_sum_t3'] = reg_dataset['no2_emission_sum'].shift(3)

reg_dataset['no2_emission_rolling_mean_t1t3'] = (reg_dataset['no2_emission_sum_t1'] + reg_dataset['no2_emission_sum_t2'] + reg_dataset['no2_emission_sum_t3']) / 3
reg_dataset.head()
reg_dataset.mean()
reg_dataset.fillna(reg_dataset.mean()).head(5)
!pip install dabl
import dabl

from dabl import plot



plot(reg_dataset, 'no2_emission_sum')

plt.show()
ec = dabl.SimpleClassifier(random_state=0).fit(reg_dataset, target_col="month")
reg_dataset_df = reg_dataset[['date','no2_emission_sum']]



#Set Date column as the index column.

reg_dataset_df.set_index('date', inplace=True)

reg_dataset_df.head()
plt.figure(figsize=(15, 5))

plt.ylabel("no2_emission_sum")

plt.xlabel("date")

plt.plot(reg_dataset_df)

plt.show()