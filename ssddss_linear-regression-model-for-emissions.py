# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import os

import folium

import rasterio as rio

import tifffile as tiff

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.metrics import mean_squared_error



print(os.listdir('/kaggle/input/ds4g-environmental-insights-explorer/eie_data'))

# Any results you write to the current directory are saved as output.

eie_data_path = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data'
gpp_df = pd.read_csv(eie_data_path+'/gppd/gppd_120_pr.csv')

gpp_df.head()
#code source: https://www.kaggle.com/paultimothymooney/overview-of-the-eie-analytics-challenge

def plot_points_on_map(dataframe,begin_index,end_index,latitude_column,latitude_value,longitude_column,longitude_value,zoom):

    df = dataframe[begin_index:end_index]

    location = [latitude_value,longitude_value]

    plot = folium.Map(location=location,zoom_start=zoom)

    for i in range(0,len(df)):

        popup = folium.Popup(str(df.primary_fuel[i:i+1]))

        folium.Marker([df[latitude_column].iloc[i],df[longitude_column].iloc[i]],popup=popup).add_to(plot)

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



def plot_scaled(file_name):

    vmin, vmax = np.nanpercentile(file_name, (5,95))  # 5-95% stretch

    img_plt = plt.imshow(file_name, cmap='gray', vmin=vmin, vmax=vmax)

    plt.show()



def split_column_into_new_columns(dataframe,column_to_split,new_column_one,begin_column_one,end_column_one):

    for i in range(0, len(dataframe)):

        dataframe.loc[i, new_column_one] = dataframe.loc[i, column_to_split][begin_column_one:end_column_one]

    return dataframe



gpp_df = split_column_into_new_columns(gpp_df,'.geo','latitude',50,66)

gpp_df = split_column_into_new_columns(gpp_df,'.geo','longitude',31,48)

gpp_df['latitude'] = gpp_df['latitude'].astype(float)

a = np.array(gpp_df['latitude'].values.tolist()) # 18 insted of 8

gpp_df['latitude'] = np.where(a < 10, a + 10, a).tolist()

lat = 18.200178; lon = -66.664513

plot_points_on_map(gpp_df, 0, 425, 'latitude', lat, 'longitude', lon, 9)
print('There are ', gpp_df.shape[0],' power plants')
gpp_df.head()

#For the purposes of the model, we will use estimated power generation in gwh as a feature

total_power_generation_2017 = gpp_df['estimated_generation_gwh'].sum()

print('The total estimated power generation for all power plants in puerto for 2017 rico is: ', total_power_generation_2017, ' gwh')
no2_emissions_image = eie_data_path+'/s5p_no2/s5p_no2_20180701T161259_20180707T175356.tif'

latitude=18.1429005246921; longitude=-65.4440010699994

overlay_image_on_puerto_rico(no2_emissions_image,band_layer=1,lat=latitude,lon=longitude,zoom=8)
overlay_image_on_puerto_rico(no2_emissions_image,band_layer=2,lat=latitude,lon=longitude,zoom=8)
activity = total_power_generation_2017

#Using the emissions snapshot from above

no2_emissions_sum = tiff.imread(no2_emissions_image)[:,:,0:4].sum()

emissions_factor = (no2_emissions_sum ) / (total_power_generation_2017)

print("Emissions Factor for power plant energy generation activity : ", emissions_factor)
#Get the date information from each files name

def get_dates(file_path, data_source):

    if data_source == 's5p':

        fname_only = file_path.split('/')

        dates_only = (fname_only[-1].split('_')[2], fname_only[-1].split('_')[3])

        start_date = dates_only[0][:8]

        end_date = dates_only[1][:8]

        return start_date, end_date

    elif data_source == 'gfs':

        file_name = file_path.split('/')[-1]

        date_w_extension = file_name.split('_')[-1]

        date = date_w_extension.split('.')[0][:8]

        return date

    elif data_source == 'gldas':

        file_name = file_path.split('/')[-1]

        date = file_name.split('_')[1]

        return date

    

#Read the data from the files and assemble a dataframe 

def read_data(data_path,data_source):

    data = []

    for file in os.listdir(data_path):

        file_path = data_path+file

        img = tiff.imread(file_path)

        img_to_add = {}

        if data_source == 's5p':

            start_date, end_date = get_dates(file_path, 's5p')

            img_to_add['start_date'] = start_date

            img_to_add['end_date'] = end_date

            img_to_add['no2_emissions_mean'] = np.nanmean(img[:,:,0:4])

        elif data_source == 'gfs':

            img_to_add['date'] = get_dates(file_path, 'gfs')

            img_to_add['temp'] = img[:,:,0]

            img_to_add['specific_humidity'] = img[:,:,1]

            img_to_add['relative_humidity'] = img[:, :, 2]

            img_to_add['u_component_wind'] = img[:, :, 3]

            img_to_add['v_component_wind'] = img[:, :, 4]

            img_to_add['total_precipation'] = img[:, :, 5]

        elif data_source == 'gldas':

            img_to_add['date'] = get_dates(file_path, 'gldas')

            for band in range(1, 13):

                img_to_add['band'+str(band)] = rio.open(file_path).read(band)    

                

        data.append(img_to_add)

        data_df = pd.DataFrame(data)

        

        if data_source == 's5p':

            data_df['start_date'] = pd.to_datetime(data_df['start_date'])

            data_df['end_date'] = pd.to_datetime(data_df['end_date'])

            data_df.sort_values('start_date', inplace = True)

            data_df.reset_index(drop = True, inplace = True)

        else: 

            data_df['date'] = pd.to_datetime(data_df['date'])

            data_df.sort_values('date', inplace=True)

       

    return data_df
no2_path = eie_data_path+'/s5p_no2/'

print("Number of sentinel satellite pictures: ", len(os.listdir(no2_path)))
emissions_mean = read_data(no2_path, 's5p')

emissions_mean.head()
emissions_mean.plot.line(x='start_date', y='no2_emissions_mean')
gfs_path = eie_data_path + '/gfs/'

print("We have " , len(gfs_path), " pictures of the global forecast system")

weather_df = read_data(gfs_path, 'gfs')

weather_df.head()
def get_weather_feature_stats(df):

    for c in [col for col in df.columns if col not in ['date']]:

        df[c] = df[c].apply(np.mean)

    return df

weather_mean_stats = get_weather_feature_stats(weather_df.copy())

weather_mean_stats.head()
#As there are multiple weather measurements per day, group them by the date and take the mean

weather_features = weather_mean_stats.groupby('date').mean()

#Also group the emissions  by date 

emissions_mean = emissions_mean.groupby('start_date').mean()

#concatenate the mean emissions dataframe and the weather features

training_data = pd.concat([weather_features, emissions_mean], axis=1 , join='outer')

#Drop any NaN values

training_data.dropna(how='any',inplace=True)

training_data.head()
gldas_path = eie_data_path + '/gldas/'

print("We have ", len(os.listdir(gldas_path)), " pictures of land data")

gldas_df = read_data(gldas_path, 'gldas')

gldas_df.head()
def get_gldas_mean_stats(df):

    cols = [col for col in df.columns if col not in ['date']]

    print(cols)

    for c in cols:

        df[c] = df[c].apply(np.mean)

    return df

gldas_mean_stats = get_gldas_mean_stats(gldas_df.copy())

gldas_mean_stats['date'] = gldas_mean_stats['date'].apply(pd.Timestamp.date)

gldas_mean_stats = gldas_mean_stats.fillna(0)

gldas_mean_stats = gldas_mean_stats.groupby('date').mean()



training_data = pd.concat([training_data, gldas_mean_stats], axis=1 , join='outer')

training_data.dropna(how='any', inplace=True)

#Add the total power generation of all the power plants

training_data['total_power_generation'] = total_power_generation_2017

training_data.head()
ds = training_data.copy()

train_df = ds.iloc[:324]

test_df = ds[324:]

X = train_df[[col for col in ds.columns if col not in ['no2_emissions_mean']]]

Y = train_df['no2_emissions_mean']

model = LinearRegression().fit(X,Y)

test_X = test_df[[col for col in ds.columns if col not in ['no2_emissions_mean']]]

test_Y = test_df['no2_emissions_mean']

preds = model.predict(test_X)

preds = pd.Series(preds)
plt.plot(preds.index, preds.values, color='red')
plt.plot(test_Y.index, test_Y.values)
rms = np.sqrt(mean_squared_error(test_Y, preds))
print(rms)