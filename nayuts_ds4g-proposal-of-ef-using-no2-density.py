from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN

from datetime import datetime, timedelta

import folium

import glob

import math

import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import rasterio as rio

import seaborn as sns

from sklearn.cluster import KMeans

import tifffile as tiff 
#If you want to analyze other area, please change here!

LAT_MAX = 18.563112930177304

LAT_MIN = 17.903121359128956

LON_MAX = -65.19437297127342

LON_MIN = -67.32297404549217
def overlay_image_on_puerto_rico_df(df, img, zoom):

    """

    show image on google map with marker of power plants.

    """

    lat_map=df.iloc[[0]].loc[:,["latitude"]].iat[0,0]

    lon_map=df.iloc[[0]].loc[:,["longitude"]].iat[0,0]

    m = folium.Map([lat_map, lon_map], zoom_start=zoom, tiles= 'Stamen Terrain')

    color={ 'Hydro' : 'lightblue', 'Solar' : 'orange', 'Oil' : 'darkblue', 'Coal' : 'black', 'Gas' : 'lightgray', 'Wind' : 'green' }

    folium.raster_layers.ImageOverlay(

        image=img,

        bounds = [[LAT_MAX,LON_MIN,],[LAT_MIN,LON_MAX]],

        #bounds = [[18.56,-67.32,],[17.90,-65.194]],

        colormap=lambda x: (1, 0, 0, x),

    ).add_to(m)

    

    for i in range(0,len(df)):

        popup = folium.Popup(str(df.primary_fuel[i:i+1]))

        folium.Marker([df["latitude"].iloc[i],df["longitude"].iloc[i]],

                     icon=folium.Icon(icon_color='red',icon ='bolt',prefix='fa',color=color[df.primary_fuel.iloc[i]])).add_to(m)

        

    return m
def split_column_into_new_columns(dataframe,column_to_split,new_column_one,begin_column_one,end_column_one):

    """

    Add latitude and longtitude to dataframe.

    """

    for i in range(0, len(dataframe)):

        dataframe.loc[i, new_column_one] = dataframe.loc[i, column_to_split][begin_column_one:end_column_one]

    return dataframe
#import Global Power Plant Database data

power_plants = pd.read_csv('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')



#Make latitude and longitude data easier to use

power_plants = split_column_into_new_columns(power_plants,'.geo','latitude',50,66)

power_plants = split_column_into_new_columns(power_plants,'.geo','longitude',31,48)

power_plants['latitude'] = power_plants['latitude'].astype(float)

a = np.array(power_plants['latitude'].values.tolist()) # 18 instead of 8

power_plants['latitude'] = np.where(a < 10, a+10, a).tolist() 



#sort data by their capacity

power_plants_df = power_plants.sort_values('capacity_mw',ascending=False).reset_index()
power_plants_df
#Check NO2_column_number_density of Sentinel-5P Data

image = tiff.imread('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180701T161259_20180707T175356.tif')

overlay_image_on_puerto_rico_df(power_plants_df,image[:,:,0],8)



##From https://www.kaggle.com/paultimothymooney/explore-image-metadata-s5p-gfs-gldas

##You can check which bands correspond which properties. 

##Now, band1 is NO2_column_number_density.
#Calculate total estimated electricity generation(GWh)

quantity_of_electricity_generated = np.sum(power_plants_df['estimated_generation_gwh'])

print('Quanity of Electricity Generated: ', quantity_of_electricity_generated)
#import path of Sentinel-5P Data

s5p_no2_timeseries = glob.glob('../input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/*')
s5p_no2_timeseries_no_duplication = []

checked_date = []



for data in sorted(s5p_no2_timeseries):

     

    data_date =  datetime.strptime(data[:79], '../input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_%Y%m%d')

    data_date = data_date.strftime("%Y/%m%d")

    

    if not data_date in checked_date:

        checked_date.append(data_date)

        s5p_no2_timeseries_no_duplication.append(data)



#Data path without duplicates

s5p_no2_timeseries = s5p_no2_timeseries_no_duplication
#Divide the data by month.

data_monthly_divided = {}

for data in s5p_no2_timeseries:

     

    data_date =  datetime.strptime(data[:77], '../input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_%Y%m')

    data_date = data_date.strftime("%Y/%m")

    

    if not data_date in data_monthly_divided.keys():

        data_monthly_divided[data_date] = []

        

    data_monthly_divided[data_date].append(data)
month = []

emissions = []



for key in sorted(data_monthly_divided.keys()):

    total_emissions = []

    datas = data_monthly_divided[key]

    

    for data in datas:        

        img = tiff.imread(data)[:,:,0] #import data here

        img = np.nan_to_num(img, nan=np.nanmean(img))  #fill nan by average  

        total_emission = np.nansum(img)  #take total NO2 density of the data

        

        total_emissions.append(total_emission)

    

    #take monthly total density.

    month.append(key)

    emissions.append(np.nansum(total_emissions))
#calculate amount of NO2.

#amount[T] = density[mol/m^2] * 0.25m^2 * number of whole pixels * 46.0055[g/mol] * 1e-6



emissions = np.array(emissions) * ((0.25 * img.shape[0]*img.shape[1]) * 46.0055 *1e-6)
results_monthly = pd.DataFrame(columns=['month', 'emission','emisson factor'])

results_monthly = pd.DataFrame({'emission':emissions,

                       'emission factor':emissions/(quantity_of_electricity_generated)}, #devide emissions by estimated generation

                    index=month)
results_monthly.head()
fig = plt.figure(figsize=(30, 4))

ax = results_monthly["emission factor"].plot()

plt.title('Monthly Emissions Factor in Puerto Rico')

ax.set(xlabel='YYYY/mm', ylabel='Emission factor [T/GWh]')
#Divide the data by minimal time span.

data_minimal_divided = {}

for data in s5p_no2_timeseries:

     

    data_date =  datetime.strptime(data[:79], '../input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_%Y%m%d')

    data_date = data_date.strftime("%Y/%m/%d")

    

    if not data_date in data_minimal_divided.keys():

        data_minimal_divided[data_date] = [data]
span = []

emissions = []



for key in sorted(data_minimal_divided.keys()):

    total_emissions = []

    datas = data_minimal_divided[key]

    for data in datas:

        

        img = tiff.imread(data)[:,:,0]

        img = np.nan_to_num(img, nan=np.nanmean(img))

        total_emission = np.nansum(img)

        

        total_emissions.append(total_emission)

    

    span.append(key)

    emissions.append(np.nansum(total_emissions))
#calculate amount of NO2.

#amount[T] = density[mol/m^2] * 0.25m^2 * number of whole pixels * 46.0055[g/mol] * 1e-6

emissions = np.array(emissions) * ((0.25 * img.shape[0]*img.shape[1]) * 46.0055 *1e-6)
results_minimal = pd.DataFrame(columns=['week', 'emission','emisson factor'])

results_minimal = pd.DataFrame({'emission':emissions,

                       'emission factor':emissions/(quantity_of_electricity_generated)},

                    index=span)
results_minimal.head()
fig = plt.figure(figsize=(30, 4))

ax = results_minimal["emission factor"].plot()

plt.title('Monthly Mean Simplified Emissions Factor in Puerto Rico')

ax.set(xlabel='YYYY/mm/dd', ylabel='Emission factor [T/GWh]')
print( "Total NO2 emissions in Puerto Rico is", int(np.sum(emissions) * 12), "T/year")
#From https://www.kaggle.com/ajulian/capacity-factor-in-power-plants



total_capacity_mw = power_plants_df['capacity_mw'].sum()

print('Total Installed Capacity: '+'{:.2f}'.format(total_capacity_mw) + ' MW')

capacity = (power_plants_df.groupby(['primary_fuel'])['capacity_mw'].sum()).to_frame()

capacity = capacity.sort_values('capacity_mw',ascending=False)

capacity['percentage_of_total'] = (capacity['capacity_mw']/total_capacity_mw)*100

ax = capacity.sort_values(by='percentage_of_total', ascending=True)['percentage_of_total'].plot(kind='bar',color=['lightblue', 'green', 'orange', 'black','lightgray','darkblue'])

ax.set(ylabel='percentage')

image = tiff.imread('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180701T161259_20180707T175356.tif')

monotonous = KMeans(n_clusters=3, random_state=6).fit_predict(image[:,:,0].reshape(-1, 1))

overlay_image_on_puerto_rico_df(power_plants_df, monotonous.reshape((148, 475)), 8)
gldas_files = glob.glob('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gldas/*')

gldas_files = sorted(gldas_files)

gfs_files = glob.glob('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/*')

gfs_files = sorted(gfs_files)
#Data are separated by day for the future, but it is not necessary.

gldas_files_par_day = []

for i in range(0,len(gldas_files[6:54]),8):

    gldas_files_par_day.append(gldas_files[i:i+8])
#Data are separated by day for the future, but it is not necessary.

gfs_files_par_day = []

for i in range(0,len(gfs_files[3:27]),4):

    #print(gfs_files[i:i+4])

    gfs_files_par_day.append(gfs_files[i:i+4])
ave_wind_u = []

ave_wind_v = []

ave_wind_speed = []



#Get data of a day

for i in range(len(gfs_files_par_day)):

    gfs_tmp = gfs_files_par_day[i]

    gldas_tmp = gldas_files_par_day[i]

    array_wind_u = []

    array_wind_v = []

    array_wind_speed = []

    

    #Get datas in the day

    for j in range(len(gfs_tmp)):

        gfs_image_u = tiff.imread(gfs_tmp[j])[:,:,3]

        gfs_image_v = tiff.imread(gfs_tmp[j])[:,:,4]

        gldas_image1 = tiff.imread(gldas_tmp[2*j])[:,:,11]

        gldas_image2 = tiff.imread(gldas_tmp[2*j + 1])[:,:,11]



        #fill na by mean

        gfs_image_u = np.nan_to_num(gfs_image_u, nan=np.nanmean(gfs_image_u))

        gfs_image_v = np.nan_to_num(gfs_image_v, nan=np.nanmean(gfs_image_v))

        gldas_image1 = np.nan_to_num(gldas_image1, nan=np.nanmean(gldas_image1))

        gldas_image2 = np.nan_to_num(gldas_image2, nan=np.nanmean(gldas_image2))

        

        #GLDAS has twice detailed time span than GFS

        gldas_image = (gldas_image1 + gldas_image2)/2

        

        array_wind_u.append(gfs_image_u)

        array_wind_v.append(gfs_image_v)

        array_wind_speed.append(gldas_image)

    

#Calculate average        

ave_wind_u = np.nanmean(np.array(array_wind_u), axis=0)

ave_wind_v = np.nanmean(np.array(array_wind_v), axis=0)

ave_wind_speed = np.nanmean(np.array(array_wind_speed), axis=0)
lon = []

lat = []

NO2 = []

wind_u = []

wind_v = []

wind_speed = []



for i in range(image[:,:,0].shape[0]):

    for j in range(image[:,:,0].shape[1]):

        #print(image[:,:,0][i,j])

        NO2.append(image[:,:,0][i,j])

        lon.append(i)

        lat.append(j)

        wind_u.append(ave_wind_u.reshape((148, 475))[i,j])

        wind_v.append(ave_wind_v.reshape((148, 475))[i,j])

        wind_speed.append(ave_wind_speed.reshape((148, 475))[i,j])

        

NO2 = np.array(NO2)

lon = np.array(lon)

lat = np.array(lat)

wind_u = np.array(wind_u)

wind_v = np.array(wind_v)

wind_spped = np.array(wind_speed)

        

features_df = pd.DataFrame(columns=['NO2', 'lat', 'lon', 'wind_u', 'wind_v', 'wind_speed'])

features_df = pd.DataFrame({

                    'NO2': NO2/max(NO2),

                    'lat': lat/max(lat),

                    'lon': lon/max(lon),

                    'wind_u' : wind_u/(- min(wind_u)),

                    'wind_v' : wind_v/(- min(wind_v)),

                    'wind_speed': wind_speed/max(wind_speed)})
overlay_image_on_puerto_rico_df(power_plants_df, np.zeros((148, 475)), 8)
group_pred = KMeans(n_clusters=7, random_state=3).fit_predict(features_df)

plt.figure()

sns.heatmap(group_pred.reshape((148, 475)))
overlay_image_on_puerto_rico_df(power_plants_df, group_pred.reshape((148, 475)), 8)
def which_pixel_and_group(df,pred,img):

    """

    Add information (which pixel and class label) to input DataFrame

    

    Parameters

    ----------

    df : pandas.DataFrame

        This dataframe must have latitude and longitude.

    pred : numpy.array

        Label classified by k-means.

    img : numpy.array

        NO2 density.



    Returns

    -------

    df : pandas.DataFrame

        Information of powerplant added pixel and class label.

    """

    

    lat_pixel = []

    lon_pixel = []

    kmean_groups = []

    

    for i in range(len(df)):

        lat = float(df.iloc[[i]].loc[:,["latitude"]].iat[0,0])

        lon = float(df.iloc[[i]].loc[:,["longitude"]].iat[0,0])



    

        f_lat = (lat - LAT_MIN)*img.shape[0]/(LAT_MAX - LAT_MIN)

        f_lon = (lon + LON_MAX)*img.shape[1]/(-LON_MIN + LON_MAX)

        f_lat_int = int(Decimal(str(f_lat -1)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))

        f_lon_int = int(Decimal(str(f_lon -1)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))

        pixel = (f_lat_int - 1) * img.shape[1] + f_lon_int

        

        lat_pixel.append(img.shape[0] - f_lat_int)

        lon_pixel.append(f_lon_int)

        kmean_groups.append(pred[pixel])

 

        

    df["lat_pixel"] = lat_pixel

    df["lon_pixel"] = lon_pixel

    df["kmean_group"] = kmean_groups

    

    return df
def calc_gwh_in_group(df):

    """

    Add information (which pixel and class label) to input DataFrame

    

    Parameters

    ----------

    df : pandas.DataFrame

        This dataframe must have primary_fuel, kmean_group and estimated_generation_gwh.



    Returns

    -------

    df : pandas.DataFrame

        Information of powerplant added gwh_rate_group.

    """

    

    pplant_gwh_rate_ingroup = []

    

    for i in range(len(df)):

        

        #Exclude power plants that do not emit NO2

        if not df.iloc[[i]].loc[:,["primary_fuel"]].iat[0,0] in ["Oil","Gas", "Oil"]:

            pplant_gwh_rate_ingroup.append(0)

            

        else:      

            pplant_cap = df.iloc[[i]].loc[:,["capacity_mw"]].iat[0,0]

            pplant_group = df.iloc[[i]].loc[:,["kmean_group"]].iat[0,0]

        

            pplants_emitsno2_group = df[(df["kmean_group"]==pplant_group) | \

                                         power_plants_df["primary_fuel"].map(lambda primary_fuel: primary_fuel in ["Oil","Gas", "Oil"])]

        

            total_cap_ingroup = sum(pplants_emitsno2_group["capacity_mw"])

            pplant_gwh_rate_ingroup.append(pplant_cap/total_cap_ingroup)

        

    df["cap_rate_group"] = pplant_gwh_rate_ingroup

    

    return df
power_plants_df = which_pixel_and_group(power_plants_df, group_pred, image)

power_plants_df = calc_gwh_in_group(power_plants_df)
power_plants_df.loc[:,["name", "primary_fuel", "kmean_group","cap_rate_group"]].head()
def calc_no2amount_each_group(img, pred):

    """

    Calculate amount of substance of NO2 for each group.

    

    Parameters

    ----------

    img : numpy.array

        NO2 density.

    pred : numpy.array

        Label classified by k-means.



    Returns

    -------

    no2amount : dictionary

        value: group number, value: amount of substance of NO2(g)

    """

    no2amount = dict()

    

    for i in set(pred):

        no2amount[i] = 0

    

    pred = pred.reshape(img[:,:,0].shape)

    

    for i in range(img.shape[0]):

        for j in range(img.shape[1]):

            no2amount[pred[i,j]] += img[:,:,0][i,j] * 0.25 *48 #Simultaneous conversion from density to quantity and summation 

            

    return no2amount
def calc_own_ef(df, no2amount):

    """

    Calculate own emission factor of each power plant.

    

    Parameters

    ----------

    df : pandas.DataFrame

        This dataframe must have gwh_rate_group and kmean_group.

    no2amount : dictionary

        value: group number, value: amount of substance of NO2



    Returns

    -------

    df : pandas.DataFrame

        Information of powerplant added estimated merginal_EF and emission for one year.

    """

    own_efs = []

    own_emission = []

    

    for i in range(len(df)):

        pplant_cap_rate_group = df.iloc[[i]].loc[:,["cap_rate_group"]].iat[0,0]

        pplant_est_gen_gwh = df.iloc[[i]].loc[:,["estimated_generation_gwh"]].iat[0,0]

        pplant_kmean_group = df.iloc[[i]].loc[:,["kmean_group"]].iat[0,0]

        

        #I calculate emissions factor of each power plant here!

        #I assumed that one Sentinel-5P data coressponding to daily data.

        own_efs.append(no2amount[pplant_kmean_group] * pplant_cap_rate_group / pplant_est_gen_gwh) 

        own_emission.append(no2amount[pplant_kmean_group] * pplant_cap_rate_group) 

        

    df["own_EF"] = own_efs

    df["own_emission"] = own_emission

    

    return df
no2amount_dict_group = calc_no2amount_each_group(image, group_pred)



print("Coreration of 'group: Total NO2 amount of each group(g)' are following:")

no2amount_dict_group #Total NO2 amount of each group
power_plants_df = calc_own_ef(power_plants_df, no2amount_dict_group)

power_plants_df.loc[:,["name", "primary_fuel", 'estimated_generation_gwh','capacity_mw' , "kmean_group", "own_emission", "own_EF"]]
power_plants_group0_df = power_plants_df[power_plants_df["kmean_group"]==0]

power_plants_group0_df.loc[:,["name", "primary_fuel", "kmean_group", "own_emission", "own_EF"]]
total_capacity_mw_group0 = power_plants_group0_df['capacity_mw'].sum()

print('Total Installed Capacity: '+'{:.2f}'.format(total_capacity_mw_group0) + ' MW')

capacity = (power_plants_group0_df.groupby(['primary_fuel'])['capacity_mw'].sum()).to_frame()

capacity = capacity.sort_values('capacity_mw',ascending=False)

capacity['percentage_of_total'] = (capacity['capacity_mw']/total_capacity_mw_group0)*100

ax = capacity.sort_values(by='percentage_of_total', ascending=True)['percentage_of_total'].plot(kind='bar',color=['lightblue', 'green', 'orange', 'black','lightgray','darkblue'])

ax.set(ylabel='percentage')
capacity