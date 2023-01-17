import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import rasterio as rio

import folium

import tifffile as tiff 



import seaborn as sns



import datetime as dt

from datetime import datetime 



from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN

import math



import folium

import rasterio as rio



import warnings

warnings.filterwarnings('ignore')
global_power_plants = pd.read_csv('../input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')
def read_im(file,dir_path):

    return rio.open(dir_path+file)



def mean_yearly_image(band):

    dir_s5p_no2 = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/'

    all_data = [read_im(file,dir_s5p_no2).read(band) for file in os.listdir(dir_s5p_no2)]

    return np.array(all_data)



all_im_no2 = mean_yearly_image(1)

all_im_tropono2 = mean_yearly_image(2)

all_im_stratono2 = mean_yearly_image(3)

all_im_slantno2 = mean_yearly_image(4)



all_im_troppresure = mean_yearly_image(5)

all_im_absindex= mean_yearly_image(6)

all_im_cloudfraction= mean_yearly_image(7)
#Upside Down NO2 image file

##because fgs image and NO2 image is oposition latitude



all_im_no2=np.flip(all_im_no2,axis=1)

all_im_tropono2=np.flip(all_im_tropono2,axis=1)

all_im_stratono2=np.flip(all_im_stratono2,axis=1)

all_im_slantno2 = np.flip(all_im_slantno2,axis=1)

all_im_troppresure =np.flip(all_im_troppresure,axis=1)

all_im_absindex= np.flip(all_im_absindex,axis=1)

all_im_cloudfraction= np.flip(all_im_cloudfraction,axis=1)
#NO2 density on Puerto Rico area

sns.heatmap(np.nanmean(all_im_no2,axis=0),square=True,cmap='inferno',cbar_kws={'shrink':0.7})
sns.heatmap(np.nanmean(all_im_cloudfraction,axis=0),square=True,cmap='inferno',cbar_kws={'shrink':0.7})
def read_im(file,dir_path):

    return rio.open(dir_path+file)



def mean_yearly_image(band):

    dir_gfs = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/'

    all_data = [read_im(file,dir_gfs).read(band) for file in os.listdir(dir_gfs)]

    return np.array(all_data)



all_im_temp = mean_yearly_image(1)

all_im_humidity = mean_yearly_image(3)

all_im_u_wind = mean_yearly_image(4)

all_im_v_wind = mean_yearly_image(5)
#Create Global Power Plant data

#new column "latitude", "longitude"



global_power_plants['latitude']=0

global_power_plants['longitude']=0

for i in range(0, len(global_power_plants)):

    global_power_plants['latitude'][i]=global_power_plants['.geo'][i][50:66]

    global_power_plants['longitude'][i]=global_power_plants['.geo'][i][31:48]

    

global_power_plants['latitude'] = global_power_plants['latitude'].astype(float)

a = np.array(global_power_plants['latitude'].values.tolist()) # 18 instead of 8

global_power_plants['latitude'] = np.where(a < 10, a+10, a).tolist() 



#Delete useless columns



power_plants=global_power_plants.drop('generation_gwh_2013',axis=1)

power_plants=power_plants.drop('generation_gwh_2014',axis=1)

power_plants=power_plants.drop('generation_gwh_2015',axis=1)

power_plants=power_plants.drop('generation_gwh_2016',axis=1)

power_plants=power_plants.drop('generation_gwh_2017',axis=1)

power_plants=power_plants.drop('other_fuel1',axis=1)

power_plants=power_plants.drop('other_fuel2',axis=1)

power_plants=power_plants.drop('other_fuel3',axis=1)

power_plants=power_plants.drop('year_of_capacity_data',axis=1)

power_plants=power_plants.drop('system:index',axis=1)

power_plants=power_plants.drop('country',axis=1)

power_plants=power_plants.drop('country_long',axis=1)

power_plants=power_plants.drop('geolocation_source',axis=1)

power_plants=power_plants.drop('gppd_idnr',axis=1)

power_plants=power_plants.drop('url',axis=1)

power_plants=power_plants.drop('wepp_id',axis=1)

power_plants=power_plants.drop('.geo',axis=1)



power_plants.head()
import rasterio as rio



gfs_file = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/gfs_2018070206.tif'



def preview_meta_data(file_name):

    with rio.open(file_name) as img_filename:

        print('Bounding Box:',img_filename.bounds)

        print('Coordinates of Top Left Corner: ',img_filename.transform * (0,0))

        print('Coordinates of Bottom Right Corner: ',img_filename.transform * (img_filename.width,img_filename.height))

        print(img_filename.index(-65.19081511310455, 18.564903861343627))

        print(img_filename.index(-67.32354977311168, 18.564903861343627))

        

preview_meta_data(gfs_file)
#1pixel is equal to ...

left=-67.32354977311168

right=-65.19005097332781

top=18.56520446703891

bottom=17.900451156790464



pixel_length=(right-left)/474*225216.070 #[m]　Distance is calculated by https://vldb.gsi.go.jp/sokuchi/surveycalc/surveycalc/bl2stf.html

pixel_width=(top-bottom)/147*73576.937 #[m]



pixel_area=pixel_length*pixel_width #[m^2]
file_name = gfs_file



with rio.open(file_name) as img_filename:

    power_plants['gfs_pixel_x']=img_filename.index(power_plants.longitude.astype(float), power_plants.latitude.astype(float))[0]

    power_plants['gfs_pixel_y']=img_filename.index(power_plants.longitude.astype(float), power_plants.latitude.astype(float))[1]

    

power_plants
#Check vieques power plant data, again

vieques_data=power_plants.loc[19:19]
#Transform NO2 image data and gfs image data to dataframe day by day



import os



no2_path='../input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/'

no2_list=os.listdir('../input/ds4g-environmental-insights-explorer/eie_data/s5p_no2')

no2_list.sort()



gfs_path='../input/ds4g-environmental-insights-explorer/eie_data/gfs/'

gfs_list=os.listdir('../input/ds4g-environmental-insights-explorer/eie_data/gfs/')

gfs_list.sort()
# Get date and time from image file name

from datetime import datetime, timedelta

datatime_no2=[datetime.strptime(i[:16],'s5p_no2_%Y%m%d') for i in no2_list]

datatime_gfs=[datetime.strptime(i,'gfs_%Y%m%d%H.tif') for i in gfs_list]
#Make daily data of Vieques Power Plant at fgs_pixel_x and y

#To compare the NO2 density far from power plant, make data at fgs pixel_x+10,y+10

#(because maily oposition direction of wind)



vieques_daily_no2=pd.DataFrame(datatime_no2)

vieques_daily_gfs=pd.DataFrame(datatime_gfs)



#Mean of NO2 density around Vieques Power Plant

vieques_daily_no2['tropono2']=all_im_tropono2[:,vieques_data.gfs_pixel_x,vieques_data.gfs_pixel_y]

vieques_daily_no2['no2']=all_im_no2[:,vieques_data.gfs_pixel_x,vieques_data.gfs_pixel_y]



vieques_daily_no2['no2_max']=all_im_no2[:,vieques_data.gfs_pixel_x,vieques_data.gfs_pixel_y]

vieques_daily_no2['no2_min']=all_im_no2[:,vieques_data.gfs_pixel_x,vieques_data.gfs_pixel_y]



for k in range(len(all_im_no2)):

    for i in range(-5,5):

        for j in range(-5,5):

            vieques_daily_no2.no2_max[k]=max(vieques_daily_no2.no2_max[k],all_im_no2[k,vieques_data.gfs_pixel_x+i,vieques_data.gfs_pixel_y+j])

            vieques_daily_no2.no2_min[k]=min(vieques_daily_no2.no2_min[k],all_im_no2[k,vieques_data.gfs_pixel_x+i,vieques_data.gfs_pixel_y+j])



vieques_daily_no2['stratono2']=all_im_stratono2[:,vieques_data.gfs_pixel_x,vieques_data.gfs_pixel_y]

vieques_daily_no2['slantno2']=all_im_slantno2[:,vieques_data.gfs_pixel_x,vieques_data.gfs_pixel_y]

vieques_daily_no2['troppresure']=all_im_troppresure[:,vieques_data.gfs_pixel_x,vieques_data.gfs_pixel_y]

vieques_daily_no2['absindex']=all_im_absindex[:,vieques_data.gfs_pixel_x,vieques_data.gfs_pixel_y]

vieques_daily_no2['cloudfraction']=all_im_cloudfraction[:,vieques_data.gfs_pixel_x,vieques_data.gfs_pixel_y]



#weather data at Vieques Power Plant

vieques_daily_gfs['temp']=all_im_temp[:,vieques_data.gfs_pixel_x,vieques_data.gfs_pixel_y]

vieques_daily_gfs['humidity']=all_im_humidity[:,vieques_data.gfs_pixel_x,vieques_data.gfs_pixel_y]

vieques_daily_gfs['u_wind']=all_im_u_wind[:,vieques_data.gfs_pixel_x,vieques_data.gfs_pixel_y]

vieques_daily_gfs['v_wind']=all_im_v_wind[:,vieques_data.gfs_pixel_x,vieques_data.gfs_pixel_y]



#Change column name to set_index

vieques_daily_gfs=vieques_daily_gfs.rename(columns={0:'gfsdate'})

vieques_daily_no2=vieques_daily_no2.rename(columns={0:'no2date'})
vieques_daily_no2.set_index('no2date',inplace=True)

vieques_daily_gfs.set_index('gfsdate',inplace=True)

vieques_daily_no2
vieques_daily_gfs
#There are four wether data each day(0:00,6:00,12:00,18:00), so select 12:00 data

vieques_daily_gfs=vieques_daily_gfs[vieques_daily_gfs.index.hour == 12]

vieques_daily_gfs
#To merge NO2 density data and weather data

vieques_daily_gfs['day']=vieques_daily_gfs.index

vieques_daily_no2['day']=vieques_daily_no2.index



vieques_daily_gfs['day']=vieques_daily_gfs.day.dt.date



vieques_daily_no2.day=vieques_daily_no2.day.astype('str')

vieques_daily_gfs.day=vieques_daily_gfs.day.astype('str')



vieques_daily=pd.merge(vieques_daily_no2,vieques_daily_gfs,on=['day'],how='inner')

vieques_daily=vieques_daily[['day','no2','no2_max','no2_min','tropono2','stratono2', 'slantno2', 'troppresure',

                             'absindex','cloudfraction','temp','humidity','u_wind','v_wind']]
#Make wind velocity data from u and v

vieques_daily['wind']=np.sqrt(vieques_daily.u_wind**2+vieques_daily.v_wind**2)
### Final dataframe for Vieques Power plant

vieques_daily
plt.figure(figsize=(20, 10))

sns.distplot(vieques_daily.no2)
sns.pairplot(vieques_daily)
plt.figure(figsize=(20, 10))

sns.jointplot('no2','temp',data=vieques_daily,xlim=(-0.0003,0.0003))
plt.figure(figsize=(20, 10))

sns.jointplot('no2','wind',data=vieques_daily,xlim=(-0.00003,0.0001))
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(18, 6))

plt.plot(vieques_daily.no2)

plt.show()
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(18, 6))

plt.plot(vieques_daily.wind)

plt.show()
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(18, 6))

plt.plot(vieques_daily.absindex)

plt.show()
vieques_daily.head()
vieques_daily['dif_no2']=vieques_daily.no2_max-vieques_daily.no2_min

vieques_daily
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20, 6))

plt.plot(vieques_daily.day,vieques_daily.dif_no2)

plt.xlim(210,220)

plt.show()
#To sumarize the daily data to the monthly data, 'day' object change to datetime

vieques_daily['day_pd']=pd.to_datetime(vieques_daily.day,format='%Y-%m-%d')
vieques_daily.set_index('day_pd',inplace=True)

vieques_daily
vieques_month=vieques_daily.resample(rule="M").sum()

vieques_month
sns.pairplot(vieques_month)
vieques_month['emission']=vieques_month.dif_no2*pixel_area*46.0055/1000000

vieques_month
power_plants.loc[19:19]
EF_year = sum(vieques_month.emission)/power_plants.loc[19:19].estimated_generation_gwh/24/365

print("EF_year = {}" .format(EF_year[19])) 
vieques_month['EF_month']=EF_year[19]*vieques_month.emission/sum(vieques_month.emission)

vieques_month