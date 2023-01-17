import numpy as np

import pandas as pd

import rasterio as rio

import matplotlib.pyplot as plt

import seaborn as sns

import os

import folium

import tifffile as tiff

from datetime import datetime, timedelta

from branca.element import Template, MacroElement

from skimage.transform import resize

import branca

from tqdm import tqdm

from scipy.stats import pearsonr

import geopy.distance

from sklearn.preprocessing import normalize

import cv2

import warnings

from affine import Affine

from pyproj import Proj, transform
warnings.simplefilter('ignore')
s5p_no2_path = '../input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/'

gfs_path = '../input/ds4g-environmental-insights-explorer/eie_data/gfs/'



electricity = pd.read_csv('../input/ds4g-puerto-rico-electricity-consumption-by-month/EIA_puerto_rico_electric_consumption.csv')

month_av_elect = electricity.iloc[30:42]['all_sectors'].values

#sorting from Jan to Dec

month_av_elec = np.append(month_av_elect[6:],month_av_elect[:6])



gppd = pd.read_csv('../input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')

gppd = gppd.drop(['system:index', 'capacity_mw', 'country',

       'country_long', 'estimated_generation_gwh', 'generation_gwh_2013',

       'generation_gwh_2014', 'generation_gwh_2015', 'generation_gwh_2016',

       'generation_gwh_2017', 'geolocation_source', 'gppd_idnr',

       'source', 'url', 'wepp_id', 'year_of_capacity_data'], axis=1)

gppd['long'] = gppd['.geo'].apply(lambda x: float(x[31:48]))

gppd['lat'] = gppd['.geo'].apply(lambda x: float(x[50:66]))

gppd['lat'] = np.where(gppd['lat']<10, gppd['lat']+10, gppd['lat'])

gppd['Ix']=gppd.index

gppd['coords'] = list(zip(gppd.lat, gppd.long))



pop_map_file = '/kaggle/input/population/imageToDriveExample.tif'

pop_map = rio.open(pop_map_file).read(1)



eia = pd.read_csv('../input/eia923/PR_EIA_923.csv')
s5p_bounds = rio.open(s5p_no2_path+os.listdir(s5p_no2_path)[0]).bounds

gfs_bounds = rio.open(gfs_path+os.listdir(gfs_path)[0]).bounds

s5p_top, s5p_bot, s5p_lef, s5p_rit = s5p_bounds[1], s5p_bounds[3], s5p_bounds[0], s5p_bounds[2] #will be flipped while reading

gfs_top, gfs_bot, gfs_lef, gfs_rit = gfs_bounds[3], gfs_bounds[1], gfs_bounds[0], gfs_bounds[2]



pop_bounds = rio.open(pop_map_file).bounds

pop_top, pop_bot, pop_lef, pop_rit = pop_bounds[3], pop_bounds[1], pop_bounds[0], pop_bounds[2]



#get the x,y location on map image for each longitude, latitude

no2_latp = (gppd['lat']-s5p_top)/(s5p_bot-s5p_top)*100

no2_longp = (gppd['long']-s5p_lef)/(s5p_rit-s5p_lef)*100

no2_col = np.round(475*no2_longp/100).astype(int)

no2_row = np.round(148*no2_latp/100).astype(int)



pop_latp = (gppd['lat']-pop_top)/(pop_bot-pop_top)*100

pop_longp = (gppd['long']-pop_lef)/(pop_rit-pop_lef)*100

pop_col = np.round(256*pop_longp/100).astype(int)

pop_row = np.round(80*pop_latp/100).astype(int)



with rio.open(pop_map_file) as r:

    T0 = r.transform  # upper-left pixel corner affine transform

    p1 = Proj(r.crs)

    A = r.read()  # pixel values



cols, rows = np.meshgrid(np.arange(A.shape[2]), np.arange(A.shape[1]))



T1 = T0 * Affine.translation(0.5, 0.5)

# Function to convert pixel row/column index (from 0) to easting/northing at centre

rc2en = lambda r, c: (c, r) * T1



# All eastings and northings

eastings, northings = np.vectorize(rc2en, otypes=[np.float, np.float])(rows, cols)



# Project all longitudes, latitudes

p2 = Proj(proj='latlong',datum='WGS84')

longs, lats = transform(p1, p2, eastings, northings)
#color dictionary for map and plots

fuels = ['Coal','Gas','Hydro','Oil','Solar','Wind']

colors = ['black','gray','green','darkred','orange','blue']

fudic = {}

for k,v in zip(fuels, colors):

        fudic[k]=v
#no2 columns grouped by day of the week

week_no2 = {}

week_no2_trop = {}

week_no2_strat = {}

for i in range(7):

    week_no2[i]=[]

    week_no2_trop[i] = []

    week_no2_strat[i] = []

for file in tqdm(os.listdir(s5p_no2_path)):

    ao = datetime.strptime(file[8:16], '%Y%m%d')

    filr = rio.open(s5p_no2_path+file)

    cf = np.flipud(filr.read(7))

    no = np.flipud(filr.read(1))

    tno = np.flipud(filr.read(2))

    sno = np.flipud(filr.read(3))

    #no = np.where(cf>0.3,np.nan,no)

    #tno = np.where(cf>0.3,np.nan,tno)

    #sno = np.where(cf>0.3,np.nan,sno)

    week_no2[ao.weekday()].append(no)

    week_no2_trop[ao.weekday()].append(no)

    week_no2_strat[ao.weekday()].append(no)
all_yr_no2 = np.empty([0,148,475])

for i in range(7):

    all_yr_no2 = np.concatenate((all_yr_no2, np.array(week_no2[i])), axis=0)

annual_no2_avg = np.nanmean(all_yr_no2, axis=0)

cmap = (annual_no2_avg-annual_no2_avg.min())/(annual_no2_avg.max()-annual_no2_avg.min())
gppd['annual_no2_av']=np.nan

gppd['population_dens']=np.nan

for i in range(len(gppd)):

    plant=gppd.iloc[i]

    gppd.loc[i,'annual_no2_av']=annual_no2_avg[int(no2_row[i]),int(no2_col[i])]

    gppd.loc[i,'population_dens']=np.nanmean(pop_map[int(pop_row[i])-10:int(pop_row[i])+10,int(pop_col[i])-10:int(pop_col[i])+10])

#gppd['population_dens']=gppd['population_dens'].fillna(0)
gppd.groupby('primary_fuel')[['annual_no2_av','population_dens']].mean()
popn =normalize(np.nan_to_num(pop_map))
lat=(s5p_top+s5p_bot)/2; lon=(s5p_lef+s5p_rit)/2

m = folium.Map([lat, lon], zoom_start=9, height=350, width = 1000)

folium.raster_layers.ImageOverlay(

        image=popn,

        bounds = [[pop_bot,pop_lef],[pop_top,pop_rit]],

        colormap=lambda x: (1, 0, 1, x),

    ).add_to(m)

for i in range(len(gppd)):

    folium.Marker([gppd['lat'].iloc[i],gppd['long'].iloc[i]],icon=folium.Icon(color=fudic[gppd['primary_fuel'].iloc[i]])).add_to(m)

title_html = '''

             <h3 align="center" style="font-size:20px"><b>Power Plants in Puerto Rico with Population Density</b></h3>

             '''

m.get_root().html.add_child(folium.Element(title_html))

m
lat=(s5p_top+s5p_bot)/2; lon=(s5p_lef+s5p_rit)/2

m = folium.Map([lat, lon], zoom_start=9, height=350, width = 1000)

folium.raster_layers.ImageOverlay(

        image=cmap,

        bounds = [[s5p_bot,s5p_lef],[s5p_top,s5p_rit]],

        colormap=lambda x: (1, 0, 0, x),

    ).add_to(m)

popup = folium.Popup(str(gppd.primary_fuel[0:1]))

for i in range(len(gppd)):

    folium.Marker([gppd['lat'].iloc[i],gppd['long'].iloc[i]],icon=folium.Icon(color=fudic[gppd['primary_fuel'].iloc[i]])).add_to(m)



colormap = branca.colormap.LinearColormap([(255,255,255),(255,0,0)], vmin=0, vmax=annual_no2_avg.max(), caption='NO2 Column')
template = """

{% macro html(this, kwargs) %}

<!doctype html><html lang="en"><head> <meta charset="utf-8"> <meta name="viewport" content="width=device-width, initial-scale=1"> <title>jQuery UI Draggable - Default functionality</title> <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css"> <script src="https://code.jquery.com/jquery-1.12.4.js"></script> <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script></head><body><div id='maplegend' class='maplegend' style='position: absolute; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 0.8); border-radius:6px; padding: 10px; font-size:14px; right: 20px; top: 20px;'><div class='legend-scale'> <ul class='legend-labels'> <li><span style='background:black'></span>Coal</li><li><span style='background:darkred'></span>Oil</li><li><span style='background:#585858'></span>Gas</li><li><span style='background:#4CAF50'></span>Hydro</li><li><span style='background:#33adff'></span>Wind</li><li><span style='background:orange'></span>Solar</li></ul></div></div></body></html>

<style type='text/css'>

.maplegend .legend-scale ul{margin:0;margin-bottom:5px;padding:0;float:left;list-style:none}.maplegend .legend-scale ul li{font-size:80%;list-style:none;margin-left:0;line-height:18px;margin-bottom:2px}.maplegend ul.legend-labels li span{display:block;float:left;height:16px;width:30px;margin-right:5px;margin-left:0}.maplegend .legend-source{font-size:80%;color:#777;clear:both}.maplegend a{color:#777}

</style>

{% endmacro %}"""



macro = MacroElement()

macro._template = Template(template)

title_html = '''

             <h3 align="center" style="font-size:20px"><b>Power Plants in Puerto Rico</b></h3>

             '''

m.get_root().html.add_child(folium.Element(title_html))

colormap.add_to(m)

m.get_root().add_child(macro)
p = resize(pop_map, (148,475))

fig, ax = plt.subplots(1,2, figsize=(18,12))

ax[0].axhline(y=0, color='k',linewidth=1)

ax[0].axhline(y=147, color='k',linewidth=1)

ax[0].axvline(x=0, color='k',linewidth=1)

ax[0].axvline(x=474, color='k',linewidth=1)

sns.heatmap(p,ax=ax[0],square=True,cmap=sns.color_palette("Blues"), cbar_kws={"orientation": "horizontal",'shrink':0.5})

sns.heatmap(cmap, ax=ax[1], square=True, cmap='Reds',cbar_kws={"orientation": "horizontal",'shrink':0.5})

sns.heatmap(p,ax=ax[1],square=True,cmap=sns.color_palette("Blues"), cbar=False, alpha=0.2)

for fuel in fuels:

    i = gppd[gppd['primary_fuel']==fuel].index

    plt.scatter(no2_col[i], no2_row[i], color=fudic[fuel], label=fuel)    

#ax[0].axis('off')

#ax[1].axis('off')

ax[0].set_title('Population Density')

ax[1].set_title('Relative NO2 Concentration with Population Density')

#ax.legend()
weekend = np.concatenate((np.array(week_no2[5]),np.array(week_no2[6])), axis=0)

weekday = np.concatenate((np.array(week_no2[0]),np.array(week_no2[1]),np.array(week_no2[2]),np.array(week_no2[3]),np.array(week_no2[4])), axis=0)

weekend_trop = np.concatenate((np.array(week_no2_trop[5]),np.array(week_no2_trop[6])), axis=0)

weekday_trop = np.concatenate((np.array(week_no2_trop[0]),np.array(week_no2_trop[1]),np.array(week_no2_trop[2]),np.array(week_no2_trop[3]),np.array(week_no2_trop[4])), axis=0)

weekend_strat = np.concatenate((np.array(week_no2_strat[5]),np.array(week_no2_strat[6])), axis=0)

weekday_strat = np.concatenate((np.array(week_no2_strat[0]),np.array(week_no2_strat[1]),np.array(week_no2_strat[2]),np.array(week_no2_strat[3]),np.array(week_no2_strat[4])), axis=0)
weekendav = np.nanmean(weekend, axis=0)

weekdayav = np.nanmean(weekday, axis=0)

weekendav = np.where(np.isnan(p), np.nan,weekendav)

weekdayav = np.where(np.isnan(p), np.nan,weekdayav)



weekendav_trop = np.nanmean(weekend_trop, axis=0)

weekdayav_trop = np.nanmean(weekday_trop, axis=0)

weekendav_trop = np.where(np.isnan(p), np.nan,weekendav_trop)

weekdayav_trop = np.where(np.isnan(p), np.nan,weekdayav_trop)



weekendav_strat = np.nanmean(weekend_strat, axis=0)

weekdayav_strat = np.nanmean(weekday_strat, axis=0)

weekendav_strat = np.where(np.isnan(p), np.nan,weekendav_strat)

weekdayav_strat = np.where(np.isnan(p), np.nan,weekdayav_strat)



fig, ax = plt.subplots(1,2, figsize=(16,10))

ax = ax.ravel()

sns.heatmap(weekdayav,ax=ax[0],square=True,cmap='inferno',cbar_kws={"orientation": "horizontal"}, vmax=6.518138194350083e-05, vmin=4.5764838692296064e-05)

sns.heatmap(weekendav,ax=ax[1],square=True,cmap='inferno',cbar_kws={"orientation": "horizontal"}, vmax=6.518138194350083e-05, vmin=4.5764838692296064e-05)

[x.axis('off') for x in ax]



ax[0].set_title('Weekday Total NO2')

ax[1].set_title('Weekend Total NO2')



fig, ax = plt.subplots(1,2, figsize=(16,10))

ax = ax.ravel()

sns.heatmap(weekdayav_trop,ax=ax[0],square=True,cmap='inferno',cbar_kws={"orientation": "horizontal"}, vmax=np.nanmax(weekdayav_trop), vmin=np.nanmin(weekendav_trop))

sns.heatmap(weekendav_trop,ax=ax[1],square=True,cmap='inferno',cbar_kws={"orientation": "horizontal"}, vmax=np.nanmax(weekdayav_trop), vmin=np.nanmin(weekendav_trop))

[x.axis('off') for x in ax]



ax[0].set_title('Weekday Tropospheric NO2')

ax[1].set_title('Weekend Tropospheric NO2')



fig, ax = plt.subplots(1,2, figsize=(16,10))

ax = ax.ravel()

sns.heatmap(weekdayav_strat,ax=ax[0],square=True,cmap='inferno',cbar_kws={"orientation": "horizontal"}, vmax=np.nanmax(weekdayav_strat), vmin=np.nanmax(weekendav_strat))

sns.heatmap(weekendav_strat,ax=ax[1],square=True,cmap='inferno',cbar_kws={"orientation": "horizontal"}, vmax=np.nanmax(weekdayav_strat), vmin=np.nanmax(weekendav_strat))

[x.axis('off') for x in ax]



ax[0].set_title('Weekday Stratospheric NO2')

ax[1].set_title('Weekend Stratospheric NO2')
##nox dataframe of dates with corresponding files

nodf = {}

for file in os.listdir(s5p_no2_path):

    ao = datetime.strptime(file[8:16], '%Y%m%d')

    nodf[ao] = file

    

nodf = pd.DataFrame.from_dict(nodf, orient='index', columns=['file'])

nodf['date'] = nodf.index

nodf = nodf.sort_values(by='date')
month_no2 = {}

month_cf = {}

for mth in range(1,13):

    month_no2[mth]=[]

    month_cf[mth]=[]

for file in nodf.file:

    ao = datetime.strptime(file[8:16], '%Y%m%d')

    filr = rio.open(s5p_no2_path+file)

    no = np.flipud(filr.read(1))

    cf = np.flipud(filr.read(7))

    nocf = np.divide(no, 1-0.95*cf)

    nocf = np.nan_to_num(nocf, posinf=np.nan)

    month_no2[ao.month].append(nocf)

    month_cf[ao.month].append(cf)
wind, angle, temp, humidity = {}, {}, {}, {}

for mth in range(1,13):

    wind[mth]=[]

    angle[mth]=[]

    temp[mth]=[]

    humidity[mth]=[]

for dt in tqdm(pd.to_datetime(nodf.date)):

    file = gfs_path + 'gfs_'+ dt.strftime('%Y%m%d') + '12.tif'

    gfs_im = rio.open(file)

    us = gfs_im.read(4)

    vs = gfs_im.read(5)

    wind[dt.month].append(np.sqrt(us**2+vs**2))

    angle[dt.month].append(np.arctan2(vs,us))

    temp[dt.month].append(gfs_im.read(1))

    humidity[dt.month].append(gfs_im.read(2))
month_av_no2 ={}

month_av_cf ={}

month_av_wind = {}

month_av_ang = {}

month_av_temp ={}

month_av_hum ={}

for i in range(1,13):

    month_av_no2[i] = np.nanmean(np.array(month_no2[i]), axis=0)

    month_av_cf[i] = np.nanmean(np.array(month_cf[i]), axis=0)

    month_av_wind[i] = np.nanmean(np.array(wind[i]), axis=0)

    month_av_ang[i] = np.nanmean(np.array(angle[i]), axis=0)

    month_av_temp[i] = np.nanmean(np.array(temp[i]), axis=0)

    month_av_hum[i] = np.nanmean(np.array(humidity[i]), axis=0)
avno = []

avcf = []

avwind = []

avtemp = []

avhum = []

for i in range(1,13):

    avno.append(np.nanmean(month_av_no2[i]))

    avcf.append(np.nanmean(month_av_cf[i]))

    avwind.append(np.nanmean(month_av_wind[i]))

    avtemp.append(np.nanmean(month_av_temp[i]))

    avhum.append(np.nanmean(month_av_hum[i]))
fig, ax = plt.subplots(1,3, figsize=(20,5))

ax[0].plot(avno, label='NOx')

ax[0].twinx().plot(avhum, label='Humid', color='red')

ax[1].plot(avwind, label='Wind', color='green')

ax[2].plot(avtemp, label='Temp', color='yellow')

ax[2].twinx().plot(avcf, label='CF', color='orange')

fig.legend()
fig, ax = plt.subplots(1,1)

plt.plot(avno, label='Avg NOx')

ax2 = ax.twinx()

plt.plot(month_av_elec, color='red', label='Total Electricity Gen')

fig.legend()
month_ef = (avno/month_av_elec)*10**8 #multiply 10^8 for convenience

month_ef.mean(), month_ef.std()
sns.kdeplot(month_ef)
mths = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

no2cols = [mth+'_no2' for mth in mths]

tempcols = [mth+'_temp' for mth in mths]

humidcols = [mth+'_hum' for mth in mths]

windcols = [mth+'_wind' for mth in mths]

angcols = [mth+'_ang' for mth in mths]

for col in no2cols:

    gppd[col]=np.nan

for col in tempcols:

    gppd[col]=np.nan

for col in humidcols:

    gppd[col]=np.nan

for col in windcols:

    gppd[col]=np.nan

for col in angcols:

    gppd[col]=np.nan
for plant in range(len(gppd)):

    plantrow = gppd.iloc[plant]

    for i in range(12):

        gppd.loc[plant,mths[i]+'_no2']=month_av_no2[i+1][int(no2_row[plant]),int(no2_col[plant])]

        gppd.loc[plant,mths[i]+'_temp']=month_av_temp[i+1][int(no2_row[plant]),int(no2_col[plant])]

        gppd.loc[plant,mths[i]+'_hum']=month_av_hum[i+1][int(no2_row[plant]),int(no2_col[plant])]

        gppd.loc[plant,mths[i]+'_wind']=month_av_wind[i+1][int(no2_row[plant]),int(no2_col[plant])]

        gppd.loc[plant,mths[i]+'_ang']=month_av_ang[i+1][int(no2_row[plant]),int(no2_col[plant])]
nd=2

for plant in range(len(gppd)):

    plantrow = gppd.iloc[plant]

    for i in range(12):

        gppd.loc[plant,mths[i]+'_no2']=np.nanmean(month_av_no2[i+1][no2_row[i]-nd:no2_row[i]+nd,no2_col[i]-nd:no2_col[i]+nd])
#distance matrix to find the nearest plant and its primary fuel type for each power plant.

dist = np.zeros((35,35))

#dist[:] = np.nan

for i in range(len(gppd)):

    for j in range(i+1,len(gppd)):

        dist[i,j] = geopy.distance.distance(gppd['coords'].values[i],gppd['coords'].values[j]).km

dist = dist + dist.T - np.diag(np.diag(dist))

near_dist = []

near_type = []

near_ix = []

for i in range(35):

    near_index = np.argsort(dist[i])[1]

    near_dist.append(dist[i][near_index])

    near_type.append(gppd['primary_fuel'].values[near_index])

    near_ix.append(near_index)

gppd['near_dist']=near_dist

gppd['near_type']=near_type

gppd['near_index']=near_ix
gppd[['name','near_dist','near_index','primary_fuel']].sort_values(by='near_dist')[:25]
gppd2 = pd.merge(gppd, eia, on='Ix', how='left')

netgencols = [x for x in gppd2.columns if 'Netgen' in x]

for ncol in netgencols:

    gppd2[ncol]=np.where(gppd2[ncol]=='.',0,gppd2[ncol])

    gppd2[ncol] = gppd2[ncol].str.replace(',', '').astype(float)

    gppd2[ncol]=np.where(gppd2[ncol].isna(),0,gppd2[ncol])
gppdf = gppd2[gppd2['primary_fuel'].isin(['Coal','Oil','Gas'])]
sap = ['Yabucoa', 'Daguao','Mayagüez', 'Vega Baja', 'Vieques EPP','Cambalache', 'San Juan CC']

fig, ax = plt.subplots(3,3, figsize=(20,15))



for i in range(7):

    g = gppdf[gppdf['name']==sap[i]]

    r = int((i/3)%3)

    c = int(i%3)

    ax[r,c].plot(g[netgencols].sum()[6:], label='elec')

    ax[r,c].twinx().plot(g[no2cols].sum()[6:], color='red', label='no2')

    ax[r,c].set_title(sap[i])

fig.legend()
gppdf1 = gppdf[gppdf['name']=='Aguirre']

gppdf2 = gppdf[gppdf['name']=='Palo Seco']

gppdf3 = gppdf[gppdf['name'].isin(['EcoEléctrica','Costa Sur'])]

gppdf4 = gppdf[gppdf['name'].isin(['Jobos','A.E.S. Corp.'])]
fig, ax = plt.subplots(2,2, figsize=(20,5))

ax[0,0].plot(gppdf1[netgencols].sum()[6:], label='elec')

ax2=ax[0,0].twinx()

ax2.plot(gppdf1[no2cols].sum()[6:], color='red', label='no2')



ax[0,1].plot(gppdf2[netgencols].sum()[6:])

ax3=ax[0,1].twinx()

ax3.plot(gppdf2[no2cols].sum()[6:], color='red')



ax[1,0].plot(gppdf3[netgencols].sum()[6:])

ax4=ax[1,0].twinx()

ax4.plot(gppdf3[no2cols].sum()[6:], color='red')



ax[1,1].plot(gppdf4[netgencols].sum()[6:])

ax5=ax[1,1].twinx()

ax5.plot(gppdf4[no2cols].sum()[6:], color='red')



fig.legend()