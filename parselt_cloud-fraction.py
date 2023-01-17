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
s5p_no2_path = '../input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/'

gfs_path = '../input/ds4g-environmental-insights-explorer/eie_data/gfs/'
pop_map_file = '/kaggle/input/population/imageToDriveExample.tif'

pop_map = rio.open(pop_map_file).read(1)

p = resize(pop_map, (148,475))
s5p_bounds = rio.open(s5p_no2_path+os.listdir(s5p_no2_path)[0]).bounds

gfs_bounds = rio.open(gfs_path+os.listdir(gfs_path)[0]).bounds

s5p_top, s5p_bot, s5p_lef, s5p_rit = s5p_bounds[1], s5p_bounds[3], s5p_bounds[0], s5p_bounds[2] #will be flipped while reading

gfs_top, gfs_bot, gfs_lef, gfs_rit = gfs_bounds[3], gfs_bounds[1], gfs_bounds[0], gfs_bounds[2]
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



no2_latp = (gppd['lat']-s5p_top)/(s5p_bot-s5p_top)*100

no2_longp = (gppd['long']-s5p_lef)/(s5p_rit-s5p_lef)*100

no2_col = np.round(475*no2_longp/100).astype('int')

no2_row = np.round(148*no2_latp/100).astype('int')
#nox dataframe of dates with corresponding files

nodf = {}

for file in os.listdir(s5p_no2_path):

    ao = datetime.strptime(file[8:16], '%Y%m%d')

    nodf[ao] = file
nodf = pd.DataFrame.from_dict(nodf, orient='index', columns=['file'])

nodf['date'] = nodf.index

nodf = nodf.sort_values(by='date')

nodf.head(2)
#nox, cloud fraction collated by month

no2 = {}

cf = {}

for mth in range(1,13):

    no2[mth]=[]

    cf[mth]=[]

for file in nodf.file:

    ao = datetime.strptime(file[8:16], '%Y%m%d')

    no2[ao.month].append(np.flipud(rio.open(s5p_no2_path+file).read(1)))

    cf[ao.month].append(np.flipud(rio.open(s5p_no2_path+file).read(7)))
#wind speed, wind angle collated by month

wind, angle = {}, {}

for mth in range(1,13):

    wind[mth]=[]

    angle[mth]=[]

    

for dt in tqdm(pd.to_datetime(nodf.date)):

    file = gfs_path + 'gfs_'+ dt.strftime('%Y%m%d') + '12.tif'

    gfs_im = rio.open(file)

    us = gfs_im.read(4)

    vs = gfs_im.read(5)

    wind[dt.month].append(np.sqrt(us**2+vs**2))

    angle[dt.month].append(np.arctan2(vs,us))

    

month_av_no2 ={}

month_av_cf ={}

month_av_wind = {}

month_av_ang = {}

for i in range(1,13):

    month_av_no2[i] = np.nanmean(np.array(no2[i]), axis=0)

    month_av_cf[i] = np.nanmean(np.array(cf[i]), axis=0)

    month_av_wind[i] = np.nanmean(np.array(wind[i]), axis=0)

    month_av_ang[i] = np.nanmean(np.array(angle[i]), axis=0)
fuels = ['Coal','Gas','Hydro','Oil','Solar','Wind']

colors = ['black','gray','green','darkred','orange','blue']

fudic = {}

for k,v in zip(fuels, colors):

        fudic[k]=v
r = no2_row[16]

c = no2_col[16]
all_yr_no2 = np.empty([12,148,475])

all_yr_cf = np.empty([12,148,475])

all_yr_ang = np.empty([12,148,475])

for i in range(1,13):

    all_yr_no2[i-1] = month_av_no2[i]

    all_yr_ang[i-1] = month_av_ang[i]

    all_yr_cf[i-1] = month_av_cf[i]

annual_no2_avg = np.nanmean(all_yr_no2, axis=0)

annual_ang_avg = np.nanmean(all_yr_ang, axis=0)

annual_cf_avg = np.nanmean(all_yr_cf, axis=0)
fig, ax = plt.subplots(1,2, figsize=(20,5))

nocf = np.multiply(annual_no2_avg, annual_cf_avg)

date = '2018-08-08'

print (date)

no2map = rio.open(s5p_no2_path+nodf.loc[date]['file']).read(1)

cfmap = rio.open(s5p_no2_path+nodf.loc[date]['file']).read(7)

sns.heatmap(no2map[60:70,180:200] , cmap='Reds', ax=ax[0])

sns.heatmap(cfmap[60:70,180:200] , cmap='Reds', ax=ax[1])

ax[0].set_title('NO2 Column')

ax[1].set_title('Cloud Fraction')

#[x.axis('off') for x in ax]

ax[0].grid()

ax[1].grid()
fig, ax = plt.subplots(1,3, figsize=(20,5))

nocf = np.multiply(annual_no2_avg, annual_cf_avg)

date = '2018-07-01'

print (date)

no2map = rio.open(s5p_no2_path+nodf.loc[date]['file']).read(1)

cfmap = rio.open(s5p_no2_path+nodf.loc[date]['file']).read(7)

sns.heatmap(no2map[60:100,:80] , cmap='Reds', ax=ax[0])

sns.heatmap(cfmap[60:100,:80] , cmap='Reds', ax=ax[1])

sns.heatmap(p[60:100,:80] , cmap='Blues', vmin=0, vmax=750, alpha=0.2, ax=ax[2])

plt.scatter(c,r-60, color=fudic[gppd['primary_fuel'].iloc[i]])

a = angle[7][1][r,c]

for i in range(3):

    ax[i].arrow(c, r-60, 5*np.cos(a), 5*np.sin(a), color='black', head_width=2) 

ax[0].set_title('NO2 Column')

ax[1].set_title('Cloud Fraction')

ax[2].set_title('Population Density')

[x.axis('off') for x in ax]

ax[0].grid()

ax[1].grid()
fig, ax = plt.subplots(1,2, figsize=(20,5))

nocf = np.multiply(annual_no2_avg, annual_cf_avg)

date = '2018-10-12'

print (date)

no2map = rio.open(s5p_no2_path+nodf.loc[date]['file']).read(1)

cfmap = rio.open(s5p_no2_path+nodf.loc[date]['file']).read(7)

sns.heatmap(no2map[60:100,:80] , cmap='Reds', ax=ax[0])

sns.heatmap(cfmap[60:100,:80] , cmap='Reds', ax=ax[1])

plt.scatter(c,r-60, color=fudic[gppd['primary_fuel'].iloc[i]])

a = angle[10][12][r,c]

for i in range(2):

    ax[i].arrow(c, r-60, 5*np.cos(a), 5*np.sin(a), color='black', head_width=2) 

ax[0].set_title('NO2 Column')

ax[1].set_title('Cloud Fraction')

[x.axis('off') for x in ax]
fig, ax = plt.subplots(1,2, figsize=(20,5))

nocf = np.multiply(annual_no2_avg, annual_cf_avg)

date = '2019-02-02'

print (date)

no2map = rio.open(s5p_no2_path+nodf.loc[date]['file']).read(1)

cfmap = rio.open(s5p_no2_path+nodf.loc[date]['file']).read(7)

sns.heatmap(no2map[60:100,:80] , cmap='Reds', ax=ax[0])

sns.heatmap(cfmap[60:100,:80] , cmap='Reds', ax=ax[1])

plt.scatter(c,r-60, color=fudic[gppd['primary_fuel'].iloc[i]])

a = angle[2][2][r,c]

for i in range(2):

    ax[i].arrow(c, r-60, 5*np.cos(a), 5*np.sin(a), color='black', head_width=2) 

ax[0].set_title('NO2 Column')

ax[1].set_title('Cloud Fraction')

[x.axis('off') for x in ax]
fig, ax = plt.subplots(1,3, figsize=(20,5))

date = '2018-07-02'

print (date)

no2map = rio.open(s5p_no2_path+nodf.loc[date]['file']).read(1)

cfmap = rio.open(s5p_no2_path+nodf.loc[date]['file']).read(7)

sns.heatmap(no2map[60:100,:80] , cmap='Reds', ax=ax[0])

sns.heatmap(cfmap[60:100,:80] , cmap='Reds', ax=ax[1])

sns.heatmap(np.divide(no2map, 1-0.9*cfmap)[60:100,:80] , cmap='Reds', ax=ax[2])

a = angle[7][2][r,c]

for i in range(3):

    ax[i].scatter(c,r-60, color=fudic[gppd['primary_fuel'].iloc[i]])

    ax[i].arrow(c, r-60, 5*np.cos(a), 5*np.sin(a), color='black', head_width=2) 

ax[0].set_title('NO2 Column')

ax[1].set_title('Cloud Fraction')

ax[2].set_title('Adjusted NO2 Column')

[x.axis('off') for x in ax]