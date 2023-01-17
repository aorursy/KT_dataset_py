# We need to install country_converter to convert various country names to ISO3 format.

!pip install country_converter
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import geopandas as gpd

import country_converter as coco

import json



import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.animation as animation

from matplotlib.animation import FuncAnimation

from matplotlib.colors import Normalize, LogNorm
ts_confirmed = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

ts_confirmed = ts_confirmed.drop(['Province/State', 'Lat', 'Long'], axis=1)

# We need to rename "UK" as "United Kingdom" for it to be properly recognized.

ts_confirmed.loc[ts_confirmed['Country/Region']=='UK','Country/Region'] = 'United Kingdom'

days = list(ts_confirmed.columns.values[1:])
ts_countries = ts_confirmed.groupby(['Country/Region']).sum().reset_index()

ts_countries['ISO3'] = coco.convert(names=ts_countries['Country/Region'].to_list())
ts_countries_diff = ts_countries.loc[:,['Country/Region', 'ISO3']]



sliding_window_size = 3

ts_countries_diff[days] = ts_countries.loc[:,days].diff(periods=3, axis=1) / sliding_window_size

ts_countries_diff = ts_countries_diff.drop(days[0:sliding_window_size], axis=1)

diff_days = days[sliding_window_size:]

# ts_countries_diff.head()
gdf = gpd.read_file('/kaggle/input/natural-earth-1110m-countries/ne_110m_admin_0_countries.shp')[['ADMIN', 'ADM0_A3_IS', 'geometry']]

gdf.columns = ['country', 'country_code', 'geometry']
merged = gdf.merge(ts_countries[['ISO3'] + days], left_on = 'country_code', right_on = 'ISO3', how = 'left')

merged[days] = merged[days].fillna(0)

merged = merged.drop(merged.index[merged['country'] == 'Antarctica'])



merged_diff = gdf.merge(ts_countries_diff[['ISO3'] + diff_days], left_on = 'country_code', right_on = 'ISO3', how = 'left')

merged_diff[diff_days] = merged_diff[diff_days].fillna(0)

merged_diff = merged_diff.drop(merged_diff.index[merged_diff['country'] == 'Antarctica'])
fig, ax = plt.subplots(1, figsize=(20, 10))

ax.axis('off')

ax.set_title('Confirmed COVID-19 cases on {}'.format(days[-1]), fontdict={'fontsize': '25', 'fontweight' : '3'})

# fig.patch.set_facecolor('black')

sm = plt.cm.ScalarMappable(cmap='Blues', norm=LogNorm(vmin=1, vmax=100000))

sm._A = []

cax = fig.add_axes([ax.get_position().x0,ax.get_position().y0,ax.get_position().width, 0.01])

cb = fig.colorbar(sm, cax=cax, orientation='horizontal')

plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), size='16')

ax.annotate('Source: Johns Hopkins University via Kaggle', xy=(0, 0), xycoords='axes pixels', fontsize=12, color='#555555')

merged.plot(column=days[-1], cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8', norm=LogNorm(vmin=1, vmax=100000))
fig, ax = plt.subplots(1, figsize=(20, 10))

ax.axis('off')

draw = plt.plot([], [])[0]

cax = fig.add_axes([ax.get_position().x0,ax.get_position().y0,ax.get_position().width, 0.01])

sm = plt.cm.ScalarMappable(cmap='Blues', norm=LogNorm(vmin=1, vmax=100000))

sm._A = []

cb = fig.colorbar(sm, cax=cax, orientation='horizontal')

ax.annotate('Source: Johns Hopkins University via Kaggle', xy=(0, 0), xycoords='axes pixels', fontsize=12, color='#555555')

    

def animate(x):

    ax.set_title('Confirmed COVID-19 cases on {}'.format(x), fontdict={'fontsize': '25', 'fontweight' : '3'})

    plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), size='16')

    merged.plot(column=x, cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8', norm=LogNorm(vmin=1, vmax=100000))

    return draw,



output = animation.FuncAnimation(fig, animate, days, interval=400, blit=True, repeat=True)

output.save('covid2019_confirmed.gif', writer='imagemagick')
fig, ax = plt.subplots(1, figsize=(20, 10))

ax.axis('off')

draw = plt.plot([], [])[0]

cax = fig.add_axes([ax.get_position().x0,ax.get_position().y0,ax.get_position().width, 0.01])

sm = plt.cm.ScalarMappable(cmap='Blues', norm=Normalize(vmin=1, vmax=1000))

sm._A = []

cb = fig.colorbar(sm, cax=cax, orientation='horizontal')

ax.annotate('Source: Johns Hopkins University via Kaggle', xy=(0, 0), xycoords='axes pixels', fontsize=12, color='#555555')

    

def animate(x):

    ax.set_title('New cases of COVID-19 on {}'.format(x), fontdict={'fontsize': '25', 'fontweight' : '3'})

    plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), size='16')

    merged_diff.plot(column=x, cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8', norm=Normalize(vmin=1, vmax=1000))

    return draw,



output = animation.FuncAnimation(fig, animate, diff_days, interval=400, blit=True, repeat=True)

output.save('covid2019_confirmed_diff.gif', writer='imagemagick')