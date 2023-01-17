import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import folium

from folium.plugins import HeatMap, HeatMapWithTime

from netCDF4 import Dataset

import cartopy.crs as ccrs
data = Dataset("/kaggle/input/earthdata-merra2-co/MERRA2_400.tavgM_2d_chm_Nx.202004.nc4", more="r")
print(data)
lons = data.variables['lon'][:]

lats = data.variables['lat'][:]

time = data.variables['time'][:]

COCL = data.variables['COCL'][:,:,:]; COCL = COCL[0,:,:]

COEM = data.variables['COEM'][:,:,:]; COEM = COEM[0,:,:]

COLS = data.variables['COLS'][:,:,:]; COLS = COLS[0,:,:]

TO3 =  data.variables['TO3'][:,:,:];  TO3 =  TO3[0,:,:]
print(f"longitudes: {len(lons)}")

print(f"latitudes: {len(lats)}")

print(f"time: {len(time)}")

print(f"COCL: {len(COCL)}")

print(f"COEM: {len(COEM)}")

print(f"COLS: {len(COLS)}")

print(f"TO3:  {len(TO3)}")
print(f"Latitudes: {len(lats)}, vals[:]: {len(COCL[0])}, Longitudes: {len(lons)}, vals[:,:]: {len(COCL[0])}")
def plot_merra_data(data, title='', date='April 2020'):

    fig = plt.figure(figsize=(16,8))

    ax = plt.axes(projection=ccrs.Robinson())

    ax.set_global()

    ax.coastlines(resolution="110m",linewidth=1)

    ax.gridlines(linestyle='--',color='black')

    plt.contourf(lons, lats, data, transform=ccrs.PlateCarree(),cmap=plt.cm.jet)

    plt.title(f'MERRA-2 {title} levels, {date}', size=14)

    cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)

    cb.set_label('K',size=12,rotation=0,labelpad=15)

    cb.ax.tick_params(labelsize=10)
plot_merra_data(COCL, 'COCL')
plot_merra_data(COEM, 'COEM')
plot_merra_data(COLS, 'COLS')
plot_merra_data(TO3, 'TO3')