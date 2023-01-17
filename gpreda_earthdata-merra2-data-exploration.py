import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import folium

from folium.plugins import HeatMap, HeatMapWithTime

from netCDF4 import Dataset

import cartopy.crs as ccrs

import os
def plot_merra_data(data, title='', date='April 2020', label="kg m-2"):

    fig = plt.figure(figsize=(16,8))

    ax = plt.axes(projection=ccrs.Robinson())

    ax.set_global()

    ax.coastlines(resolution="110m",linewidth=1)

    ax.gridlines(linestyle='--',color='black')

    plt.contourf(lons, lats, data, transform=ccrs.PlateCarree(),cmap=plt.cm.jet)

    plt.title(f'MERRA-2 {title} levels, {date}', size=14)

    cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)

    cb.set_label(f'{label}',size=12,rotation=0,labelpad=15)

    cb.ax.tick_params(labelsize=10)

    plt.show()
KAGGLE_PATH = "/kaggle/input/earthdata-merra2-co/"

for file in os.listdir(KAGGLE_PATH):

    file_date = file.split(".")[-2]

    print(file, file_date)

    file_path = os.path.join(KAGGLE_PATH, file)

    data = Dataset(file_path, more="r")

    lons = data.variables['lon'][:]

    lats = data.variables['lat'][:]

    time = data.variables['time'][:]

    COCL = data.variables['COCL'][:,:,:]; COCL = COCL[0,:,:]

    plot_merra_data(COCL, title='CO Column Burden (COCL)', date=file_date, label="kg m-2")
for file in os.listdir(KAGGLE_PATH):

    file_date = file.split(".")[-2]

    print(file, file_date)

    file_path = os.path.join(KAGGLE_PATH, file)

    data = Dataset(file_path, more="r")

    lons = data.variables['lon'][:]

    lats = data.variables['lat'][:]

    time = data.variables['time'][:]

    COEM = data.variables['COEM'][:,:,:]; COEM = COEM[0,:,:]

    plot_merra_data(COEM, title='CO Emissions (COEM)', date=file_date, label="kg m-2 s-1")
for file in os.listdir(KAGGLE_PATH):

    file_date = file.split(".")[-2]

    print(file, file_date)

    file_path = os.path.join(KAGGLE_PATH, file)

    data = Dataset(file_path, more="r")

    lons = data.variables['lon'][:]

    lats = data.variables['lat'][:]

    time = data.variables['time'][:]

    TO3 = data.variables['TO3'][:,:,:]; TO3 = TO3[0,:,:]

    plot_merra_data(TO3, title='Total column ozon (TO3)', date=file_date)
for file in os.listdir(KAGGLE_PATH):

    file_date = file.split(".")[-2]

    print(file, file_date)

    file_path = os.path.join(KAGGLE_PATH, file)

    data = Dataset(file_path, more="r")

    lons = data.variables['lon'][:]

    lats = data.variables['lat'][:]

    time = data.variables['time'][:]

    COSC = data.variables['COSC'][:,:,:]; COSC = COSC[0,:,:]

    plot_merra_data(COSC, title='CO Surface Concentration  (COSC)', date=file_date, label="1e-9")