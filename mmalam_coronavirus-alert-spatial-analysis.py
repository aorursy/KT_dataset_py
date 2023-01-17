import pandas as pd

import numpy as np

import sklearn

import seaborn as sns

import matplotlib.pyplot as plt

import geopandas as gpd

from shapely.geometry import Point, Polygon

import matplotlib.gridspec as gridspec

import geopandas as gpd

from shapely.geometry import Point, Polygon

from mpl_toolkits.axes_grid1 import make_axes_locatable
confirmed = pd.read_csv('../input/Confirmed.csv')

fig,ax = plt.subplots(figsize = (20,20))

title = plt.title('Confirmed Cases by Location', fontsize=20)

title.set_position([0.5, 1.05])

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

world.plot(ax = ax, color='grey', edgecolor='black',linewidth=1, alpha=0.1)

sns.scatterplot(confirmed.Lon,confirmed.Lat,size=confirmed["4_2"], ax=ax)
confirmed_china = confirmed[confirmed["Country"] == 'Mainland China']

fig,ax = plt.subplots(figsize = (20,20))

title = plt.title('Confirmed Cases in China', fontsize=20)

title.set_position([0.5, 1.05])

china = gpd.read_file('../input/province.shp')

china.plot(ax = ax, color='grey', edgecolor='black',linewidth=1, alpha=0.1)

sns.scatterplot(confirmed_china.Lon,confirmed_china.Lat,size=confirmed_china["4_2"], ax=ax)


confirmed_by_country = pd.read_csv('../input/confirmed_by_country.csv')

world_confirmed = world.merge(confirmed_by_country, on='name', how='left')

world_confirmed["TotalConfirmed"].fillna(0, inplace=True)
fig,ax = plt.subplots(figsize = (20,10))

title = plt.title('Confirmed Cases Comparison by Country', fontsize=20)

title.set_position([0.5, 1.05])

world_confirmed.plot( column='TotalConfirmed', 

                  cmap='YlOrRd', legend=True, ax=ax

                  ,vmin=0, vmax=50

                  ,edgecolor='black',linewidth=0.1)

ax.axes.get_xaxis().set_visible(False)

ax.axes.get_yaxis().set_visible(False)
confirmed_china_province = pd.read_csv('../input/confirmed_china_province.csv')

china_confirmed = china.merge(confirmed_china_province, on='NAME', how='left')

china_confirmed["TotalConfirmed"].fillna(0, inplace=True)
fig,ax = plt.subplots(figsize = (20,10))

title = plt.title('Confirmed Cases in China by Provinces', fontsize=20)

title.set_position([0.5, 1.05])

china_confirmed.plot( column='TotalConfirmed', 

                  cmap='YlOrRd', legend=True, ax=ax

                  ,vmin=1, vmax=1500

                  ,edgecolor='black',linewidth=0.1)

ax.axes.get_xaxis().set_visible(False)

ax.axes.get_yaxis().set_visible(False)
recovery_percentage = pd.read_csv('../input/recovery_percentage.csv')

china_recovery_percentage = china.merge(recovery_percentage, on='NAME', how='left')

china_recovery_percentage["RecoveryPercentage"].fillna(0, inplace=True)



death_percentage = pd.read_csv('../input/Death_percentage.csv')

china_death_percentage = china.merge(death_percentage, on='NAME', how='left')

china_death_percentage["DeathPercentage"].fillna(0, inplace=True)
f,ax=plt.subplots(1,2,figsize=(20,5))



china_recovery_percentage.plot( column='RecoveryPercentage', 

                  cmap='Greens', legend=True, ax=ax[0]

                  ,edgecolor='black',linewidth=0.1)



china_death_percentage.plot( column='DeathPercentage', 

                  cmap='YlOrRd', legend=True, ax=ax[1]

                  ,edgecolor='black',linewidth=0.1)



f.suptitle('Recovered and Death Percentages', fontsize=14)

ax[0].set_title('Recovered Percent', fontsize=12)

ax[1].set_title('Death Percent', fontsize=12)

ax[0].axes.get_xaxis().set_visible(False)

ax[0].axes.get_yaxis().set_visible(False)

ax[1].axes.get_xaxis().set_visible(False)

ax[1].axes.get_yaxis().set_visible(False)