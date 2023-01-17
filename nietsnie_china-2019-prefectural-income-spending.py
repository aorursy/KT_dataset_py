!wget https://raw.githubusercontent.com/tsycstang/china_geojson/master/china_prov.zip

!wget https://raw.githubusercontent.com/tsycstang/china_geojson/master/china_pref.zip
import geopandas as gpd

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.colors as colors

import cartopy.crs as ccrs



%config InlineBackend.figure_formats = ['retina']

%matplotlib inline
china_prov = gpd.read_file('zip://china_prov.zip').astype({'code': int})

china_pref = gpd.read_file('zip://china_pref.zip').astype({'code': int})

disp_inc = pd.read_csv('../input/china-2019-prefectural-disposable-income/china_disposable_income.csv').astype({'code': int})

china_pref = china_pref.merge(disp_inc, on='code')
china_prov.head()
china_pref.head()
crs = ccrs.Orthographic(central_longitude=109.0, central_latitude=34.0)

crs_proj4 = crs.proj4_init

china_prov_ae = china_prov.to_crs(crs_proj4)

china_prov_ae.boundary.plot(figsize=(10, 10))

ax = plt.gca()

ax.axes.xaxis.set_visible(False)

ax.axes.yaxis.set_visible(False)
crs = ccrs.Orthographic(central_longitude=109.0, central_latitude=34.0)

crs_proj4 = crs.proj4_init

china_pref_ae = china_pref.to_crs(crs_proj4)

china_pref_ae.boundary.plot(figsize=(10, 10))

ax = plt.gca()

ax.axes.xaxis.set_visible(False)

ax.axes.yaxis.set_visible(False)
china_pref['inc diff'] = china_pref['disposable income, overall'] - 30733
base = china_pref.to_crs(crs_proj4).plot('inc diff', 

                                         legend=True, 

                                         figsize=(18, 18), 

                                         cmap='seismic', 

                                         missing_kwds={'color': 'lightgrey'},

                                         legend_kwds={'label': "Prefectural Level Disposable Income Diff (Ref: National 30733) of 2019 Mainland China"}, 

                                         vmin=-50000., 

                                         vmax=50000.)

china_prov.to_crs(crs_proj4).plot(ax=base, facecolor='none', edgecolor='lightgrey')

ax = plt.gca()

ax.axes.xaxis.set_visible(False)

ax.axes.yaxis.set_visible(False)
base = china_pref.to_crs(crs_proj4).plot('disposable income, overall', 

                                         legend=True, 

                                         figsize=(18, 18), 

                                         cmap='YlOrBr', 

                                         missing_kwds={'color': 'lightgrey'},

                                         legend_kwds={'label': "Prefectural Level Disposable Income (Ref: National Avg 30733) of 2019 Mainland China"}, 

                                         vmin=0., 

                                         vmax=80000.)

china_prov.to_crs(crs_proj4).plot(ax=base, facecolor='none', edgecolor='lightgrey')

ax = plt.gca()

ax.axes.xaxis.set_visible(False)

ax.axes.yaxis.set_visible(False)
base = china_pref.to_crs(crs_proj4).plot('urban rural income ratio', 

                                         legend=True, 

                                         figsize=(18, 18), 

                                         cmap='YlOrBr', 

                                         missing_kwds={'color': 'lightgrey'},

                                         legend_kwds={'label': "Prefectural Level Urban-Rural Income Ratio of 2019 Mainland China"}, 

                                         )

china_prov.to_crs(crs_proj4).plot(ax=base, facecolor='none', edgecolor='lightgrey')

ax = plt.gca()

ax.axes.xaxis.set_visible(False)

ax.axes.yaxis.set_visible(False)
base = china_pref.to_crs(crs_proj4).plot('urban rural spending ratio', 

                                         legend=True, 

                                         figsize=(18, 18), 

                                         cmap='YlOrBr', 

                                         missing_kwds={'color': 'lightgrey'},

                                         legend_kwds={'label': "Prefectural Level Urban-Rural Spending Ratio of 2019 Mainland China"}, 

                                         )

china_prov.to_crs(crs_proj4).plot(ax=base, facecolor='none', edgecolor='lightgrey')

ax = plt.gca()

ax.axes.xaxis.set_visible(False)

ax.axes.yaxis.set_visible(False)