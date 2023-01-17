# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import datashader as ds
from datashader import transfer_functions as tf
from datashader.colors import colormap_select, Greys9, Hot, viridis, inferno
from datashader.bokeh_ext import create_ramp_legend, create_categorical_legend
import warnings
warnings.filterwarnings('ignore')

from bokeh.io import output_notebook, show

from gc import collect
import seaborn as sns
from mpl_toolkits.basemap import Basemap

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Filtering and cleaning rows as we go and using only Captured Time, Latitude, Longitude, Value, and Unit cols
chunksize = 10 ** 6
chunk_list = []
for chunk in pd.read_csv('../input/measurements.csv', usecols=[0, 1, 2, 3, 4], chunksize=chunksize):
    chunk = chunk[chunk['Unit'] == 'cpm']
    chunk.value = chunk.Value/350
    chunk = chunk[chunk.Value > 0]
    chunk['year'] = pd.to_datetime(chunk['Captured Time'],  errors = 'coerce').dt.to_period('Y')
    chunk.dropna(axis=0, how='any', inplace=True)
    chunk['year'] = chunk['year'].astype('str').astype('int')
    chunk = chunk[chunk['year'] <= 2020]
    chunk = chunk.sample(n=12000)
    chunk_list.append(chunk)
    collect()
    if chunk.index[0] > 20000000:
        break
df = pd.concat(chunk_list)

del(chunk_list)
collect()
# Number of measurements
print('Number of measurements: ', df.shape[0])
df = df.drop(columns=['Unit'])
# Renaming columns
df.columns = ['time', 'lat', 'lon', 'value', 'year']
df.head()
years = [year for year in range(2012,2018)]
fig = plt.figure(figsize=(15,12), edgecolor='w')

for i in range(6):
    ax = fig.add_subplot(2,3,i+1)
    #m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,resolution='c')
    #m.drawcoastlines()
    year_df = df[df['year'] == years[i]]
    ax.scatter(year_df['lat'], year_df['lon'], color='orange', alpha=0.3) #cmap=np.log(year_df['value'])
    #sns.scatterplot(year_df['lat'], year_df['lon'], hue=np.log(year_df['value']), ax=ax[i])
    ax.set_title(str(years[i]))
fig.show()
plot_width  = int(600)
plot_height = int(plot_width//1.2)
def draw_radiation(df, plot_width, plot_height, colors, agg_func, interp, background_col):
    cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height)
    agg = cvs.points(df, 'lon', 'lat',  agg_func('value'))
    img = tf.shade(agg, cmap=colors, how=interp)
    return tf.set_background(img, color=background_col)
# NEED PROPER MAP LEGEND
img = draw_radiation(df, plot_width, plot_height, inferno, ds.mean, 'log', 'black')
img
x_min_jpn, y_min_jpn, x_max_jpn, y_max_jpn = 128.03, 30.22, 148.65, 45.83
df_jpn = df[(df.lon > x_min_jpn) & (df.lon < x_max_jpn) & (df.lat > y_min_jpn) & (df.lat < y_max_jpn)]
img = draw_radiation(df_jpn, plot_width, plot_height, inferno, ds.mean, 'log', 'black')
img
x_min_fk, y_min_fk, x_max_fk, y_max_fk = 140.0166, 37.0047, 141.2251, 38.195
df_fk= df[(df.lon > x_min_fk) & (df.lon < x_max_fk) & (df.lat > y_min_fk) & (df.lat < y_max_fk)]
img = draw_radiation(df_fk, plot_width, plot_height, inferno, ds.mean, 'log', 'black')
img
#x_min_sea, y_min_sea, x_max_sea, y_max_sea = 
#df_sea = df[(df.lon > x_min_sea) & (df.lon < x_max_sea) & (df.lat > y_min_sea) & (df.lat < y_max_sea)]
img = draw_radiation(df_sea, plot_width, plot_height, inferno, ds.mean, 'log', 'black')
img