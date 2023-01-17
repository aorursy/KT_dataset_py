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

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Taking a quick look at the first 10 rows
df = pd.read_csv('../input/measurements.csv', nrows=10)
df.head()
# Loading all rows but only Latitude, Longitude, Value, Unit and LoaderID
df = pd.read_csv('../input/measurements.csv', usecols=[1, 2, 3, 4, 12])
df.head()
# Number of measurements
print('Number of measurements: ', df.shape[0])
# Renaming columns
df.columns = ['lat', 'lon', 'value','unit', 'loader_id']
# Keeping only cpm (counts per minutes)
df = df[df.unit == 'cpm']
# Convert cpm to µSv/h
# http://safecast.org/tilemap/methodology.html
df.value = df.value / 350
# Drop any NA
df.dropna(axis=0, how='any', inplace=True)
# Keep only positive values
df = df[df.value > 0]
# "Casting" Safecast data
df.loader_id = df.loader_id.astype(int)
loaders = df['loader_id'].value_counts()
print("Min. # of measurements loaded: ", loaders.min())
print("Max. # of measurements loaded: ", loaders.max())
fig, ax = plt.subplots()
loaders.hist(ax=ax, bins=100, bottom=0.1)
ax.set_yscale('log')
ax.set_xlabel("# of measurements")
ax.set_ylabel("# of loaders (Log scale)")
plot_width  = int(800)
plot_height = int(plot_width//1.2)
def draw_map(df, plot_width, plot_height, colors, agg_func, interp, background_col):
    cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height)
    agg = cvs.points(df, 'lon', 'lat',  agg_func('value'))
    img = tf.shade(agg, cmap=colors, how=interp)
    return tf.set_background(img, color=background_col)
# NEED PROPER MAP LEGEND
img = draw_map(df, plot_width, plot_height, inferno, ds.count, 'log', 'black')
img
x_min_jpn, y_min_jpn, x_max_jpn, y_max_jpn = 128.03, 30.22, 148.65, 45.83
df_jpn = df[(df.lon > x_min_jpn) & (df.lon < x_max_jpn) & (df.lat > y_min_jpn) & (df.lat < y_max_jpn)]
img = draw_map(df_jpn, plot_width, plot_height, inferno, ds.count, 'log', 'black')
img
x_min_fk, y_min_fk, x_max_fk, y_max_fk = 140.0166, 37.0047, 141.2251, 38.195
df_fk= df[(df.lon > x_min_fk) & (df.lon < x_max_fk) & (df.lat > y_min_fk) & (df.lat < y_max_fk)]
img = draw_map(df_fk, plot_width, plot_height, inferno, ds.count, 'log', 'black')
img
