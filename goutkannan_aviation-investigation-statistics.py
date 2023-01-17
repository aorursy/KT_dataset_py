# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# plots

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline

import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



ds = pd.read_csv('../input/AviationDataUP.csv')
ds.head()
import collections 

from operator import *

count = []



for make,model in zip(ds['Make'],ds['Model']):

    count.append(str(make).strip()+"-"+str(model).strip())

  

        

collections.Counter(count).most_common(20)
fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(9, 6))

fig.subplots_adjust(hspace=.6)

colors = ['#99cc33', '#a333cc', '#333dcc']

ds['Broad.Phase.of.Flight'].value_counts().plot(ax=axes[0,0], kind='bar', title='Phase of Flight')

ds['Broad.Phase.of.Flight'].value_counts().plot(ax=axes[0,1], kind='pie', title='Phase of Flight')

ds['Weather.Condition'].value_counts().plot(ax=axes[1,0], kind='pie', colors=colors, title='Weather Condition')

# TODO: clean up to add "other"

# ds['cleaned.make'].value_counts().plot(ax=axes[1,1], kind='pie', title='Aircraft Make')
from mpl_toolkits.basemap import Basemap

from matplotlib import cm

fig = plt.figure()

ax = fig.add_axes([0.1,0.1,0.8,0.8])

north, south, east, west = 71.39, 24.52, -66.95, 172.5

#m = Basemap(

#    projection='lcc',

#    llcrnrlat=south,

#    urcrnrlat=north,

#    llcrnrlon=west,

#    urcrnrlon=east,

#    lat_1=33,

#    lat_2=45,

#    lon_0=-95,

#    resolution='l')

m = Basemap(llcrnrlon=-145.5,llcrnrlat=1.0,urcrnrlon=-2.566,urcrnrlat=46.352,

            rsphere=(6378137.00,6356752.3142),

            resolution='l',area_thresh=1000.0,projection='lcc',

            lat_1=50.0,lon_0=-107.0,ax=ax)

x, y = m(ds['Longitude'].values, ds['Latitude'].values)

m.drawcoastlines()

m.drawcountries()

m.hexbin(x, y, gridsize=1000, bins='log', cmap=cm.YlOrRd)