import numpy as np 

import pandas as pd 



# plots

import matplotlib.pyplot as plt

#import seaborn as sns

#sns.set_style('whitegrid')

%matplotlib inline



from bokeh.io import show

from bokeh.charts import output_notebook

from bokeh.sampledata import us_states

from bokeh.plotting import figure

output_notebook()



from mpl_toolkits.basemap import Basemap

from matplotlib import cm



from sklearn.cluster import KMeans



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



ds = pd.read_csv('../input/AviationDataUP.csv')
ds.head()
ds['Event.Date']
ds.dtypes
ds['times'] = ds['Event.Date'].astype('datetime64[ns]', errors='coerce')

ds['Latitude'] = pd.to_numeric(ds.Latitude, errors='coerce')

ds['Latitude'] = pd.to_numeric(ds.Latitude, errors='coerce')

ds = ds.dropna(axis=0, subset=['Latitude', 'Longitude'])
ds.head()
ds['month'] = ds['times'].map(lambda x: x.month)
ds.head()
fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(8, 8))

fig.subplots_adjust(hspace=0.8)

total_month = ds['month'].value_counts()

print(total_month)

ds['month'].value_counts().plot(ax=axes[0], kind='bar', title='month-wise accidents')

ds['cleaned.make'] = ds['Make'].map(lambda x: "{}".format(x).lower().strip())
fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(9, 6))

fig.subplots_adjust(hspace=.6)

colors = ['#99cc33', '#a333cc', '#333dcc']

ds['Broad.Phase.of.Flight'].value_counts().plot(ax=axes[0,0], kind='bar', title='Phase of Flight')

ds['Broad.Phase.of.Flight'].value_counts().plot(ax=axes[0,1], kind='pie', title='Phase of Flight')

ds['Weather.Condition'].value_counts().plot(ax=axes[1,0], kind='pie', colors=colors, title='Weather Condition')

# TODO: clean up to add "other"

# ds['cleaned.make'].value_counts().plot(ax=axes[1,1], kind='pie', title='Aircraft Make')
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
latlon = ds[['Longitude', 'Latitude']]

latlon.head()
kmeans = KMeans(n_clusters=50)

kmodel = kmeans.fit(latlon)

centroids = kmodel.cluster_centers_
centroids

lons, lats = zip(*centroids)

print(lats)

print(lons)
fig = plt.figure()

ax = fig.add_axes([0.1,0.1,0.8,0.8])

north, south, east, west = 71.39, 24.52, -66.95, 172.5

m = Basemap(llcrnrlon=-145.5,llcrnrlat=1.0,urcrnrlon=-2.566,urcrnrlat=46.352,

            rsphere=(6378137.00,6356752.3142),

            resolution='l',area_thresh=1000.0,projection='lcc',

            lat_1=50.0,lon_0=-107.0,ax=ax)

x, y = m(ds['Longitude'].values, ds['Latitude'].values)

m.drawcoastlines()

m.drawcountries()

m.hexbin(x, y, gridsize=1000, bins='log', cmap=cm.YlOrRd)

cx, cy = m(lons, lats)

m.scatter(cx, cy, 3, color='g')
from bokeh.sampledata import us_states

us_states = us_states.data.copy()

state_xs = [us_states[code]["lons"] for code in us_states]

state_ys = [us_states[code]["lats"] for code in us_states]

p = figure(title="Aviation Incidents and Centroids", 

           toolbar_location="left", plot_width=1100, plot_height=700)

p.patches(state_xs, state_ys, fill_alpha=0.0,

    line_color="#884444", line_width=1.5)

p.circle(ds['Longitude'].data, ds['Latitude'].data, size=8, color='navy', alpha=1)

p.circle(lons, lats, size=8, color='navy', alpha=1)

show(p)