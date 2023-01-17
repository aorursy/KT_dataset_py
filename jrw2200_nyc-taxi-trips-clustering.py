import numpy as np

import pandas as pd 

import matplotlib

import matplotlib.pyplot as plt



import matplotlib.style as style 

style.use('ggplot')



import seaborn as sns

sns.set_context("paper")



import itertools



from sklearn.cluster import KMeans
original_data = pd.read_csv('/kaggle/input/yellow_tripdata_2016-06.csv', parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])
original_data.head(5)
original_data.columns
original_data.dtypes
original_data = original_data[(original_data['tpep_pickup_datetime'] > '2016-06-05') & (original_data['tpep_pickup_datetime'] > '2016-06-13')]
data = original_data[['passenger_count', 'trip_distance', 'pickup_longitude','pickup_latitude','dropoff_longitude', 'dropoff_latitude', 

                      'tpep_pickup_datetime', 'tpep_dropoff_datetime', ]]
#del original_data

data.head(5)
# Compute trip duration

data['trip_duration_mins'] = (data.tpep_dropoff_datetime - data.tpep_pickup_datetime).dt.seconds /60
print('There are {} features and {} examples.'.format(data.shape[1], data.shape[0]))
# find pick up and dropoff longitude and latitude range

print('Pickup longitude range: [{},{}]'.format(np.min(data['pickup_longitude']), np.max(data['pickup_longitude'])))

print('Pickup latitude range: [{},{}]'.format(np.min(data['pickup_latitude']), np.max(data['pickup_latitude'])))

print('Dropoff longitude range: [{},{}]'.format(np.min(data['dropoff_longitude']), np.max(data['dropoff_longitude'])))

print('Dropoff latitude range: [{},{}]'.format(np.min(data['dropoff_latitude']), np.max(data['dropoff_latitude'])))
sns.distplot(data[(data['pickup_longitude']>-74.05) & (data['pickup_longitude']<-73.75)]['pickup_longitude'])
sns.distplot(data[(data['dropoff_longitude']>-74.05) & (data['dropoff_longitude']<-73.75)]['dropoff_longitude'])
sns.distplot(data[(data['pickup_latitude']> 40.6) & (data['pickup_latitude']<40.9)]['pickup_latitude'])
sns.distplot(data[(data['dropoff_latitude']> 40.6) & (data['dropoff_latitude']<40.9)]['dropoff_latitude'])
sns.distplot(data[data['trip_distance'] < 25]['trip_distance'])
sns.distplot(data[data['trip_duration_mins']<100]['trip_duration_mins'])
data.dtypes
data = data[

    (data['pickup_longitude']>-74.05) & (data['pickup_longitude']<-73.75) &

    (data['pickup_latitude']> 40.6) & (data['pickup_latitude']<40.9) &

    (data['dropoff_longitude']>-74.05) & (data['dropoff_longitude']<-73.75) &

    (data['dropoff_latitude']> 40.6) & (data['dropoff_latitude']<40.9)&

    (data['trip_distance'] < 25) &

    (data['trip_duration_mins'] < 100)

]
NYC = x_range, y_range = ((-74.05, -73.7), (40.6, 40.9))
from bokeh.plotting import figure, output_notebook, show # bokeh plotting library



plot_width = int(750)

plot_height = int(plot_width//1.2)



def base_plot(tools='pan, wheel_zoom, reset', plot_width=plot_width, plot_height=plot_height, **plot_args):

    p = figure(tools=tools, plot_width=plot_width, plot_height=plot_height,

              x_range=x_range, y_range=y_range, outline_line_color=None,

              min_border=0, min_border_left=0, min_border_right=0,

              min_border_top=0, min_border_bottom=0, **plot_args)

    

    p.xgrid.grid_line_color = None

    p.ygrid.grid_line_color = None

    return p



options = dict(line_color=None, fill_color='blue', size=5)
import datashader as ds

from datashader import transfer_functions as tr_fns

from datashader.colors import Greys9

Greys9_r = list(reversed(Greys9))[:2]
from datashader.bokeh_ext import InteractiveImage

from functools import partial

from datashader.utils import export_image

from datashader.colors import colormap_select, Greys9, Hot, viridis, inferno

from IPython.core.display import HTML, display



background = "black"

export = partial(export_image, export_path="export", background=background)

cm = partial(colormap_select, reverse=(background=="black"))



def create_image(x_range, y_range, w=plot_width, h=plot_height):

    cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)

    agg = cvs.points(data, 'pickup_longitude', 'pickup_latitude', ds.count('passenger_count'))

    img = tr_fns.shade(agg, cmap=Hot, how='eq_hist')

    return tr_fns.dynspread(img, threshold=0.5, max_px=4)



p = base_plot(background_fill_color=background)

export(create_image(*NYC), "NYCT_pickups_hot")

InteractiveImage(p, create_image)
from datashader.bokeh_ext import InteractiveImage

from functools import partial

from datashader.utils import export_image

from datashader.colors import colormap_select, Greys9, Hot, viridis, inferno

from IPython.core.display import HTML, display



background = "black"

export = partial(export_image, export_path="export", background=background)

cm = partial(colormap_select, reverse=(background=="black"))



def create_image(x_range, y_range, w=plot_width, h=plot_height, data=data):

    cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)

    agg = cvs.points(data, 'dropoff_longitude', 'dropoff_latitude', ds.count('passenger_count'))

    img = tr_fns.shade(agg, cmap=Hot, how='eq_hist')

    return tr_fns.dynspread(img, threshold=0.5, max_px=4)



p = base_plot(background_fill_color=background)

export(create_image(*NYC), "NYCT_dropoffs_hot")

InteractiveImage(p, create_image)
idx1, idx2 = 0, 6571387
X0 = data.iloc[idx1:idx2]
X1 = data[['pickup_longitude', 'pickup_latitude']].iloc[idx1:idx2].values
no_classes = 12

est = KMeans(n_clusters=no_classes)

est.fit(X1)

labels = est.labels_
X0['pickup_cluster'] = labels
fignum=1

fig, ax = plt.subplots(fignum, figsize=(8, 6))



ax.scatter(X1[:, 0], X1[:, 1], c=labels.astype(np.float), marker='o')



#ax.w_xaxis.set_ticklabels([])

#ax.w_yaxis.set_ticklabels([])

#ax.w_zaxis.set_ticklabels([])

ax.set_xlabel('longitude')

ax.set_ylabel('latitude')

ax.legend()
del X1
X2 = data[['dropoff_longitude', 'dropoff_latitude']].iloc[idx1:idx2].values
no_classes = 12

est = KMeans(n_clusters=no_classes)

est.fit(X2)

labels = est.labels_
X0['dropoff_cluster'] = labels
fig, ax = plt.subplots(1, figsize=(8, 6))



ax.scatter(X2[:, 0], X2[:, 1], c=labels.astype(np.float), marker='o')



#ax.w_xaxis.set_ticklabels([])

#ax.w_yaxis.set_ticklabels([])

#ax.w_zaxis.set_ticklabels([])

ax.set_xlabel('longitude')

ax.set_ylabel('latitude')

ax.legend()
del X2
dropoff_clusters = X0['dropoff_cluster'].unique()

pickup_clusters = X0['pickup_cluster'].unique()
dropoff_clusters_dict = {}

for d in dropoff_clusters:

    #print(d, X0[X0['dropoff_cluster'] == d]['dropoff_latitude'].mean())

    dropoff_clusters_dict[d] = (X0[X0['dropoff_cluster'] == d]['dropoff_longitude'].mean(), X0[X0['dropoff_cluster'] == d]['dropoff_latitude'].mean())

dropoff_clusters_dict

#pd.DataFrame(dropoff_clusters_dict)
pickup_clusters_dict = {}

for p in pickup_clusters:

    #print(d, X0[X0['dropoff_cluster'] == d]['dropoff_latitude'].mean())

    pickup_clusters_dict[p] = (X0[X0['pickup_cluster'] == d]['pickup_longitude'].mean(), X0[X0['pickup_cluster'] == d]['pickup_latitude'].mean())

pickup_clusters_dict

#pd.DataFrame(dropoff_clusters_dict)
dist_dict = {}

for d, p in itertools.product(dropoff_clusters, pickup_clusters):

    #print(d, p, X0[(X0['dropoff_cluster'] == d) & (X0['pickup_cluster'] == p)]['trip_distance'].mean())

    dist_dict[(d, p)] = X0[(X0['dropoff_cluster'] == d) & (X0['pickup_cluster'] == p)]['trip_distance'].mean()

#print(dist_dict)
X3 = X0[['tpep_pickup_datetime']]
X3['time_of_day_mins'] = 60* X3['tpep_pickup_datetime'].dt.hour + X3['tpep_pickup_datetime'].dt.minute
X3.drop(['tpep_pickup_datetime'], axis=1, inplace=True)
no_classes = 12

est = KMeans(n_clusters=no_classes)

est.fit(X3)

labels = est.labels_
X0['time_of_day_cluster'] = labels
sns.distplot(X0['time_of_day_cluster'], kde=False)
#X0.to_csv('cluster_labels.csv', index=False)
X0 = X0[['passenger_count', 'pickup_cluster','dropoff_cluster','time_of_day_cluster']]

X0.head(5)
res = []

for tau in X0['time_of_day_cluster'].unique():

    print(tau)

    for d, p in itertools.product(dropoff_clusters, pickup_clusters):

        X_dum = X0[(X0['time_of_day_cluster'] == tau) & (X0['pickup_cluster'] == p) & (X0['dropoff_cluster'] == d)]

        tot_pass = X_dum['passenger_count'].sum()

        num_trips = X_dum.shape[0]

        total_miles = tot_pass  * dist_dict[(d, p)]

        miles_saved = (tot_pass - num_trips) * dist_dict[(d, p)] 

        res.append({'tau':tau,'d':d, 'p':p, 'tot_pass': tot_pass, 'num_trips': num_trips, 'total_miles': total_miles, 'miles_saved': miles_saved})
summary_stats = pd.DataFrame(res)

summary_stats['tau'] 
summary_stats.to_csv('summary_stats.csv', index=False)
print('Total miles saved for the yellow taxi dataset: {}'.format(summary_stats['miles_saved'].sum()))
print('Efficiency for the yellow taxi dataset: {}'.format(summary_stats['miles_saved'].sum()/summary_stats['total_miles'].sum()))