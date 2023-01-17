# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%time

# We'll load some important columns only

df = pd.read_csv('../input/train.csv',

                 usecols=['pickup_datetime', 'dropoff_datetime', 'passenger_count', 'pickup_longitude', 'pickup_latitude',

                          'dropoff_longitude', 'dropoff_latitude',  'trip_duration'])
# Let's see some records

df.head()
# size of the training set

print('Size:', df.shape[0])
from bokeh.plotting import figure, output_notebook, show # bokeh plotting library

# We'll show the plots in the cells of this notebook

output_notebook()
import numpy as np # linear algebra and Scientific calculation



# let's find pickup and dropoff longitude and latitude range

print(np.min(df['pickup_longitude']), np.min(df['pickup_latitude']))

print(np.max(df['pickup_longitude']), np.max(df['pickup_latitude']))



print(np.min(df['dropoff_longitude']), np.min(df['dropoff_latitude']))

print(np.max(df['dropoff_longitude']), np.max(df['dropoff_latitude']))
# NYC = x_range, y_range = ((-121.93334198, -61.3355293274), (32.1811408997, 51.8810844421))

NYC = x_range, y_range = ((-74.05, -73.7), (40.6, 40.9))



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
%%time

# let's plot 10k sample pickup

samples = df.sample(n=10000)

p = base_plot()



p.circle(x=samples['pickup_longitude'], y=samples['pickup_latitude'], **options)

show(p)
%%time

# Again, let's plot 10k sample dropoff

samples = df.sample(n=10000)

p = base_plot()



p.circle(x=samples['dropoff_longitude'], y=samples['dropoff_latitude'], **options)

show(p)
import datashader as ds

from datashader import transfer_functions as tr_fns

from datashader.colors import Greys9

Greys9_r = list(reversed(Greys9))[:2]
%%time

cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height, x_range=x_range, y_range=y_range)

agg = cvs.points(df, 'dropoff_longitude', 'dropoff_latitude', ds.count('passenger_count'))

img = tr_fns.shade(agg, cmap=["white", 'darkblue'], how='linear')



img
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

    agg = cvs.points(df, 'dropoff_longitude', 'dropoff_latitude', ds.count('passenger_count'))

    img = tr_fns.shade(agg, cmap=Hot, how='eq_hist')

    return tr_fns.dynspread(img, threshold=0.5, max_px=4)



p = base_plot(background_fill_color=background)

export(create_image(*NYC), "NYCT_hot")

InteractiveImage(p, create_image)
from functools import partial



def create_image90(x_range, y_range, w=plot_width, h=plot_height):

    cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)

    agg = cvs.points(df, 'dropoff_longitude', 'dropoff_latitude', ds.count('passenger_count'))

    img = tr_fns.shade(agg.where(agg > np.percentile(agg, 90)), cmap=inferno, how='eq_hist')

    return tr_fns.dynspread(img, threshold=0.3, max_px=4)

    

p = base_plot()

export(create_image(*NYC), "NYCT_90th")

InteractiveImage(p, create_image90)
def merged_images(x_range, y_range, w=plot_width, h=plot_height, how='log'):

    cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)

    picks = cvs.points(df, 'pickup_longitude', 'pickup_latitude', ds.count('passenger_count'))

    drops = cvs.points(df, 'dropoff_longitude', 'dropoff_latitude', ds.count('passenger_count'))

    more_drops = tr_fns.shade(drops.where(drops > picks), cmap=["darkblue", 'cornflowerblue'], how=how)

    more_picks = tr_fns.shade(drops.where(picks > drops), cmap=["darkred", 'orangered'], how=how)

    img = tr_fns.stack(more_picks, more_drops)

    return tr_fns.dynspread(img, threshold=0.3, max_px=4)



p = base_plot(background_fill_color=background)

export(merged_images(*NYC), "NYCT_pickups_vs_drops")

InteractiveImage(p, merged_images)