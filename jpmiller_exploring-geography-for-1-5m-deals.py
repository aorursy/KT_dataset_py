import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial

import colorcet as cc
from bokeh.models import BoxZoomTool
from bokeh.plotting import figure, output_notebook, show
from bokeh.tile_providers import STAMEN_TERRAIN_RETINA
import datashader as ds
from datashader import transfer_functions as txf
from datashader.bokeh_ext import InteractiveImage
from datashader.utils import export_image
#import data
train = pd.read_csv('../input/avito-demand-prediction/train.csv', usecols = ['city',  'region', 'deal_probability'])
train['location'] = train['city'] + ', ' + train['region']


# # get coordinates from Google - for running locally

# import googlemaps
# gmaps = googlemaps.Client(key='yourAPIkey')
# locations = train['location'].unique()

# queries = []
# for l in tqdm(locations):
#     coords = gmaps.geocode(l, language='ru', region='ru')
#     queries.append(coords)

# coordlist = []
# for i in range(len(queries)):
#     if not queries[i]:
#         coordlist.append([50, 50])
#     else:
#         q=queries[i][0]
#         qpair = (list(q['geometry']['location'].values()))
#         coordlist.append(qpair)

# lats = [c[0] for c in coordlist]
# lons = [c[1] for c in coordlist]

# locs_df = pd.DataFrame(np.column_stack([locations, lons, lats]), columns = ['location', 'lon', 'lat'])

# locs_df['lon'] = pd.to_numeric(locs_df['lon'])
# locs_df['lat'] = pd.to_numeric(locs_df['lat'])
# get coordinates from file
locs_df = pd.read_csv('../input/russian-cities/city_latlons.csv')

# merge dataframes and convert to Mercator
train = train.merge(locs_df, how='left', on='location')

def merc_from_arrays(lons, lats):
    r_major = 6378137.000
    x = r_major * np.radians(lons)
    scale = x/lons
    y = 180.0/np.pi * np.log(np.tan(np.pi/4.0 + lats * (np.pi/180.0)/2.0)) * scale
    return (x, y)

train['deal_x'], train['deal_y'] = merc_from_arrays(train['lon'].values, train['lat'].values)
train.head()
# make the map
output_notebook()
RU = x_range, y_range = ((2050000, 12720000), (5250000, 12000000))
plot_width  = int(750)
plot_height = int(plot_width//1.6)

def base_plot(tools='pan,wheel_zoom,reset',plot_width=plot_width, plot_height=plot_height, **plot_args):
    p = figure(tools=tools, plot_width=plot_width, plot_height=plot_height,
        x_range=x_range, y_range=y_range, outline_line_color=None,
        min_border=0, min_border_left=0, min_border_right=0,
        min_border_top=0, min_border_bottom=0, **plot_args)
    p.axis.visible = False
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.add_tools(BoxZoomTool(match_aspect=True))
    return p

background = "black"
export = partial(export_image, export_path="export", background=background)

def colorized_images(x_range, y_range, w=plot_width, h=plot_height):   
    cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
    agg = cvs.points(train, 'deal_x', 'deal_y', ds.count('deal_probability'))   # reference to data
    img = txf.shade(agg, cmap=list(reversed(cc.fire)), how='eq_hist')   
    return txf.dynspread(img, threshold=0.3, max_px=4)

p = base_plot(background_fill_color=background)
p.add_tile(STAMEN_TERRAIN_RETINA)
export(colorized_images(*RU),"Avito_Deals")
InteractiveImage(p, colorized_images)
