import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import ast
import geopandas as gpd
import matplotlib.cm as cm

import plotly.plotly as py
import plotly.graph_objs as go

from datetime import datetime

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
# load LA Calendar dataset
colnames = ["name", "desc", "fee", "type", "subject", "age", "service_area", "sponsor",
            "dt_start", "dt_end", "loc_name", "loc_address", "contact_name", "contact_number", 
            "contact_email", "web", "official_office", "official_name", "council", "ref_id"]
events_data = pd.read_csv("../input/what's-happening-la-calendar-dataset/whats-happening-la-calendar-dataset.csv",
                          names=colnames, header=0, usecols=range(18))
# parse dates
events_data["dt_start"] = pd.to_datetime(
    events_data["dt_start"], infer_datetime_format=True, errors="coerce")
events_data["dt_end"] = pd.to_datetime(
    events_data["dt_end"], infer_datetime_format=True, errors="coerce")
events_data.head()
dt_start = events_data[~events_data["dt_start"].isnull()]["dt_start"]
# leave only events already started (don't look planned events)
dt_start = dt_start[dt_start < datetime.now()]

events_by_month = dt_start.groupby([dt_start.dt.year, dt_start.dt.month]).agg('count')

# convert to dataframe
events_by_month = events_by_month.to_frame()
# move date month from index to column
events_by_month['date'] = events_by_month.index
# rename column
events_by_month = events_by_month.rename(columns={events_by_month.columns[0]:"events"})
# re-parse dates
events_by_month['date'] = pd.to_datetime(events_by_month['date'], format="(%Y, %m)")
# remove index
events_by_month = events_by_month.reset_index(drop=True)
# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis
data = [go.Scatter(x=events_by_month.date, y=events_by_month.events)]

# specify the layout of our figure
layout = dict(title = "Number of Events in LA per Month",
              xaxis= dict(title='Date',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)
# load Los Angeles districts shapefiles (used to draw LA on a map)
shapefile = gpd.read_file("../input/los-angeles-county-shapefiles/CAMS_ZIPCODE_PARCEL_SPECIFIC.shp")
# parse addresses
parsed_addrs = events_data["loc_address"].fillna("{}").apply(ast.literal_eval)
human_addrs = parsed_addrs.apply(lambda x: x.get("human_address", "{}"))
human_addrs = human_addrs.apply(ast.literal_eval)
zip_codes = human_addrs.apply(lambda x: x.get("zip", np.nan)).dropna()
zip_code_to_count = zip_codes.value_counts()
events_max_count = zip_code_to_count.max()
zip_code_to_count = zip_code_to_count.to_dict()
layout = dict(
    title = "Los Angeles Events by District",
    hovermode = 'closest',
    xaxis = dict(
        showgrid = False,
        zeroline = False,
    ),
    yaxis = dict(
        showgrid = False,
        zeroline = False,
        scaleanchor = "x",
    ),
    width=800,
    height=900,
    dragmode = 'select'
)
def to_color(x):
    rgba = cm.Purples(x)
    rgba = tuple(map(lambda x: int(x * 255.0), rgba))
    color_str = "rgb({},{},{})".format(rgba[0], rgba[1], rgba[2])
    return color_str
plot_data = []
SIMPLIFY_FACTOR = 0.005

x_centroids = []
y_centroids = []
centroid_texts = []
for index, row in shapefile.iterrows():
    centroid_text = "Zipcode: {}, Count: {}".format(row.ZIPCODE, zip_code_to_count.get(str(row.ZIPCODE), 0))
    if shapefile['geometry'][index].type == 'Polygon':
        x, y = row.geometry.simplify(SIMPLIFY_FACTOR).exterior.xy
        c_x, c_y = row.geometry.centroid.xy
        # latest plotly expects list, tuple or np.array (x is array.array here)
        x = np.frombuffer(x)
        y = np.frombuffer(y)
        
        x_centroids.append(c_x[0])
        y_centroids.append(c_y[0])
        centroid_texts.append(centroid_text)
        
    elif shapefile['geometry'][index].type == 'MultiPolygon':
        x = [poly.simplify(SIMPLIFY_FACTOR).exterior.xy[0] for poly in shapefile['geometry'][index]]
        y = [poly.simplify(SIMPLIFY_FACTOR).exterior.xy[1] for poly in shapefile['geometry'][index]]
        c_x = [poly.centroid.xy[0] for poly in shapefile['geometry'][index]]
        c_y = [poly.centroid.xy[1] for poly in shapefile['geometry'][index]]
        cts = [centroid_text for poly in shapefile['geometry'][index]]
        x_centroids += c_x
        y_centroids += c_y
        centroid_texts += cts
    else:
        print('stop', shapefile['geometry'][index].type)
        break
        
    county_outline = dict(
            type = 'scatter',
            line = dict(color='black', width=1),
            name = "ZipCode {}".format(row.ZIPCODE),
            x=x,
            y=y,
            fill = 'toself',
            fillcolor = to_color(zip_code_to_count.get(str(row.ZIPCODE), 0) / events_max_count),
            hoverinfo = 'none',
            mode = 'lines'
    )
    plot_data.append(county_outline)

hover_point = dict(
        type = 'scatter',
        showlegend = False,
        legendgroup = "centroids",
        name = "LA districts",
        text = centroid_texts,
        marker = dict(size=4, color='green'),
        x = x_centroids,
        y = y_centroids,
        mode = 'markers'
)
plot_data.append(hover_point)
# create and show our figure
fig = dict(data = plot_data, layout = layout)
iplot(fig, filename='la-cloropleth-map')
