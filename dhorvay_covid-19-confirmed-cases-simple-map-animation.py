import time

import pandas as pd

import numpy as np

from bokeh.plotting import figure, show

from bokeh.io import output_notebook, push_notebook

from bokeh.models import ColumnDataSource, HoverTool

from bokeh.tile_providers import get_provider

from pyproj import Proj, transform
# Read data

df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

df.head()
# Get total confirm as of today

df['TotalConfirmed'] = df.iloc[:,-1]

# Rename columns for easier access for tooltips

df.rename(columns={'Province/State':'ProvinceState', 'Country/Region':'CountryRegion'}, inplace=True) 

df.head()
# Helper function to convert latitude/longitude to Web Mercator for mapping

# See: https://gis.stackexchange.com/questions/247871/convert-gps-coordinates-to-web-mercator-epsg3857-using-python-pyproj

def to_web_mercator(long, lat):

    try:

        return transform(Proj(init='epsg:4326'), Proj(init='epsg:3857'), long, lat)

    except:

        return None, None

    

# Convert all latitude/longitude to Web Mercator and stores in new easting and northing columns

df['E'], df['N'] = zip(*df.apply(lambda x: to_web_mercator(x['Long'], x['Lat']), axis=1))

df['Size'] = 0

source = ColumnDataSource(df)
title = "COVID-19 | Growth of confirmed cases {} - {}".format(df.columns[4], df.columns[df.shape[1]-5])

hover = HoverTool(tooltips=[("Province / State", "@ProvinceState"),

                            ("Country / Region", "@CountryRegion"),

                            ("Total confirmed cases as of today", "@TotalConfirmed")])

p = figure(plot_width=800, plot_height=500, title=title, tools=[hover, 'pan', 'wheel_zoom','save', 'reset'])

tile_provider = get_provider('CARTODBPOSITRON_RETINA')

p.add_tile(tile_provider)

p.circle(x='E', y='N', source=source, line_color='grey', fill_color='red', alpha=0.7, size='Size')

output_notebook()



handle = show(p, notebook_handle = True)

day = 4 # Time series data starts at the 4th col

today = df.shape[1]-5 # Time series data stops here since we added 4 new columns

max_size = 20.0 # Represents 5000+ cases

while day < today:

    df['Size'] = np.where(df.iloc[:,day]*.004<=max_size, df.iloc[:,day]*.004, max_size)

    # Push new data

    source.stream(df)

    # Purge old data

    source.data = df

    push_notebook(handle=handle)

    day += 1

    time.sleep(.5)