import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))
boston_schools = pd.read_csv('../input/Public_Schools.csv')
boston_schools.columns = map(str.lower, boston_schools.columns)
boston_schools.rename(columns={'x':'lon', 'y':'lat'}, inplace=True)
boston_schools.head()
# converts (lat,lon) into mercator_x and mercator_y coordinates
import math
def merc(Coords):
    lat = Coords[0]
    lon = Coords[1]
    
    r_major = 6378137.000
    x = r_major * math.radians(lon)
    scale = x/lon
    y = 180.0/math.pi * math.log(math.tan(math.pi/4.0 + 
        lat * (math.pi/180.0)/2.0)) * scale
    return x, y
# calculates mercator x and y coordinate columns
boston_schools['merc_x'], boston_schools['merc_y'] = ([merc((lat,lon))[i] for lat, lon in zip(boston_schools.lat, boston_schools.lon)] for i in range(2))
boston_schools.head()
from bokeh.plotting import figure, show, output_notebook
from bokeh.tile_providers import CARTODBPOSITRON
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.transform import factor_cmap
from bokeh.palettes import inferno

# Build a data source and configure for inline output
source = ColumnDataSource(boston_schools)
output_notebook()

# define x and y ranges
merc_x_range = [a(boston_schools.merc_x) for a in [min,max]]
merc_y_range = [a(boston_schools.merc_y) for a in [min,max]]


p = figure(x_range=merc_x_range, y_range=merc_y_range, x_axis_type="mercator", y_axis_type="mercator")
p.add_tile(CARTODBPOSITRON)

index_cmap = factor_cmap('city', palette=inferno(len(boston_schools.city.unique())), factors=sorted(boston_schools.city.unique()), end=1)
p.circle(source=source, x='merc_x', y='merc_y', fill_color=index_cmap, fill_alpha=0.2, radius=500)
p.circle(source=source, x='merc_x', y='merc_y',fill_color='black', radius=100)

p.add_tools(HoverTool(tooltips=[("Name", "@sch_name"),("Address", "@address, @city, @zipcode"),("PL", "@pl")]))
show(p)