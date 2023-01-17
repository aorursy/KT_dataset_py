import sqlite3

import pandas as pd

import numpy as np

import colorcet as cc

from bokeh.io import output_notebook

from bokeh.plotting import figure, show

from bokeh.models import ColumnDataSource, LogColorMapper
cnx = sqlite3.connect('../input/FPA_FOD_20170508.sqlite')

df = pd.read_sql_query("SELECT LATITUDE, LONGITUDE, FIRE_SIZE, STATE FROM fires", cnx)

df.head(5)
pd.options.mode.chained_assignment = None
new = df.loc[(df.loc[:,'STATE']!='AK') & (df.loc[:,'STATE']!='HI') & (df.loc[:,'STATE']!='PR')]
new.loc[:,'LATITUDE'] = ((new.loc[:,'LATITUDE']*10).apply(np.floor))/10

new.loc[:,'LONGITUDE'] = ((new.loc[:,'LONGITUDE']*10).apply(np.floor))/10

new.loc[:,'LL_COMBO'] = new.loc[:,'LATITUDE'].map(str) + '-' + new.loc[:,'LONGITUDE'].map(str)

grouped = new.groupby(['LL_COMBO', 'LATITUDE', 'LONGITUDE'])
number_of_wf = grouped['FIRE_SIZE'].agg(['count']).reset_index()

number_of_wf.head(5)
size_of_wf = grouped['FIRE_SIZE'].agg(['mean']).reset_index()

size_of_wf.head(5)
source = ColumnDataSource(number_of_wf)

p1 = figure(title="Number of wildfires occurring from 1992 to 2015 " + \

            "(lighter color means more wildfires)",

           toolbar_location=None, plot_width=600, plot_height=400)

p1.background_fill_color = "black"

p1.grid.grid_line_color = None

p1.axis.visible = False

color_mapper = LogColorMapper(palette=cc.fire)

glyph = p1.circle('LONGITUDE', 'LATITUDE', source=source,

          color={'field': 'count', 'transform' : color_mapper},

          size=1)

output_notebook()

show(p1)
source = ColumnDataSource(size_of_wf)

p2 = figure(title="Average size of wildfires occurring from 1992 to 2015 " + \

            "(lighter color means bigger fire)",

           toolbar_location=None, plot_width=600, plot_height=400)

p2.background_fill_color = "black"

p2.grid.grid_line_color = None

p2.axis.visible = False

glyph = p2.circle('LONGITUDE', 'LATITUDE', source=source,

          color={'field': 'mean', 'transform' : color_mapper},

          size=1)

show(p2)