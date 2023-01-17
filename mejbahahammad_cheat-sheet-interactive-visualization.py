# Standard imports 



from bokeh.io import output_notebook, show

output_notebook()
# Plot a complex chart with interactive hover in a few lines of code



from bokeh.models import ColumnDataSource, HoverTool

from bokeh.plotting import figure

from bokeh.sampledata.autompg import autompg_clean as df

from bokeh.transform import factor_cmap



df.cyl = df.cyl.astype(str)

df.yr = df.yr.astype(str)



group = df.groupby(by=['cyl', 'mfr'])

source = ColumnDataSource(group)



p = figure(plot_width=800, plot_height=300, title="Mean MPG by # Cylinders and Manufacturer",

           x_range=group, toolbar_location=None, tools="")



p.xgrid.grid_line_color = None

p.xaxis.axis_label = "Manufacturer grouped by # Cylinders"

p.xaxis.major_label_orientation = 1.2



index_cmap = factor_cmap('cyl_mfr', palette=['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c'], 

                         factors=sorted(df.cyl.unique()), end=1)



p.vbar(x='cyl_mfr', top='mpg_mean', width=1, source=source,

       line_color="white", fill_color=index_cmap, 

       hover_line_color="darkgrey", hover_fill_color=index_cmap)



p.add_tools(HoverTool(tooltips=[("MPG", "@mpg_mean"), ("Cyl, Mfr", "@cyl_mfr")]))



show(p)
# Create and deploy interactive data applications



from IPython.display import IFrame

IFrame('https://demo.bokeh.org/sliders', width=900, height=500)
from IPython import __version__ as ipython_version

from pandas import __version__ as pandas_version

from bokeh import __version__ as bokeh_version

print("IPython - %s" % ipython_version)

print("Pandas - %s" % pandas_version)

print("Bokeh - %s" % bokeh_version)
import numpy as np # we will use this later, so import it now



from bokeh.io import output_notebook, show

from bokeh.plotting import figure
output_notebook()
import bokeh.sampledata

bokeh.sampledata.download()
# create a new plot with default tools, using figure

p = figure(plot_width=400, plot_height=400)



# add a circle renderer with x and y coordinates, size, color, and alpha

p.circle([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], size=15, line_color="navy", fill_color="orange", fill_alpha=0.5)



show(p) # show the results
# create a new plot using figure

p = figure(plot_width=400, plot_height=400)



# add a square renderer with a size, color, alpha, and sizes

p.square([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], size=[10, 15, 20, 25, 30], color="firebrick", alpha=0.6)



show(p) # show the results
# create a new plot (with a title) using figure

p = figure(plot_width=400, plot_height=400, title="My Line Plot")



# add a line renderer

p.line([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], line_width=2)



show(p) # show the results
from bokeh.sampledata.glucose import data

data.head()
# reduce data size to one week

week = data.loc['2010-10-01':'2010-10-08']



p = figure(x_axis_type="datetime", title="Glocose Range", plot_height=350, plot_width=800)

p.xgrid.grid_line_color=None

p.ygrid.grid_line_alpha=0.5

p.xaxis.axis_label = 'Time'

p.yaxis.axis_label = 'Value'



p.line(week.index, week.glucose)



show(p)
# EXERCISE: Look at the AAPL data from bokeh.sampledata.stocks and create a line plot using it

from bokeh.sampledata.stocks import AAPL



# AAPL.keys()

# dict_keys(['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close'])



dates = np.array(AAPL['date'], dtype=np.datetime64) # convert date strings to real datetimes



from bokeh.palettes import Viridis256

from bokeh.util.hex import hexbin



n = 50000

x = np.random.standard_normal(n)

y = np.random.standard_normal(n)



bins = hexbin(x, y, 0.1)



# color map the bins by hand, will see how to use linear_cmap later

color = [Viridis256[int(i)] for i in bins.counts/max(bins.counts)*255]



# match_aspect ensures neither dimension is squished, regardless of the plot size

p = figure(tools="wheel_zoom,reset", match_aspect=True, background_fill_color='#440154')

p.grid.visible = False



p.hex_tile(bins.q, bins.r, size=0.1, line_color=None, fill_color=color)



show(p)
N = 500

x = np.linspace(0, 10, N)

y = np.linspace(0, 10, N)

xx, yy = np.meshgrid(x, y)



img = np.sin(xx)*np.cos(yy)



p = figure(x_range=(0, 10), y_range=(0, 10))



# must give a vector of image data for image parameter

p.image(image=[img], x=0, y=0, dw=10, dh=10, palette="Spectral11")



show(p)  
from __future__ import division

import numpy as np

 

N = 20

img = np.empty((N,N), dtype=np.uint32) 



# use an array view to set each RGBA channel individiually

view = img.view(dtype=np.uint8).reshape((N, N, 4))

for i in range(N):

    for j in range(N):

        view[i, j, 0] = int(i/N*255) # red

        view[i, j, 1] = 158          # green

        view[i, j, 2] = int(j/N*255) # blue

        view[i, j, 3] = 255          # alpha

        

# create a new plot (with a fixed range) using figure

p = figure(x_range=[0,10], y_range=[0,10])



# add an RGBA image renderer

p.image_rgba(image=[img], x=[0], y=[0], dw=[10], dh=[10])



show(p) 
# set up some data

x = [1, 2, 3, 4, 5]

y = [6, 7, 8, 7, 3]



# create a new plot with figure

p = figure(plot_width=400, plot_height=400)



# add both a line and circles on the same plot

p.line(x, y, line_width=2)

p.circle(x, y, fill_color="white", size=8)



show(p) # show the results
# create a new plot with a title

p = figure(plot_width=400, plot_height=400)

p.outline_line_width = 7

p.outline_line_alpha = 0.3

p.outline_line_color = "navy"



p.circle([1,2,3,4,5], [2,5,8,2,7], size=10)



show(p)
p = figure(plot_width=400, plot_height=400)



# keep a reference to the returned GlyphRenderer

r = p.circle([1,2,3,4,5], [2,5,8,2,7])



r.glyph.size = 50

r.glyph.fill_alpha = 0.2

r.glyph.line_color = "firebrick"

r.glyph.line_dash = [5, 1]

r.glyph.line_width = 2



show(p)
p = figure(plot_width=400, plot_height=400, tools="tap", title="Select a circle")

renderer = p.circle([1, 2, 3, 4, 5], [2, 5, 8, 2, 7], size=50,



                    # set visual properties for selected glyphs

                    selection_color="firebrick",



                    # set visual properties for non-selected glyphs

                    nonselection_fill_alpha=0.2,

                    nonselection_fill_color="grey",

                    nonselection_line_color="firebrick",

                    nonselection_line_alpha=1.0)



show(p)
from bokeh.models.tools import HoverTool

from bokeh.sampledata.glucose import data



subset = data.loc['2010-10-06']



x, y = subset.index.to_series(), subset['glucose']



# Basic plot setup

p = figure(width=600, height=300, x_axis_type="datetime", title='Hover over points')



p.line(x, y, line_dash="4 4", line_width=1, color='gray')



cr = p.circle(x, y, size=20,

              fill_color="grey", hover_fill_color="firebrick",

              fill_alpha=0.05, hover_alpha=0.3,

              line_color=None, hover_line_color="white")



p.add_tools(HoverTool(tooltips=None, renderers=[cr], mode='hline'))



show(p)
from math import pi



p = figure(plot_width=400, plot_height=400)

p.x([1,2,3,4,5], [2,5,8,2,7], size=10, line_width=2)



p.xaxis.major_label_orientation = pi/4

p.yaxis.major_label_orientation = "vertical"



show(p)
p = figure(plot_width=400, plot_height=400)

p.asterisk([1,2,3,4,5], [2,5,8,2,7], size=12, color="olive")



# change just some things about the x-axes

p.xaxis.axis_label = "Temp"

p.xaxis.axis_line_width = 3

p.xaxis.axis_line_color = "red"



# change just some things about the y-axes

p.yaxis.axis_label = "Pressure"

p.yaxis.major_label_text_color = "orange"

p.yaxis.major_label_orientation = "vertical"



# change things on all axes

p.axis.minor_tick_in = -3

p.axis.minor_tick_out = 6



show(p)
from math import pi

from bokeh.sampledata.glucose import data



week = data.loc['2010-10-01':'2010-10-08']



p = figure(x_axis_type="datetime", title="Glocose Range", plot_height=350, plot_width=800)

p.xaxis.formatter.days = '%m/%d/%Y'

p.xaxis.major_label_orientation = pi/3



p.line(week.index, week.glucose)



show(p)
from bokeh.models import NumeralTickFormatter



p = figure(plot_height=300, plot_width=800)

p.circle([1,2,3,4,5], [2,5,8,2,7], size=10)



p.xaxis.formatter = NumeralTickFormatter(format="0.0%")

p.yaxis.formatter = NumeralTickFormatter(format="$0.00")



show(p)
p = figure(plot_width=400, plot_height=400)

p.circle([1,2,3,4,5], [2,5,8,2,7], size=10)



# change just some things about the x-grid

p.xgrid.grid_line_color = None



# change just some things about the y-grid

p.ygrid.grid_line_alpha = 0.5

p.ygrid.grid_line_dash = [6, 4]



show(p)
p = figure(plot_width=400, plot_height=400)

p.circle([1,2,3,4,5], [2,5,8,2,7], size=10)



# change just some things about the x-grid

p.xgrid.grid_line_color = None



# change just some things about the y-grid

p.ygrid.band_fill_alpha = 0.1

p.ygrid.band_fill_color = "navy"



show(p)
from bokeh.models import ColumnDataSource
source = ColumnDataSource(data={

    'x' : [1, 2, 3, 4, 5],

    'y' : [3, 7, 8, 5, 1],

})
p = figure(plot_width=400, plot_height=400)

p.circle('x', 'y', size=20, source=source)

show(p)
from bokeh.sampledata.iris import flowers as df



source = ColumnDataSource(df)
p = figure(plot_width=400, plot_height=400)

p.circle('petal_length', 'petal_width', source=source)

show(p)
# Exercise: create a column data source with the autompg sample data frame and plot it



from bokeh.sampledata.autompg import autompg_clean as df

from bokeh.sampledata.iris import flowers as df



p = figure(plot_width=400, plot_height=400)

p.circle('petal_length', 'petal_width', source=df)

show(p)
from math import pi

import pandas as pd

from bokeh.palettes import Category20c

from bokeh.transform import cumsum



x = { 'United States': 157, 'United Kingdom': 93, 'Japan': 89, 'China': 63,

      'Germany': 44, 'India': 42, 'Italy': 40, 'Australia': 35, 'Brazil': 32,

      'France': 31, 'Taiwan': 31, 'Spain': 29 }



data = pd.Series(x).reset_index(name='value').rename(columns={'index':'country'})

data['color'] = Category20c[len(x)]



# represent each value as an angle = value / total * 2pi

data['angle'] = data['value']/data['value'].sum() * 2*pi



p = figure(plot_height=350, title="Pie Chart", toolbar_location=None,

           tools="hover", tooltips="@country: @value")



p.wedge(x=0, y=1, radius=0.4, 

        

        # use cumsum to cumulatively sum the values for start and end angles

        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),

        line_color="white", fill_color='color', legend_field='country', source=data)



p.axis.axis_label=None

p.axis.visible=False

p.grid.grid_line_color = None



show(p)
from bokeh.transform import linear_cmap



N = 4000

data = dict(x=np.random.random(size=N) * 100,

            y=np.random.random(size=N) * 100,

            r=np.random.random(size=N) * 1.5)



p = figure()



p.circle('x', 'y', radius='r', source=data, fill_alpha=0.6,

        

         # color map based on the x-coordinate

         color=linear_cmap('x', 'Viridis256', 0, 100))



show(p) 