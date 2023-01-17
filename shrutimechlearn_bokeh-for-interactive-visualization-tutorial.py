import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

# bokeh packages

from bokeh.io import output_file,show,output_notebook,push_notebook

from bokeh.plotting import figure

from bokeh.models import ColumnDataSource,HoverTool,CategoricalColorMapper

from bokeh.layouts import row,column,gridplot

from bokeh.models.widgets import Tabs,Panel

output_notebook()
from bokeh.io import output_file, show

from bokeh.plotting import figure



plot = figure(plot_width=400, tools='pan,box_zoom')

plot.circle([1,2,3,4,5], [8,6,5,2,3])

output_file('test.html')

show(plot)
plot = figure()

plot.circle(x=10, y=[2,5,8,12], size=[10,20,30,40])

show(plot)
from bokeh.io import output_file, show

from bokeh.plotting import figure

x = [1,2,3,4,5]

y = [8,6,5,2,3]

plot = figure()

plot.line(x, y, line_width=3)

output_file('line.html')

show(plot)
#Lines and Markers Together

from bokeh.io import output_file, show

from bokeh.plotting import figure

x = [1,2,3,4,5]

y = [8,6,5,2,3]

plot = figure()

plot.line(x, y, line_width=2)

plot.circle(x, y, fill_color='red', size=10)

output_file('line.html')

show(plot)
#Python Basic Types

from bokeh.io import output_file, show

from bokeh.plotting import figure

x = [1,2,3,4,5]

y = [8,6,5,2,3]

plot = figure()

plot.line(x, y, line_width=3)

plot.circle(x, y, fill_color='purple', size=10)

output_file('basic.html')

show(plot)
#NumPy Arrays

from bokeh.io import output_file, show

from bokeh.plotting import figure

import numpy as np

x = np.linspace(0, 10, 1000)

y = np.sin(x) + np.random.random(1000) * 0.2

plot = figure()

plot.line(x, y)

output_file('numpy.html')

show(plot)
#Pandas

from bokeh.io import output_file, show

from bokeh.plotting import figure

# Flowers is a Pandas DataFrame

from bokeh.sampledata.iris import flowers

plot = figure()

plot.circle(flowers['petal_length'],flowers['sepal_length'],size=10,fill_color = 'gold')

output_file('pandas.html')

show(plot)
#Hover appearance

from bokeh.models import HoverTool

hover = HoverTool(tooltips=None, mode='hline')

plot = figure(tools=[hover, 'crosshair'])

# x and y are lists of random points

plot.circle(x, y, size=15, hover_color='gold')

show(plot)