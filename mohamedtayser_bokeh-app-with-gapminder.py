# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#get data from file

data=pd.read_csv('/kaggle/input/gapminder/gapminder_tidy.csv')
# Perform necessary imports



from bokeh.io import output_file, show,output_notebook,push_notebook

from bokeh.plotting import figure

from bokeh.models import HoverTool, ColumnDataSource



# Make the ColumnDataSource: source

source = ColumnDataSource(data={  

    'x'       : data[data['Year'] == 1970]['fertility'],

    'y'       : data[data['Year'] == 1970]['life'],

    'country'  : data[data['Year'] == 1970]['Country']

})



# Create the figure: p

p = figure(title='1970', x_axis_label='Fertility (children per woman)', y_axis_label='Life Expectancy (years)',

           plot_height=400, plot_width=700,

           tools=[HoverTool(tooltips='@country')])



# Add a circle glyph to the figure p

p.circle(x='x', y='y', source=source)



# Output the file and show the figure

output_file('gapminder.html')

show(p)



#To show the figure in kaggle noteboke

output_file("gapminder.html", title="Gapminder")

output_notebook()
# Import the necessary modules

from bokeh.io import curdoc

from bokeh.models import ColumnDataSource

from bokeh.plotting import figure



# Make the ColumnDataSource: source

source = ColumnDataSource(data={

    'x'       : data[data['Year'] == 1970]['fertility'],

    'y'       : data[data['Year'] == 1970]['life'],

    'country' : data[data['Year'] == 1970]['Country'],

    'pop'     :(data[data['Year'] == 1970]['population'] / 20000000)  + 2,

    'region'  : data[data['Year'] == 1970]['region']

})



# Save the minimum and maximum values of the fertility column: xmin, xmax

xmin, xmax = min(data.fertility), max(data.fertility)



# Save the minimum and maximum values of the life expectancy column: ymin, ymax

ymin, ymax = min(data.life), max(data.life)



# Create the figure: plot

plot = figure(title='Gapminder Data for 1970', plot_height=400, plot_width=700, x_range=(xmin, xmax), y_range=(ymin, ymax))



# Add circle glyphs to the plot

plot.circle(x='x', y='y', fill_alpha=0.8, source=source)



# Set the x-axis label

plot.xaxis.axis_label ='Fertility (children per woman)'



# Set the y-axis label

plot.yaxis.axis_label = 'Life Expectancy (years)'



# Output the file and show the figure

output_file('gapminder.html')

show(plot)



#To show the figure in kaggle noteboke

output_file("gapminder.html", title="Gapminder")

output_notebook()
# Make a list of the unique values from the region column: regions_list

regions_list = data.region.unique().tolist()



# Import CategoricalColorMapper from bokeh.models and the Spectral6 palette from bokeh.palettes

from bokeh.models import CategoricalColorMapper

from bokeh.palettes import Spectral6



# Make a color mapper: color_mapper

color_mapper = CategoricalColorMapper(factors=regions_list, palette=Spectral6)



# Add the color mapper to the circle glyph

plot.circle(x='x', y='y', fill_alpha=0.8, source=source,

            color=dict(field='region', transform=color_mapper), legend='region')



# Set the legend.location attribute of the plot to 'top_right'

plot.legend.location = 'top_right'



# Output the file and show the figure

output_file('gapminder.html')

show(plot)



#To show the figure in kaggle noteboke

output_file("gapminder.html", title="Gapminder")

output_notebook()
# Import the necessary modules

from bokeh.layouts import widgetbox, row

from bokeh.models import Slider

from bokeh.client import push_session, pull_session

from bokeh.models import ColumnDataSource, Div, Select, Button, ColorBar, CustomJS



# Define the callback function: update_plot

def update_plot(attr, old, new):

    # set the `yr` name to `slider.value` and `source.data = new_data`

    yr = slider.value

    new_data = {

    'x'       : data[data['Year'] == yr]['fertility'],

    'y'       : data[data['Year'] == yr]['life'],

    'country' : data[data['Year'] == yr]['Country'],

    'pop'     :(data[data['Year'] == yr]['population'] / 20000000)  + 2,

    'region'  : data[data['Year'] == yr]['region']

    }

    source.data = new_data





# Make a slider object: slider

slider = Slider(start=1970, end=2010, step=1, value=1970, title='Year')



# Attach the callback to the 'value' property of slider

slider.on_change('value',update_plot)



# Make a row layout of widgetbox(slider) and plot and add it to the current document

layout = row(widgetbox(slider), plot)

curdoc().add_root(layout)







# Output the file and show the figure

output_file('gapminder.html')

show(layout)



#To show the figure in kaggle noteboke

output_file("gapminder.html", title="Gapminder")

output_notebook()


