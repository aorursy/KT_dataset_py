# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import the necessary libraries and modulles
from bokeh.plotting import figure, show
from bokeh.models.tools import HoverTool
from bokeh.io import output_notebook, push_notebook

output_notebook()
# load dataset
dataset = pd.read_csv('/kaggle/input/current-health-expenditure-per-capita/che_per_capita_by_region.csv', index_col=0) 
# Check uploaded dataset
dataset
# Re-index the dataset to reorder existing data
dataset = dataset.reindex(columns=['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', 
                                        '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'])
# We check the dataset again
dataset
# List comprehension to iterate through each column and store each year as a list in the years variable
years = [year for year in dataset.columns]

# List comprehension to iterate the list years. For each year, we store its value as the second label 
# of the 'df.loc' method and access each rows (regions) with their column (year)
che_afr = [dataset.loc['Africa', year] for year in years]
che_ame = [dataset.loc['Americas', year] for year in years]
che_sea = [dataset.loc['South-East Asia', year] for year in years]
che_eur = [dataset.loc['Europe', year] for year in years]
che_eam = [dataset.loc['Eastern Mediterranean', year] for year in years]
che_wep = [dataset.loc['Western Pacific', year] for year in years]

# Create a new Figure for plotting
plot = figure(plot_width=900,
           title='Trend in current health expenditure (CHE) per capita in US$ by WHO region',
           x_axis_label='Year',
           y_axis_label='(CHE) per capita in US$',
           tools="pan,box_zoom,reset,save",)

# Add tooltips
plot.add_tools(HoverTool(
    tooltips=[
        ('(Year, CHE)', '(@x, @y{$0,0.00})')
    ]
))

# Add Circle and Lines glyphs to this Figure
plot.line(years, che_afr, line_width=3, color='coral', legend_label='Africa')
plot.circle(years, che_afr, size=8, line_color='coral', fill_color='white', legend_label='Africa')

plot.line(years, che_ame, line_width=3, color='darkcyan', legend_label='Americas')
plot.circle(years, che_ame, size=8, line_color='darkcyan', fill_color='white', legend_label='Americas')

plot.line(years, che_sea, line_width=3, color='darkslategrey', legend_label='South-East Asia')
plot.circle(years, che_sea, size=8, line_color='darkslategrey', fill_color='white', legend_label='South-East Asia')

plot.line(years, che_eur, line_width=3, color='brown', legend_label='Europe')
plot.circle(years, che_eur, size=8, line_color='brown', fill_color='white', legend_label='Europe')

plot.line(years, che_eam, line_width=3, color='darkgreen', legend_label='Eastern Mediterranean')
plot.circle(years, che_eam, size=8, line_color='darkgreen', fill_color='white', legend_label='Eastern Mediterranean')

plot.line(years, che_wep, line_width=3, color='indigo', legend_label='Western Pacific')
plot.circle(years, che_wep, size=8, line_color='indigo', fill_color='white', legend_label='Western Pacific')

# Add location, title, font-style of the legend labels
plot.legend.location = "top_left"
plot.legend.title = 'WHO Region'
plot.legend.title_text_font_style = "bold"
plot.legend.title_text_font_size = "18px"

# Enable hide/show lines
plot.legend.click_policy = 'hide'

# change just some things about the y-grid
plot.ygrid.minor_grid_line_color = 'navy'
plot.ygrid.minor_grid_line_alpha = 0.1

# show plot
show(plot)