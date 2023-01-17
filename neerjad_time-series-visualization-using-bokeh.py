# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
from bokeh.plotting import figure, show, output_file, output_notebook
from bokeh.palettes import Spectral11, colorblind, Inferno, BuGn, brewer
from bokeh.models import HoverTool, value, LabelSet, Legend, ColumnDataSource,LinearColorMapper,BasicTicker, PrintfTickFormatter, ColorBar
import datetime

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
output_notebook()

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Monthly_Property_Crime_2005_to_2015.csv', parse_dates=['Date'])
data.head()
data.Date.min(), data.Date.max()
data.Category.value_counts()
data['Year'] = data.Date.apply(lambda x: x.year)
data['Month'] = data.Date.apply(lambda x: x.month)
data.head()
temp_df = data.groupby(['Month']).mean().reset_index()
temp_df.head()
TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom,tap"
p = figure(plot_height=350,
    title="Average Number of Crimes by Month",
    tools=TOOLS,
    toolbar_location='above')

p.vbar(x=temp_df.Month, top=temp_df.IncidntNum, width=0.9)

p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.axis.minor_tick_line_color = None
p.outline_line_color = None
p.xaxis.axis_label = 'Month'
p.yaxis.axis_label = 'Average Crimes'
p.select_one(HoverTool).tooltips = [
    ('month', '@x'),
    ('Number of crimes', '@top'),
]
output_file("barchart.html", title="barchart")
show(p)
temp_df = data.groupby(['Year']).sum().reset_index()
temp_df.head()
TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
p = figure(title="Year-wise total number of crimes", y_axis_type="linear", plot_height = 400,
           tools = TOOLS, plot_width = 800)
p.xaxis.axis_label = 'Year'
p.yaxis.axis_label = 'Total Crimes'
p.circle(2010, temp_df.IncidntNum.min(), size = 10, color = 'red')

p.line(temp_df.Year, temp_df.IncidntNum,line_color="purple", line_width = 3)
p.select_one(HoverTool).tooltips = [
    ('year', '@x'),
    ('Number of crimes', '@y'),
]

output_file("line_chart.html", title="Line Chart")
show(p)
wide = data.pivot(index='Date', columns='Category', values='IncidntNum')
wide.reset_index(inplace=True)
wide['Year'] = wide.Date.apply(lambda x: x.year)
wide['Month'] = wide.Date.apply(lambda x: x.month)

temp_df = wide.groupby(['Year']).sum().reset_index()
temp_df.head()
cats = ['ARSON','BURGLARY','LARCENY/THEFT','STOLEN PROPERTY','VANDALISM','VEHICLE THEFT'] 
temp_df.drop(['Month'], axis = 1, inplace=True)
temp_df.head()
TOOLS = "save,pan,box_zoom,reset,wheel_zoom,tap"

source = ColumnDataSource(data=temp_df)
p = figure( plot_width=800, title="Category wise count of crimes by year",toolbar_location='above', tools=TOOLS)
colors = brewer['Dark2'][6]

p.vbar_stack(cats, x='Year', width=0.9, color=colors, source=source,
             legend=[value(x) for x in cats])

p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.axis.minor_tick_line_color = None
p.outline_line_color = None
p.xaxis.axis_label = 'Year'
p.yaxis.axis_label = 'Total Crimes'
p.legend.location = "top_left"
p.legend.orientation = "horizontal"

output_file("stacked_bar.html", title="Stacked Bar Chart")

show(p)
temp_df = data.groupby(['Year', 'Month']).sum().reset_index()
# temp_df['Month_Category'] = pd.concat([temp_df['Month'], temp_df['Category']], axis = 1)
temp_df.head()
TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom,tap"
hm = figure(title="Month-Year wise crimes", tools=TOOLS, toolbar_location='above')

source = ColumnDataSource(temp_df)
colors = brewer['BuGn'][9]
colors = colors[::-1]
mapper = LinearColorMapper(
    palette=colors, low=temp_df.IncidntNum.min(), high=temp_df.IncidntNum.max())
hm.rect(x="Year", y="Month",width=2,height=1,source = source,  
    fill_color={
        'field': 'IncidntNum',
        'transform': mapper
    },
    line_color=None)
color_bar = ColorBar(
    color_mapper=mapper,
    major_label_text_font_size="10pt",
    ticker=BasicTicker(desired_num_ticks=len(colors)),
    formatter=PrintfTickFormatter(),
    label_standoff=6,
    border_line_color=None,
    location=(0, 0))

hm.add_layout(color_bar, 'right')
hm.xaxis.axis_label = 'Year'
hm.yaxis.axis_label = 'Month'
hm.select_one(HoverTool).tooltips = [
    ('Year', '@Year'),('Month', '@Month'), ('Number of Crimes', '@IncidntNum')
]

output_file("heatmap.html", title="Heat Map")

show(hm)  # open a browser
burglary = data[data.Category == 'BURGLARY'].sort_values(['Date'])
stolen_property = data[data.Category == 'STOLEN PROPERTY'].sort_values(['Date'])
vehicle_theft = data[data.Category == 'VEHICLE THEFT'].sort_values(['Date'])
vandalism = data[data.Category == 'VANDALISM'].sort_values(['Date'])
larceny = data[data.Category == 'LARCENY/THEFT'].sort_values(['Date'])
arson = data[data.Category == 'ARSON'].sort_values(['Date'])
arson.head()
TOOLS = 'crosshair,save,pan,box_zoom,reset,wheel_zoom'
p = figure(title="Category-wise crimes through Time", y_axis_type="linear",x_axis_type='datetime', tools = TOOLS)

p.line(burglary['Date'], burglary.IncidntNum, legend="burglary", line_color="purple", line_width = 3)
p.line(stolen_property['Date'], stolen_property.IncidntNum, legend="stolen_property", line_color="blue", line_width = 3)

p.line(vehicle_theft['Date'], vehicle_theft.IncidntNum, legend="vehicle_theft", line_color = 'coral', line_width = 3)

p.line(larceny['Date'], larceny.IncidntNum, legend="larceny", line_color='green', line_width = 3)

p.line(vandalism['Date'], vandalism.IncidntNum, legend="vandalism", line_color="gold", line_width = 3)

p.line(arson['Date'], arson.IncidntNum, legend="arson", line_color="magenta",line_width = 3)

p.legend.location = "top_left"

p.xaxis.axis_label = 'Year'
p.yaxis.axis_label = 'Count'

output_file("multiline_plot.html", title="Multi Line Plot")

show(p)  # open a browser

