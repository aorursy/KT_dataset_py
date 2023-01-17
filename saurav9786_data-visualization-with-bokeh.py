# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
auto_df=pd.read_csv("/kaggle/input/automobile-dataset/Automobile_data.csv")
#Importing libraries 



from bokeh.io import output_file,show,output_notebook,push_notebook

from bokeh.plotting import figure

from bokeh.models import ColumnDataSource,HoverTool,CategoricalColorMapper

from bokeh.layouts import row,column,gridplot

from bokeh.models.widgets import Tabs,Panel

output_notebook()
# Scatter Markers



p = figure(plot_width=500, plot_height=500)



# add a circle renderer with a size, color, and alpha

p.circle(auto_df["engine-size"], auto_df["wheel-base"], size=20, color="navy", alpha=0.5)



# show the results

show(p)
p = figure(plot_width=500, plot_height=500)



# add a square renderer with a size, color, and alpha

p.square(auto_df["engine-size"], auto_df["wheel-base"], size=20, color="green", alpha=0.5)



# show the results

show(p)
from bokeh.transform import factor_cmap, factor_mark

BODY_STYLE = ['sedan', 'hatchback', 'Other']

MARKERS = ['hex', 'circle_x', 'triangle']



p = figure(title = "Automobile Dataset")

p.xaxis.axis_label = 'Engine Size'

p.yaxis.axis_label = 'Wheel Base'



p.scatter("engine-size", "wheel-base", source=auto_df, legend_field="body-style", fill_alpha=0.4, size=12,

          marker=factor_mark('body-style', MARKERS, BODY_STYLE),

          color=factor_cmap('body-style', 'Category10_3', BODY_STYLE))



show(p)
#Index Filter 

from bokeh.models import  CDSView, IndexFilter



source = ColumnDataSource(auto_df)

view = CDSView(source=source, filters=[IndexFilter([70, 90, 110,130])])



tools = ["box_select", "hover", "reset"]

p = figure(plot_height=300, plot_width=300, tools=tools)

p.circle(x="engine-size", y="wheel-base", size=10, hover_color="red", source=source)



p_filtered = figure(plot_height=300, plot_width=300, tools=tools)

p_filtered.circle(x="engine-size", y="wheel-base", size=10, hover_color="red", source=source, view=view)



show(gridplot([[p, p_filtered]]))
# Boolean Filter 



from bokeh.models import BooleanFilter



booleans = [True if y_val > 110 else False for y_val in source.data['wheel-base']]

view = CDSView(source=source, filters=[BooleanFilter(booleans)])



tools = ["box_select", "hover", "reset"]

p = figure(plot_height=300, plot_width=300, tools=tools)

p.circle(x="engine-size", y="wheel-base", size=10, hover_color="red", source=source)



p_filtered = figure(plot_height=300, plot_width=300, tools=tools,

                    x_range=p.x_range, y_range=p.y_range)

p_filtered.circle(x="engine-size", y="wheel-base", size=10, hover_color="red", source=source, view=view)



show(gridplot([[p, p_filtered]]))

#GroupFilter



from bokeh.models import GroupFilter



view1 = CDSView(source=source, filters=[GroupFilter(column_name='body-style', group='hatchback')])

plot_size_and_tools = {'plot_height': 300, 'plot_width': 300,

                        'tools':['box_select', 'reset', 'help']}



p1 = figure(title="Full data set", **plot_size_and_tools)

p1.circle(x='engine-size', y='wheel-base', source=source, color='black')



p2 = figure(title="Sedan and Others only", x_range=p1.x_range, y_range=p1.y_range, **plot_size_and_tools)

p2.circle(x='engine-size', y='wheel-base', source=source, view=view1, color='red')



show(gridplot([[p1, p2]]))

#Column Layout 





# create three plots

s1 = figure(plot_width=500, plot_height=500, background_fill_color="#fafafa")

s1.circle(auto_df['engine-size'], auto_df['wheel-base'], size=12, color="#53777a", alpha=0.8)



s2 = figure(plot_width=500, plot_height=500, background_fill_color="#fafafa")

s2.triangle(auto_df['engine-size'], auto_df['length'], size=12, color="#c02942", alpha=0.8)



s3 = figure(plot_width=500, plot_height=500, background_fill_color="#fafafa")

s3.square(auto_df['engine-size'],auto_df['width'] , size=12, color="#d95b43", alpha=0.8)



# put the results in a column and show

show(column(s1, s2, s3))



#Row Layout



# create three plots

s1 = figure(plot_width=500, plot_height=500, background_fill_color="#fafafa")

s1.circle(auto_df['engine-size'], auto_df['wheel-base'], size=12, color="#53777a", alpha=0.8)



s2 = figure(plot_width=500, plot_height=500, background_fill_color="#fafafa")

s2.triangle(auto_df['engine-size'], auto_df['length'], size=12, color="#c02942", alpha=0.8)



s3 = figure(plot_width=500, plot_height=500, background_fill_color="#fafafa")

s3.square(auto_df['engine-size'],auto_df['width'] , size=12, color="#d95b43", alpha=0.8)



# put the results in a column and show

show(row(s1, s2, s3))
# create three plots

s1 = figure(plot_width=500, plot_height=500, background_fill_color="#fafafa")

s1.circle(auto_df['engine-size'], auto_df['wheel-base'], size=12, color="#53777a", alpha=0.8)



s2 = figure(plot_width=500, plot_height=500, background_fill_color="#fafafa")

s2.triangle(auto_df['engine-size'], auto_df['length'], size=12, color="#c02942", alpha=0.8)



s3 = figure(plot_width=500, plot_height=500, background_fill_color="#fafafa")

s3.square(auto_df['engine-size'],auto_df['width'] , size=12, color="#d95b43", alpha=0.8)



# make a grid

grid = gridplot([[s1, s2], [None, s3]], plot_width=250, plot_height=250)



show(grid)
from bokeh.palettes import Spectral6



body_style = ['sedan', 'hatchback', 'wagon', 'convertible']

counts = [(auto_df['body-style'].values == 'sedan').sum(),(auto_df['body-style'].values == 'hatchback').sum(),(auto_df['body-style'].values == 'wagon').sum(),(auto_df['body-style'].values == 'convertible').sum()]

source = ColumnDataSource(data=dict(body_style=body_style, counts=counts, color=Spectral6))



p = figure(x_range=body_style, y_range=(0,200), plot_height=250, title="Body Style of car Counts",

           toolbar_location=None, tools="")



p.vbar(x='body_style', top='counts', width=0.9, color='color', legend_field="body_style", source=source)



p.xgrid.grid_line_color = None

p.legend.orientation = "horizontal"

p.legend.location = "top_center"



show(p)
from bokeh.palettes import Spectral5



auto_df['body-style']= auto_df['body-style'].astype(str)

group = auto_df.groupby('body-style')



source = ColumnDataSource(group)



cyl_cmap = factor_cmap('body-style', palette=Spectral5, factors=sorted(auto_df['body-style'].unique()))



p = figure(plot_height=350, x_range=group, title="Engine size by # Cylinders",

           toolbar_location=None, tools="")



p.vbar(x='body-style', top='engine-size', width=1, source=auto_df,

       line_color=cyl_cmap, fill_color=cyl_cmap)



p.y_range.start = 0

p.xgrid.grid_line_color = None

p.xaxis.axis_label = "some stuff"

p.xaxis.major_label_orientation = 1.2

p.outline_line_color = None



show(p)
from bokeh.sampledata.sprint import sprint



output_file("sprint.html")



sprint.Year = sprint.Year.astype(str)

group = sprint.groupby('Year')

source = ColumnDataSource(group)



p = figure(y_range=group, x_range=(9.5,12.7), plot_width=400, plot_height=550, toolbar_location=None,

           title="Time Spreads for Sprint Medalists (by Year)")

p.hbar(y="Year", left='Time_min', right='Time_max', height=0.4, source=source)



p.ygrid.grid_line_color = None

p.xaxis.axis_label = "Time (seconds)"

p.outline_line_color = None



show(p)
from bokeh.transform import jitter

from bokeh.sampledata.commits import data



output_file("bars.html")



DAYS = ['Sun', 'Sat', 'Fri', 'Thu', 'Wed', 'Tue', 'Mon']



source = ColumnDataSource(data)



p = figure(plot_width=800, plot_height=300, y_range=DAYS, x_axis_type='datetime',

           title="Commits by Time of Day (US/Central) 2012—2016")



p.circle(x='time', y=jitter('day', width=0.6, range=p.y_range),  source=source, alpha=0.3)



p.xaxis[0].formatter.days = ['%Hh']

p.x_range.range_padding = 0

p.ygrid.grid_line_color = None



show(p)