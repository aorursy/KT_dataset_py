from bokeh.plotting import figure, output_notebook, show

output_notebook()
x = [1, 3, 5, 7]

y = [2, 4, 6, 8]

z = [3 , 7, 1, 9]
p = figure()



p.circle(x, y, size=10, color='pink', legend='Circle')

p.line(z, y, color='orange', legend='Line')

p.square(y, x, color='navy', size=10, legend='Square')



p.legend.location = 'top_left'
show(p)
x = [1, 2, 3, 4, 5]

y =[6, 7, 2, 4, 5]



p = figure(plot_width=400, plot_height=400)



p.circle(x, y, size=15, line_color="navy", fill_color="orange", fill_alpha=0.5)



show(p)
size = [10, 15, 20, 25, 30]



p = figure(plot_width=400, plot_height=400)



p.square(x, y, size=size, color="firebrick", alpha=0.2)



show(p)
p = figure(plot_width=400, plot_height=400, title="My Line Plot")



p.line(x, y, line_width=2)

p.circle(x, y, fill_color="white", size=8)



show(p) 
categories = ['A', 'B', 'C', 'D', 'E', 'F']

counts = [2, 4, 6, 8, 14, 7]



# Setting the x_range to the list of categories above

p = figure(x_range=categories, plot_height=250, title="Basic Bar Chart")



# Calling the vbar method and setting the y_range values

p.vbar(x=categories, top=counts, width=0.9)



# Additional Properties

p.xgrid.grid_line_color = None

p.y_range.start = 0



show(p)
from bokeh.models import ColumnDataSource
from bokeh.models import FactorRange



subjects = ['Math', 'History', 'Philosophy', 'Physics', 'Chemistry', 'Economics']

names = ['Jane', 'Kevin', 'David']



data = {'subjectss' : subjects,

        'Jane'   : [20, 10, 40, 30, 90, 40],

        'Kevin'   : [50, 30, 30, 20, 40, 60],

        'David'   : [30, 70, 40, 80, 50, 30]}



x = [ (subject, name) for subject in subjects for name in names ]

counts = sum(zip(data['Jane'], data['Kevin'], data['David']), ()) # like an hstack



source = ColumnDataSource(data=dict(x=x, counts=counts))



p = figure(x_range=FactorRange(*x), plot_height=250, title="Student Score by Subject")



p.vbar(x='x', top='counts', width=0.9, source=source)



p.y_range.start = 0

p.x_range.range_padding = 0.1

p.xaxis.major_label_orientation = 1

p.xgrid.grid_line_color = None



show(p)
from bokeh.models.annotations import Label



p = figure(plot_width=400, plot_height=400)



# keep a reference to the returned GlyphRenderer

r = p.circle([1,2,3,4,5], [2,5,8,2,7])



r.glyph.size = 50

r.glyph.fill_alpha = 0.2

r.glyph.line_color = "firebrick"

r.glyph.line_dash = [5, 1]

r.glyph.line_width = 2

p.outline_line_width = 7

p.outline_line_alpha = 0.3

p.outline_line_color = "navy"





label = Label(x=2, y=5, x_offset=20, y_offset=25, text="Second Point", text_baseline="middle")

p.add_layout(label)





show(p)
import pandas as pd

import numpy as np

import math

from math import pi
df = pd.read_csv('../input/bokehintro/Pokemon.csv')

df.head()
from bokeh.models import ColumnDataSource

from bokeh.models.tools import HoverTool
from bokeh.transform import factor_cmap

from bokeh.palettes import Spectral5, Spectral3, inferno, viridis, Category20
source = ColumnDataSource(df)

types = df['Type 1'].unique().tolist()

color_map = factor_cmap(field_name='Type 1', palette=viridis(18), factors=types)
p = figure()



p.circle(x='Attack', y='Speed', source=source, size=10, color=color_map)



p.title.text = 'Pokemon Attack vs Speed'

p.xaxis.axis_label = 'Attacking Stats'

p.yaxis.axis_label = 'Speed Stats'



hover = HoverTool()

hover.tooltips=[

    ('Attack', '@Attack'),

    ('Speed', '@Speed'),

    ('Type 1', '@{Type 1}'),

]



p.add_tools(hover)



show(p)
attribs = df.groupby('Type 1')['Attack', 'Defense', 'Speed', 'Sp. Atk', 'Sp. Def'].mean()
attribs
source = ColumnDataSource(attribs)

types = source.data['Type 1'].tolist()

p = figure(x_range=types)



color_map = factor_cmap(field_name='Type 1', palette=inferno(18), factors=types)



p.vbar(x='Type 1', top='Defense', source=source, width=0.70, color=color_map)



p.title.text ='Defense By Type'

p.xaxis.axis_label = 'Type 1'

p.yaxis.axis_label = 'Average Defense Score'



p.xaxis.major_label_orientation = math.pi/2.7



show(p)
q = figure(x_range=types)

q.vbar_stack(stackers=['Attack', 'Defense', 'Speed', 'Sp. Atk', 'Sp. Def'], 

             x='Type 1', source=source, 

             legend = ['Atk', 'Def', 'Spd', 'Sp.Atk', 'Sp.Def'],

             width=0.5, color=Spectral5)



q.title.text ='Average Score for each Stat'

q.legend.location = 'top_left'



q.xaxis.axis_label = 'Type 1'

q.xgrid.grid_line_color = None	#remove the x grid lines



q.yaxis.axis_label = 'Score'



q.xaxis.major_label_orientation = math.pi/2.7



show(q)
from datetime import datetime
aapl = pd.read_csv('../input/stocks/AAPL.csv')

aapl = aapl.set_index('Date')

aapl = pd.DataFrame(aapl['Adj Close'])

aapl.index = pd.to_datetime(aapl.index, format='%Y-%m-%d')



goog = pd.read_csv('../input/stocks/GOOG.csv')

goog = goog.set_index('Date')

goog = pd.DataFrame(goog['Adj Close'])

goog.index = pd.to_datetime(goog.index, format='%Y-%m-%d')



ibm = pd.read_csv('../input/stocks/IBM.csv')

ibm = ibm.set_index('Date')

ibm = pd.DataFrame(ibm['Adj Close'])

ibm.index = pd.to_datetime(ibm.index, format='%Y-%m-%d')
source = ColumnDataSource(aapl)

_source = ColumnDataSource(goog)

source_ = ColumnDataSource(ibm)



p = figure(x_axis_type='datetime')



p1 = p.line(x='Date', y='Adj Close', line_width=2, source=source, legend='AAPL Stock Price')

p.add_tools(HoverTool(renderers=[p1], tooltips=[('Apple',"@{Adj Close}")],mode='vline'))

p2 = p.line(x='Date', y='Adj Close', line_width=2, source=_source, color=Spectral3[2], legend='GOOG Stock Price')

p.add_tools(HoverTool(renderers=[p2], tooltips=[('Google',"@{Adj Close}")],mode='vline'))

p3 = p.line(x='Date', y='Adj Close', line_width=2, source=source_, color='#33A02C', legend='IBM Stock Price')

p.add_tools(HoverTool(renderers=[p3], tooltips=[('IBM',"@{Adj Close}")],mode='vline'))





p.legend.location = 'top_left'





p.yaxis.axis_label = 'Adjusted Closing Price'



show(p)
from bokeh.models import BoxAnnotation



box_left = pd.to_datetime('2008-01-01')

box_right = pd.to_datetime('2009-01-01')



p.legend.click_policy='hide'



box = BoxAnnotation(left=box_left, right=box_right,

                    line_width=1, line_color='black', line_dash='dashed',

                    fill_alpha=0.2, fill_color='orange')



p.add_layout(box)



show(p)