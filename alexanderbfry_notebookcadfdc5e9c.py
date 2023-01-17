import pandas as pd

import numpy as np

from bokeh.models import (

    ColumnDataSource,

    HoverTool,

    LogColorMapper

)

from bokeh.palettes import Viridis6 as palette

from bokeh.io import show

from bokeh.plotting import *

#  the bokeh sample data for state and county mapping may need to be downloaded

#  from bokeh.sampledata import download

#  download()

from bokeh.sampledata.us_states import data as states
pop = pd.read_csv("../input/CO-EST2015-alldata.csv", header=0, encoding="ISO-8859-1")                  

pop = pop[['STNAME', 'CTYNAME', 'POPESTIMATE2010']]
states = {code: state for code, state in states.items()}

state_xs = [state["lons"] for state in states.values()]

state_ys = [state["lats"] for state in states.values()]

state_names = np.asarray([state['name'] for state in states.values()])



fields = []



nogo = ["Hawaii", "Alaska", "District of Columbia"]



for i in pop.index:

    itemp = np.where(state_names == pop.loc[i, 'CTYNAME'] )[0]

    if len(itemp)==1:

        if (pop.loc[i, 'CTYNAME'] not in nogo):

            x = np.asarray(state_xs[itemp])

            y = np.asarray(state_ys[itemp])

            #if np.isnan(x).any():

            x = x[np.logical_not(np.isnan(x))]

            #if np.isnan(y).any():

            y = y[np.logical_not(np.isnan(y))] 

            mpop = np.round(pop.loc[i, 'POPESTIMATE2010']/1e6, 2)

            fields.append([state_names[itemp], x, y, mpop])



fields = np.asarray(fields)



state_source = ColumnDataSource(data=dict(

    x=fields[:, 1],

    y=fields[:, 2],

    name=fields[:, 0],

    mpop=fields[:, 3],

))
palette.reverse()

color_mapper = LogColorMapper(palette=palette)



TOOLS = "pan,wheel_zoom,box_zoom,reset,hover,save"

p = figure(

    title="Population, 2010", tools=TOOLS,

    x_axis_location=None, y_axis_location=None, plot_width=800, plot_height=500)



p.grid.grid_line_color = None



p.patches('x', 'y', source=state_source,

          fill_color={'field': 'mpop', 'transform': color_mapper},

          fill_alpha=0.7, line_color="white", line_width=0.5)



hover = p.select_one(HoverTool)

hover.point_policy = "follow_mouse"

hover.tooltips = [

    ("Name", "@name"),

    ("Population)", "@mpop million"),

    ("(Long, Lat)", "($x, $y)"),

]
from bokeh.charts import output_file, show, output_notebook

output_notebook()
show(p)