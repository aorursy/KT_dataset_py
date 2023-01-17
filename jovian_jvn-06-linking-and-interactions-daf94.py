from bokeh.io import output_notebook, show

from bokeh.plotting import figure

output_notebook()
from bokeh.layouts import gridplot



x = list(range(11))

y0, y1, y2 = x, [10-i for i in x], [abs(i-5) for i in x]



plot_options = dict(width=250, plot_height=250, tools='pan,wheel_zoom')



# create a new plot

s1 = figure(**plot_options)

s1.circle(x, y0, size=10, color="navy")



# create a new plot and share both ranges

s2 = figure(x_range=s1.x_range, y_range=s1.y_range, **plot_options)

s2.triangle(x, y1, size=10, color="firebrick")



# create a new plot and share only one range

s3 = figure(x_range=s1.x_range, **plot_options)

s3.square(x, y2, size=10, color="olive")



p = gridplot([[s1, s2, s3]])



# show the results

show(p)
# EXERCISE: create two plots in a gridplot, and link their ranges

from bokeh.models import ColumnDataSource



x = list(range(-20, 21))

y0, y1 = [abs(xx) for xx in x], [xx**2 for xx in x]



# create a column data source for the plots to share

source = ColumnDataSource(data=dict(x=x, y0=y0, y1=y1))



TOOLS = "box_select,lasso_select,help"



# create a new plot and add a renderer

left = figure(tools=TOOLS, width=300, height=300)

left.circle('x', 'y0', source=source)



# create another new plot and add a renderer

right = figure(tools=TOOLS, width=300, height=300)

right.circle('x', 'y1', source=source)



p = gridplot([[left, right]])



show(p)
# EXERCISE: create two plots in a gridplot, and link their data sources



from bokeh.models import HoverTool



source = ColumnDataSource(

        data=dict(

            x=[1, 2, 3, 4, 5],

            y=[2, 5, 8, 2, 7],

            desc=['A', 'b', 'C', 'd', 'E'],

        )

    )



hover = HoverTool(

        tooltips=[

            ("index", "$index"),

            ("(x,y)", "($x, $y)"),

            ("desc", "@desc"),

        ]

    )



p = figure(plot_width=300, plot_height=300, tools=[hover], title="Mouse over the dots")



p.circle('x', 'y', size=20, source=source)



show(p)
from bokeh.models.widgets import Slider





slider = Slider(start=0, end=10, value=1, step=.1, title="foo")



show(slider)
# EXERCISE: create and show a Select widget 

from bokeh.models import TapTool, CustomJS, ColumnDataSource



callback = CustomJS(code="alert('you tapped a circle!')")

tap = TapTool(callback=callback)



p = figure(plot_width=600, plot_height=300, tools=[tap])



p.circle(x=[1, 2, 3, 4, 5], y=[2, 5, 8, 2, 7], size=20)



show(p)
from bokeh.layouts import column

from bokeh.models import CustomJS, ColumnDataSource, Slider



x = [x*0.005 for x in range(0, 201)]



source = ColumnDataSource(data=dict(x=x, y=x))



plot = figure(plot_width=400, plot_height=400)

plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)



slider = Slider(start=0.1, end=6, value=1, step=.1, title="power")



update_curve = CustomJS(args=dict(source=source, slider=slider), code="""

    var data = source.data;

    var f = slider.value;

    var x = data['x']

    var y = data['y']

    for (var i = 0; i < x.length; i++) {

        y[i] = Math.pow(x[i], f)

    }

    

    // necessary becasue we mutated source.data in-place

    source.change.emit();

""")

slider.js_on_change('value', update_curve)





show(column(slider, plot))
# Exercise: Create a plot that updates based on a Select widget



from random import random



x = [random() for x in range(500)]

y = [random() for y in range(500)]

color = ["navy"] * len(x)



s1 = ColumnDataSource(data=dict(x=x, y=y, color=color))

p = figure(plot_width=400, plot_height=400, tools="lasso_select", title="Select Here")

p.circle('x', 'y', color='color', size=8, alpha=0.4, source=s1, 

         selection_color="firebrick", selection_alpha=0.4)



s2 = ColumnDataSource(data=dict(xm=[0,1],ym=[0.5, 0.5]))

p.line(x='xm', y='ym', color="orange", line_width=5, alpha=0.6, source=s2)



callback = CustomJS(args=dict(s1=s1, s2=s2), code="""

    var inds = s1.selected.indices;

    if (inds.length == 0)

        return;



    var ym = 0

    for (var i = 0; i < inds.length; i++) {

        ym += s1.data.y[inds[i]]

    }

    

    ym /= inds.length

    s2.data.ym = [ym, ym]



    // necessary becasue we mutated source.data in-place

    s2.change.emit();  

""")



s1.selected.js_on_change('indices', callback)



show(p)
# Exercise: Experiment with selection callbacks



from bokeh.plotting import figure

from bokeh import events

from bokeh.models import CustomJS, Div, Button

from bokeh.layouts import column, row



import numpy as np

x = np.random.random(size=2000) * 100

y = np.random.random(size=2000) * 100



p = figure(tools="box_select")

p.scatter(x, y, radius=1, fill_alpha=0.6, line_color=None)



div = Div(width=400)

button = Button(label="Button", width=300)

layout = column(button, row(p, div))



# Events with no attributes

button.js_on_event(events.ButtonClick,  CustomJS(args=dict(div=div), code="""

div.text = "Button!";

""")) 



p.js_on_event(events.SelectionGeometry, CustomJS(args=dict(div=div), code="""

div.text = "Selection! <p> <p>" + JSON.stringify(cb_obj.geometry, undefined, 2);

"""))



show(layout)
# Exercise: Create a plot that responds to different events from bokeh.events


