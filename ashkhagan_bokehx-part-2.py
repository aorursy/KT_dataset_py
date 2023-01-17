import numpy as np  #for linear processing   
import pandas as pd #for handling tabular data
from bokeh.plotting import figure #to import the interactive figure
from bokeh.io import output_notebook,show #to plot and show results inline in the  notebook
output_notebook() #similiar to %matplotlib
from numpy import cos,linspace,log,tan,sin
x=linspace(-6,6,100)
y=tan(x)
p = figure(width=500, height=500) 
p.circle(x,y, size=3, color="black", alpha=0.6)
show(p)
data_frame=pd.read_csv("../input/titanic/train.csv")
p = figure(width=2000, height=500)
p.circle(data_frame.PassengerId,data_frame.Age, size=10, color="Blue", alpha=0.6)
show(p)
from bokeh.sampledata.autompg import autompg

grouped = autompg.groupby("yr")

mpg = grouped.mpg
avg, std = mpg.mean(), mpg.std()
years = list(grouped.groups)
american = autompg[autompg["origin"]==1]
japanese = autompg[autompg["origin"]==3]
p = figure(title="MPG by Year (Japan and US)")

p.vbar(x=years, bottom=avg-std, top=avg+std, width=0.8, 
       fill_alpha=0.2, line_color=None, legend_label="MPG 1 stddev")

p.circle(x=japanese["yr"], y=japanese["mpg"], size=10, alpha=0.5,
         color="red", legend_label="Japanese")

p.triangle(x=american["yr"], y=american["mpg"], size=10, alpha=0.3,
           color="blue", legend_label="American")

p.legend.location = "top_left"
show(p)
from bokeh.models import ColumnDataSource
from bokeh.layouts import gridplot

source = ColumnDataSource(autompg)

options = dict(plot_width=300, plot_height=300,
               tools="pan,wheel_zoom,box_zoom,box_select,lasso_select")

p1 = figure(title="MPG by Year", **options)
p1.circle("yr", "mpg", color="blue", source=source)

p2 = figure(title="HP vs. Displacement", **options)
p2.circle("hp", "displ", color="green", source=source)

p3 = figure(title="MPG vs. Displacement", **options)
p3.circle("mpg", "displ", size="cyl", line_color="red", fill_color=None, source=source)

p = gridplot([[ p1, p2, p3]], toolbar_location="right")

show(p)


from IPython.display import IFrame
IFrame('https://demo.bokeh.org/sliders', width=900, height=410)


from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput
from bokeh.plotting import figure

# Set up data
N = 200
x = np.linspace(0, 4*np.pi, N)
y = np.sin(x)
source = ColumnDataSource(data=dict(x=x, y=y))


# Set up plot
plot = figure(plot_height=400, plot_width=400, title="Coronavirus measure",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[0, 4*np.pi], y_range=[-2.5, 2.5])

plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)


# Set up widgets
text = TextInput(title="title", value='my sine wave')
offset = Slider(title="offset", value=0.0, start=-5.0, end=5.0, step=0.1)
amplitude = Slider(title="amplitude", value=1.0, start=-5.0, end=5.0, step=0.1)
phase = Slider(title="phase", value=0.0, start=0.0, end=2*np.pi)
freq = Slider(title="frequency", value=1.0, start=0.1, end=5.1, step=0.1)


# Set up callbacks
def update_title(attrname, old, new):
    plot.title.text = text.value

text.on_change('value', update_title)

def update_data(attrname, old, new):

    # Get the current slider values
    a = amplitude.value
    b = offset.value
    w = phase.value
    k = freq.value

    # Generate the new curve
    x = np.linspace(0, 4*np.pi, N)
    y = a*np.sin(k*x + w) + b

    source.data = dict(x=x, y=y)

for w in [offset, amplitude, phase, freq]:
    w.on_change('value', update_data)


# Set up layouts and add to document
inputs = column(text, offset, amplitude, phase, freq)

curdoc().add_root(row(inputs, plot, width=800))
curdoc().title = "Sliders"
