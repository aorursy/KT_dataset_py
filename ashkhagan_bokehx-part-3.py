from bokeh.io import output_notebook
output_notebook()
from bokeh.layouts import column, gridplot
from bokeh.models import Div, Range1d, WMTSTileSource
from bokeh.plotting import figure, show

output_notebook()
import bokeh.sampledata
bokeh.sampledata.download()
from bokeh.sampledata.airports import data as airports
from bokeh.tile_providers import CARTODBPOSITRON, get_provider

title = "US Airports: Field Elevation > 1500m"

def plot(tile_source):

    # set to roughly extent of points
    x_range = Range1d(start=airports['x'].min() - 10000, end=airports['x'].max() + 10000, bounds=None)
    y_range = Range1d(start=airports['y'].min() - 10000, end=airports['y'].max() + 10000, bounds=None)

    # create plot and add tools
    p = figure(tools='hover,wheel_zoom,pan,reset', x_range=x_range, y_range=y_range, title=title,
               tooltips=[("Name", "@name"), ("Elevation", "@elevation (m)")],
               plot_width=400, plot_height=400)
    p.axis.visible = False
    p.add_tile(tile_source)

    # create point glyphs
    p.circle(x='x', y='y', size=10, fill_color="#F46B42", line_color="white", line_width=2, source=airports)
    return p

# create a tile source
tile_options = {}
tile_options['url'] = 'http://tile.stamen.com/terrain/{Z}/{X}/{Y}.png'
tile_options['attribution'] = """
    Map tiles by <a href="http://stamen.com">Stamen Design</a>, under
    <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>.
    Data by <a href="http://openstreetmap.org">OpenStreetMap</a>,
    under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.
    """
mq_tile_source = WMTSTileSource(**tile_options)

carto = plot(get_provider(CARTODBPOSITRON))
mq = plot(mq_tile_source)

# link panning
mq.x_range = carto.x_range
mq.y_range = carto.y_range

div = Div(text="""
<p>This example shows the same data on two separate tile plots. The left plot
is using a built-in CartoDB tile source, and is using  a customized tile source
configured for OpenStreetMap.</p>
""", width=800)

layout = column(div, gridplot([[carto, mq]], toolbar_location="right"))

show(layout)
from bokeh.layouts import layout
from bokeh.plotting import figure, output_file, show

p1 = figure(match_aspect=True, title="Circle touches all 4 sides of square")
p1.rect(0, 0, 300, 300, line_color='black')
p1.circle(x=0, y=0, radius=150, line_color='black', fill_color='grey',
          radius_units='data')

def draw_test_figure(aspect_scale=1, width=300, height=300):
    p = figure(
        plot_width=width,
        plot_height=height,
        match_aspect=True,
        aspect_scale=aspect_scale,
        title="Aspect scale = {0}".format(aspect_scale),
        toolbar_location=None)
    p.circle([-1, +1, +1, -1], [-1, -1, +1, +1])
    return p

aspect_scales = [0.25, 0.5, 1, 2, 4]
p2s = [draw_test_figure(aspect_scale=i) for i in aspect_scales]

sizes = [(100, 400), (200, 400), (400, 200), (400, 100)]
p3s = [draw_test_figure(width=a, height=b) for (a, b) in sizes]

layout = layout(children=[[p1], p2s, p3s])

output_file("aspect.html")
show(layout)
from bokeh.io import output_file, show
from bokeh.plotting import figure

#output_file("bar_basic.html")

fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
counts = [5, 3, 4, 2, 4, 6]

p = figure(x_range=fruits, plot_height=350, title="Fruit Counts",
           toolbar_location=None, tools="")

p.vbar(x=fruits, top=counts, width=0.9)

p.xgrid.grid_line_color = None
p.y_range.start = 0

show(p)
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral6
from bokeh.plotting import figure
from bokeh.transform import factor_cmap

output_file("bar_colormapped.html")

fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
counts = [5, 3, 4, 2, 4, 6]

source = ColumnDataSource(data=dict(fruits=fruits, counts=counts))

p = figure(x_range=fruits, plot_height=350, toolbar_location=None, title="Fruit Counts")
p.vbar(x='fruits', top='counts', width=0.9, source=source, legend_field="fruits",
       line_color='white', fill_color=factor_cmap('fruits', palette=Spectral6, factors=fruits))

p.xgrid.grid_line_color = None
p.y_range.start = 0
p.y_range.end = 9
p.legend.orientation = "horizontal"
p.legend.location = "top_center"

show(p)
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral6
from bokeh.plotting import figure

output_file("bar_colors.html")

fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
counts = [5, 3, 4, 2, 4, 6]

source = ColumnDataSource(data=dict(fruits=fruits, counts=counts, color=Spectral6))

p = figure(x_range=fruits, y_range=(0,9), plot_height=350, title="Fruit Counts",
           toolbar_location=None, tools="")

p.vbar(x='fruits', top='counts', width=0.9, color='color', legend_field="fruits", source=source)

p.xgrid.grid_line_color = None
p.legend.orientation = "horizontal"
p.legend.location = "top_center"

show(p)
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import dodge

output_file("bar_dodged.html")

fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
years = ['2015', '2016', '2017']

data = {'fruits' : fruits,
        '2015'   : [2, 1, 4, 3, 2, 4],
        '2016'   : [5, 3, 3, 2, 4, 6],
        '2017'   : [3, 2, 4, 4, 5, 3]}

source = ColumnDataSource(data=data)

p = figure(x_range=fruits, y_range=(0, 10), plot_height=350, title="Fruit Counts by Year",
           toolbar_location=None, tools="")

p.vbar(x=dodge('fruits', -0.25, range=p.x_range), top='2015', width=0.2, source=source,
       color="#c9d9d3", legend_label="2015")

p.vbar(x=dodge('fruits',  0.0,  range=p.x_range), top='2016', width=0.2, source=source,
       color="#718dbf", legend_label="2016")

p.vbar(x=dodge('fruits',  0.25, range=p.x_range), top='2017', width=0.2, source=source,
       color="#e84d60", legend_label="2017")

p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.legend.location = "top_left"
p.legend.orientation = "horizontal"

show(p)
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.sampledata.sprint import sprint

output_file("bar_intervals.html")

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
from bokeh.io import output_file, show
from bokeh.models import FactorRange
from bokeh.plotting import figure

output_file("bar_mixed.html")

factors = [
    ("Q1", "jan"), ("Q1", "feb"), ("Q1", "mar"),
    ("Q2", "apr"), ("Q2", "may"), ("Q2", "jun"),
    ("Q3", "jul"), ("Q3", "aug"), ("Q3", "sep"),
    ("Q4", "oct"), ("Q4", "nov"), ("Q4", "dec"),

]

p = figure(x_range=FactorRange(*factors), plot_height=350,
           toolbar_location=None, tools="")

x = [ 10, 12, 16, 9, 10, 8, 12, 13, 14, 14, 12, 16 ]
p.vbar(x=factors, top=x, width=0.9, alpha=0.5)

p.line(x=["Q1", "Q2", "Q3", "Q4"], y=[12, 9, 13, 14], color="red", line_width=2)

p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xaxis.major_label_orientation = 1
p.xgrid.grid_line_color = None

show(p)
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure

output_file("bar_nested.html")

fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
years = ['2015', '2016', '2017']

data = {'fruits' : fruits,
        '2015'   : [2, 1, 4, 3, 2, 4],
        '2016'   : [5, 3, 3, 2, 4, 6],
        '2017'   : [3, 2, 4, 4, 5, 3]}

# this creates [ ("Apples", "2015"), ("Apples", "2016"), ("Apples", "2017"), ("Pears", "2015), ... ]
x = [ (fruit, year) for fruit in fruits for year in years ]
counts = sum(zip(data['2015'], data['2016'], data['2017']), ()) # like an hstack

source = ColumnDataSource(data=dict(x=x, counts=counts))

p = figure(x_range=FactorRange(*x), plot_height=350, title="Fruit Counts by Year",
           toolbar_location=None, tools="")

p.vbar(x='x', top='counts', width=0.9, source=source)

p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xaxis.major_label_orientation = 1
p.xgrid.grid_line_color = None

show(p)
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure
from bokeh.transform import factor_cmap

output_file("bar_nested_colormapped.html")

fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
years = ['2015', '2016', '2017']

data = {'fruits' : fruits,
        '2015'   : [2, 1, 4, 3, 2, 4],
        '2016'   : [5, 3, 3, 2, 4, 6],
        '2017'   : [3, 2, 4, 4, 5, 3]}

palette = ["#c9d9d3", "#718dbf", "#e84d60"]

# this creates [ ("Apples", "2015"), ("Apples", "2016"), ("Apples", "2017"), ("Pears", "2015), ... ]
x = [ (fruit, year) for fruit in fruits for year in years ]
counts = sum(zip(data['2015'], data['2016'], data['2017']), ()) # like an hstack

source = ColumnDataSource(data=dict(x=x, counts=counts))

p = figure(x_range=FactorRange(*x), plot_height=350, title="Fruit Counts by Year",
           toolbar_location=None, tools="")

p.vbar(x='x', top='counts', width=0.9, source=source, line_color="white",
       fill_color=factor_cmap('x', palette=palette, factors=years, start=1, end=2))

p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xaxis.major_label_orientation = 1
p.xgrid.grid_line_color = None

show(p)
from bokeh.io import output_file, show
from bokeh.palettes import Spectral5
from bokeh.plotting import figure
from bokeh.sampledata.autompg import autompg_clean as df
from bokeh.transform import factor_cmap

output_file("bar_pandas_groupby_nested.html")

df.cyl = df.cyl.astype(str)
df.yr = df.yr.astype(str)

group = df.groupby(['cyl', 'mfr'])

index_cmap = factor_cmap('cyl_mfr', palette=Spectral5, factors=sorted(df.cyl.unique()), end=1)

p = figure(plot_width=800, plot_height=300, title="Mean MPG by # Cylinders and Manufacturer",
           x_range=group, toolbar_location=None, tooltips=[("MPG", "@mpg_mean"), ("Cyl, Mfr", "@cyl_mfr")])

p.vbar(x='cyl_mfr', top='mpg_mean', width=1, source=group,
       line_color="white", fill_color=index_cmap, )

p.y_range.start = 0
p.x_range.range_padding = 0.05
p.xgrid.grid_line_color = None
p.xaxis.axis_label = "Manufacturer grouped by # Cylinders"
p.xaxis.major_label_orientation = 1.2
p.outline_line_color = None

show(p)
from bokeh.io import output_file, show
from bokeh.plotting import figure

output_file("bar_sorted.html")

fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
counts = [5, 3, 4, 2, 4, 6]

# sorting the bars means sorting the range factors
sorted_fruits = sorted(fruits, key=lambda x: counts[fruits.index(x)])

p = figure(x_range=sorted_fruits, plot_height=350, title="Fruit Counts",
           toolbar_location=None, tools="")

p.vbar(x=fruits, top=counts, width=0.9)

p.xgrid.grid_line_color = None
p.y_range.start = 0

show(p)
from bokeh.io import output_file, show
from bokeh.plotting import figure

output_file("bar_stacked.html")

fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
years = ["2015", "2016", "2017"]
colors = ["#c9d9d3", "#718dbf", "#e84d60"]

data = {'fruits' : fruits,
        '2015'   : [2, 1, 4, 3, 2, 4],
        '2016'   : [5, 3, 4, 2, 4, 6],
        '2017'   : [3, 2, 4, 4, 5, 3]}

p = figure(x_range=fruits, plot_height=250, title="Fruit Counts by Year",
           toolbar_location=None, tools="hover", tooltips="$name @fruits: @$name")

p.vbar_stack(years, x='fruits', width=0.9, color=colors, source=data,
             legend_label=years)

p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.axis.minor_tick_line_color = None
p.outline_line_color = None
p.legend.location = "top_left"
p.legend.orientation = "horizontal"

show(p)
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure

output_file("bar_stacked_grouped.html")

factors = [
    ("Q1", "jan"), ("Q1", "feb"), ("Q1", "mar"),
    ("Q2", "apr"), ("Q2", "may"), ("Q2", "jun"),
    ("Q3", "jul"), ("Q3", "aug"), ("Q3", "sep"),
    ("Q4", "oct"), ("Q4", "nov"), ("Q4", "dec"),

]

regions = ['east', 'west']

source = ColumnDataSource(data=dict(
    x=factors,
    east=[ 5, 5, 6, 5, 5, 4, 5, 6, 7, 8, 6, 9 ],
    west=[ 5, 7, 9, 4, 5, 4, 7, 7, 7, 6, 6, 7 ],
))

p = figure(x_range=FactorRange(*factors), plot_height=250,
           toolbar_location=None, tools="")

p.vbar_stack(regions, x='x', width=0.9, alpha=0.5, color=["blue", "red"], source=source,
             legend_label=regions)

p.y_range.start = 0
p.y_range.end = 18
p.x_range.range_padding = 0.1
p.xaxis.major_label_orientation = 1
p.xgrid.grid_line_color = None
p.legend.location = "top_center"
p.legend.orientation = "horizontal"

show(p)

from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource
from bokeh.palettes import GnBu3, OrRd3
from bokeh.plotting import figure

output_file("bar_stacked_split.html")

fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
years = ["2015", "2016", "2017"]

exports = {'fruits' : fruits,
           '2015'   : [2, 1, 4, 3, 2, 4],
           '2016'   : [5, 3, 4, 2, 4, 6],
           '2017'   : [3, 2, 4, 4, 5, 3]}
imports = {'fruits' : fruits,
           '2015'   : [-1, 0, -1, -3, -2, -1],
           '2016'   : [-2, -1, -3, -1, -2, -2],
           '2017'   : [-1, -2, -1, 0, -2, -2]}

p = figure(y_range=fruits, plot_height=350, x_range=(-16, 16), title="Fruit import/export, by year",
           toolbar_location=None)

p.hbar_stack(years, y='fruits', height=0.9, color=GnBu3, source=ColumnDataSource(exports),
             legend_label=["%s exports" % x for x in years])

p.hbar_stack(years, y='fruits', height=0.9, color=OrRd3, source=ColumnDataSource(imports),
             legend_label=["%s imports" % x for x in years])

p.y_range.range_padding = 0.1
p.ygrid.grid_line_color = None
p.legend.location = "top_left"
p.axis.minor_tick_line_color = None
p.outline_line_color = None

show(p)
from bokeh.models import BoxAnnotation
from bokeh.plotting import figure, output_file, show
from bokeh.sampledata.glucose import data

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

data = data.loc['2010-10-04':'2010-10-04']

p = figure(x_axis_type="datetime", tools=TOOLS, title="Glocose Readings, Oct 4th (Red = Outside Range)")
p.background_fill_color = "#efefef"
p.xgrid.grid_line_color=None
p.xaxis.axis_label = 'Time'
p.yaxis.axis_label = 'Value'

p.line(data.index, data.glucose, line_color='grey')
p.circle(data.index, data.glucose, color='grey', size=1)

p.add_layout(BoxAnnotation(top=80, fill_alpha=0.1, fill_color='red', line_color='red'))
p.add_layout(BoxAnnotation(bottom=180, fill_alpha=0.1, fill_color='red', line_color='red'))

output_file("box_annotation.html", title="box_annotation.py example")

show(p)
import numpy as np
import pandas as pd

from bokeh.plotting import figure, output_file, show

# generate some synthetic time series for six different categories
cats = list("abcdef")
yy = np.random.randn(2000)
g = np.random.choice(cats, 2000)
for i, l in enumerate(cats):
    yy[g == l] += i // 2
df = pd.DataFrame(dict(score=yy, group=g))

# find the quartiles and IQR for each category
groups = df.groupby('group')
q1 = groups.quantile(q=0.25)
q2 = groups.quantile(q=0.5)
q3 = groups.quantile(q=0.75)
iqr = q3 - q1
upper = q3 + 1.5*iqr
lower = q1 - 1.5*iqr

# find the outliers for each category
def outliers(group):
    cat = group.name
    return group[(group.score > upper.loc[cat]['score']) | (group.score < lower.loc[cat]['score'])]['score']
out = groups.apply(outliers).dropna()

# prepare outlier data for plotting, we need coordinates for every outlier.
if not out.empty:
    outx = []
    outy = []
    for keys in out.index:
        outx.append(keys[0])
        outy.append(out.loc[keys[0]].loc[keys[1]])

p = figure(tools="", background_fill_color="#efefef", x_range=cats, toolbar_location=None)

# if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
qmin = groups.quantile(q=0.00)
qmax = groups.quantile(q=1.00)
upper.score = [min([x,y]) for (x,y) in zip(list(qmax.loc[:,'score']),upper.score)]
lower.score = [max([x,y]) for (x,y) in zip(list(qmin.loc[:,'score']),lower.score)]

# stems
p.segment(cats, upper.score, cats, q3.score, line_color="black")
p.segment(cats, lower.score, cats, q1.score, line_color="black")

# boxes
p.vbar(cats, 0.7, q2.score, q3.score, fill_color="#E08E79", line_color="black")
p.vbar(cats, 0.7, q1.score, q2.score, fill_color="#3B8686", line_color="black")

# whiskers (almost-0 height rects simpler than segments)
p.rect(cats, lower.score, 0.2, 0.01, line_color="black")
p.rect(cats, upper.score, 0.2, 0.01, line_color="black")

# outliers
if not out.empty:
    p.circle(outx, outy, size=6, color="#F38630", fill_alpha=0.6)

p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = "white"
p.grid.grid_line_width = 2
p.xaxis.major_label_text_font_size="16px"

output_file("boxplot.html", title="boxplot.py example")

show(p)

import numpy as np
import pandas as pd

from bokeh.palettes import brewer
from bokeh.plotting import figure, output_file, show

N = 20
cats = 10
df = pd.DataFrame(np.random.randint(10, 100, size=(N, cats))).add_prefix('y')

def stacked(df):
    df_top = df.cumsum(axis=1)
    df_bottom = df_top.shift(axis=1).fillna({'y0': 0})[::-1]
    df_stack = pd.concat([df_bottom, df_top], ignore_index=True)
    return df_stack

areas = stacked(df)
colors = brewer['Spectral'][areas.shape[1]]
x2 = np.hstack((df.index[::-1], df.index))

p = figure(x_range=(0, N-1), y_range=(0, 800))
p.grid.minor_grid_line_color = '#eeeeee'

p.patches([x2] * areas.shape[1], [areas[c].values for c in areas],
          color=colors, alpha=0.8, line_color=None)

output_file('stacked_area.html', title='brewer.py example')

show(p)

from collections import OrderedDict
from io import StringIO
from math import log, sqrt

import numpy as np
import pandas as pd

from bokeh.plotting import figure, output_file, show

antibiotics = """
bacteria,                        penicillin, streptomycin, neomycin, gram
Mycobacterium tuberculosis,      800,        5,            2,        negative
Salmonella schottmuelleri,       10,         0.8,          0.09,     negative
Proteus vulgaris,                3,          0.1,          0.1,      negative
Klebsiella pneumoniae,           850,        1.2,          1,        negative
Brucella abortus,                1,          2,            0.02,     negative
Pseudomonas aeruginosa,          850,        2,            0.4,      negative
Escherichia coli,                100,        0.4,          0.1,      negative
Salmonella (Eberthella) typhosa, 1,          0.4,          0.008,    negative
Aerobacter aerogenes,            870,        1,            1.6,      negative
Brucella antracis,               0.001,      0.01,         0.007,    positive
Streptococcus fecalis,           1,          1,            0.1,      positive
Staphylococcus aureus,           0.03,       0.03,         0.001,    positive
Staphylococcus albus,            0.007,      0.1,          0.001,    positive
Streptococcus hemolyticus,       0.001,      14,           10,       positive
Streptococcus viridans,          0.005,      10,           40,       positive
Diplococcus pneumoniae,          0.005,      11,           10,       positive
"""

drug_color = OrderedDict([
    ("Penicillin",   "#0d3362"),
    ("Streptomycin", "#c64737"),
    ("Neomycin",     "black"  ),
])

gram_color = OrderedDict([
    ("negative", "#e69584"),
    ("positive", "#aeaeb8"),
])

df = pd.read_csv(StringIO(antibiotics),
                 skiprows=1,
                 skipinitialspace=True,
                 engine='python')

width = 800
height = 800
inner_radius = 90
outer_radius = 300 - 10

minr = sqrt(log(.001 * 1E4))
maxr = sqrt(log(1000 * 1E4))
a = (outer_radius - inner_radius) / (minr - maxr)
b = inner_radius - a * maxr

def rad(mic):
    return a * np.sqrt(np.log(mic * 1E4)) + b

big_angle = 2.0 * np.pi / (len(df) + 1)
small_angle = big_angle / 7

p = figure(plot_width=width, plot_height=height, title="",
    x_axis_type=None, y_axis_type=None,
    x_range=(-420, 420), y_range=(-420, 420),
    min_border=0, outline_line_color="black",
    background_fill_color="#f0e1d2")

p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None

# annular wedges
angles = np.pi/2 - big_angle/2 - df.index.to_series()*big_angle
colors = [gram_color[gram] for gram in df.gram]
p.annular_wedge(
    0, 0, inner_radius, outer_radius, -big_angle+angles, angles, color=colors,
)

# small wedges
p.annular_wedge(0, 0, inner_radius, rad(df.penicillin),
                -big_angle+angles+5*small_angle, -big_angle+angles+6*small_angle,
                color=drug_color['Penicillin'])
p.annular_wedge(0, 0, inner_radius, rad(df.streptomycin),
                -big_angle+angles+3*small_angle, -big_angle+angles+4*small_angle,
                color=drug_color['Streptomycin'])
p.annular_wedge(0, 0, inner_radius, rad(df.neomycin),
                -big_angle+angles+1*small_angle, -big_angle+angles+2*small_angle,
                color=drug_color['Neomycin'])

# circular axes and lables
labels = np.power(10.0, np.arange(-3, 4))
radii = a * np.sqrt(np.log(labels * 1E4)) + b
p.circle(0, 0, radius=radii, fill_color=None, line_color="white")
p.text(0, radii[:-1], [str(r) for r in labels[:-1]],
       text_font_size="11px", text_align="center", text_baseline="middle")

# radial axes
p.annular_wedge(0, 0, inner_radius-10, outer_radius+10,
                -big_angle+angles, -big_angle+angles, color="black")

# bacteria labels
xr = radii[0]*np.cos(np.array(-big_angle/2 + angles))
yr = radii[0]*np.sin(np.array(-big_angle/2 + angles))
label_angle=np.array(-big_angle/2+angles)
label_angle[label_angle < -np.pi/2] += np.pi # easier to read labels on the left side
p.text(xr, yr, df.bacteria, angle=label_angle,
       text_font_size="12px", text_align="center", text_baseline="middle")

# OK, these hand drawn legends are pretty clunky, will be improved in future release
p.circle([-40, -40], [-370, -390], color=list(gram_color.values()), radius=5)
p.text([-30, -30], [-370, -390], text=["Gram-" + gr for gr in gram_color.keys()],
       text_font_size="9px", text_align="left", text_baseline="middle")

p.rect([-40, -40, -40], [18, 0, -18], width=30, height=13,
       color=list(drug_color.values()))
p.text([-15, -15, -15], [18, 0, -18], text=list(drug_color),
       text_font_size="12px", text_align="left", text_baseline="middle")

output_file("burtin.html", title="burtin.py example")

show(p)
from math import pi

import pandas as pd

from bokeh.plotting import figure, output_file, show
from bokeh.sampledata.stocks import MSFT

df = pd.DataFrame(MSFT)[:50]
df["date"] = pd.to_datetime(df["date"])

inc = df.close > df.open
dec = df.open > df.close
w = 12*60*60*1000 # half day in ms

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=1000, title = "MSFT Candlestick")
p.xaxis.major_label_orientation = pi/4
p.grid.grid_line_alpha=0.3

p.segment(df.date, df.high, df.date, df.low, color="black")
p.vbar(df.date[inc], w, df.open[inc], df.close[inc], fill_color="#D5E1DD", line_color="black")
p.vbar(df.date[dec], w, df.open[dec], df.close[dec], fill_color="#F2583E", line_color="black")

output_file("candlestick.html", title="candlestick.py example")

show(p)  # open a browser

from bokeh.layouts import row
from bokeh.plotting import figure, output_file, show

factors = ["a", "b", "c", "d", "e", "f", "g", "h"]
x =  [50, 40, 65, 10, 25, 37, 80, 60]

dot = figure(title="Categorical Dot Plot", tools="", toolbar_location=None,
            y_range=factors, x_range=[0,100])

dot.segment(0, factors, x, factors, line_width=2, line_color="green", )
dot.circle(x, factors, size=15, fill_color="orange", line_color="green", line_width=3, )

factors = ["foo 123", "bar:0.2", "baz-10"]
x = ["foo 123", "foo 123", "foo 123", "bar:0.2", "bar:0.2", "bar:0.2", "baz-10",  "baz-10",  "baz-10"]
y = ["foo 123", "bar:0.2", "baz-10",  "foo 123", "bar:0.2", "baz-10",  "foo 123", "bar:0.2", "baz-10"]
colors = [
    "#0B486B", "#79BD9A", "#CFF09E",
    "#79BD9A", "#0B486B", "#79BD9A",
    "#CFF09E", "#79BD9A", "#0B486B"
]

hm = figure(title="Categorical Heatmap", tools="hover", toolbar_location=None,
            x_range=factors, y_range=factors)

hm.rect(x, y, color=colors, width=1, height=1)

output_file("categorical.html", title="categorical.py example")

show(row(hm, dot, sizing_mode="scale_width"))  # open a browser
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.sampledata.commits import data
from bokeh.transform import jitter

output_file("categorical_scatter_jitter.html")

DAYS = ['Sun', 'Sat', 'Fri', 'Thu', 'Wed', 'Tue', 'Mon']

source = ColumnDataSource(data)

p = figure(plot_width=800, plot_height=300, y_range=DAYS, x_axis_type='datetime',
           title="Commits by Time of Day (US/Central) 2012-2016")

p.circle(x='time', y=jitter('day', width=0.6, range=p.y_range),  source=source, alpha=0.3)

p.xaxis.formatter.days = ['%Hh']
p.x_range.range_padding = 0
p.ygrid.grid_line_color = None

show(p)
from bokeh.palettes import Viridis6
from bokeh.plotting import figure, show
from bokeh.sampledata.unemployment import data as unemployment
from bokeh.sampledata.us_counties import data as counties
from bokeh.sampledata.us_states import data as states

del states["HI"]
del states["AK"]

EXCLUDED = ("ak", "hi", "pr", "gu", "vi", "mp", "as")

state_xs = [states[code]["lons"] for code in states]
state_ys = [states[code]["lats"] for code in states]

county_xs=[counties[code]["lons"] for code in counties if counties[code]["state"] not in EXCLUDED]
county_ys=[counties[code]["lats"] for code in counties if counties[code]["state"] not in EXCLUDED]

county_colors = []
for county_id in counties:
    if counties[county_id]["state"] in EXCLUDED:
        continue
    try:
        rate = unemployment[county_id]
        idx = int(rate/6)
        county_colors.append(Viridis6[idx])
    except KeyError:
        county_colors.append("black")

p = figure(title="US Unemployment 2009",
           x_axis_location=None, y_axis_location=None,
           plot_width=1000, plot_height=600)
p.grid.grid_line_color = None

p.patches(county_xs, county_ys,
          fill_color=county_colors, fill_alpha=0.7,
          line_color="white", line_width=0.5)

p.patches(state_xs, state_ys, fill_alpha=0.0,
          line_color="#884444", line_width=2, line_alpha=0.3)

show(p)  # Change to save(p) to save but not show the HTML file
import numpy as np

from bokeh.io import show
from bokeh.layouts import column, gridplot
from bokeh.models import ColorBar, ColumnDataSource, LinearColorMapper, LogColorMapper
from bokeh.plotting import figure
from bokeh.transform import transform

x = np.random.random(size=2000) * 1000
y = np.random.normal(size=2000) * 2 + 5
source = ColumnDataSource(dict(x=x, y=y))

def make_plot(mapper_type, palette):
    mapper_opts = dict(palette=palette, low=1, high=1000)
    if mapper_type == "linear":
        mapper = LinearColorMapper(**mapper_opts)
    else:
        mapper = LogColorMapper(**mapper_opts)

    p = figure(toolbar_location=None, tools='', title="", x_axis_type=mapper_type, x_range=(1, 1000))
    p.title.text = f"{palette} with {mapper_type} mapping"
    p.circle(x='x', y='y', alpha=0.8, source=source, size=6,
             fill_color=transform('x', mapper), line_color=None)

    color_bar = ColorBar(color_mapper=mapper, ticker=p.xaxis.ticker, formatter=p.xaxis.formatter,
                         location=(0,0), orientation='horizontal', padding=0)

    p.add_layout(color_bar, 'below')
    return p

p1 = make_plot('linear', 'Viridis256')
p2 = make_plot('log', 'Viridis256')
p3 = make_plot('linear', 'Viridis6')
p4 = make_plot('log', 'Viridis6')

p5 = figure(toolbar_location=None, tools='', title="", x_range=(1, 1000), plot_width=800, plot_height=300)
p5.title.text = f"Viridis256 with linear mapping and low/high = 200/800 = pink/grey"
mapper = LinearColorMapper(palette="Viridis256", low=200, high=800, low_color="pink", high_color="darkgrey")
p5.circle(x='x', y='y', alpha=0.8, source=source, size=6,
         fill_color=transform('x', mapper), line_color=None)

show(column(
    gridplot([p1, p2, p3, p4], ncols=2, plot_width=400, plot_height=300, toolbar_location=None),
    p5
))

import numpy as np

from bokeh.plotting import figure, output_file, show

N = 4000
x = np.random.random(size=N) * 100
y = np.random.random(size=N) * 100
radii = np.random.random(size=N) * 1.5
colors = [
    "#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*x, 30+2*y)
]

TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"

p = figure(tools=TOOLS)

p.scatter(x, y, radius=radii,
          fill_color=colors, fill_alpha=0.6,
          line_color=None)

output_file("color_scatter.html", title="color_scatter.py example")

show(p)  # open a browser

import colorsys

import yaml

from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CustomJS, Slider
from bokeh.plotting import curdoc, figure, output_file, show
from bokeh.themes import Theme


# for plot 2: create colour spectrum of resolution N and brightness I, return as list of decimal RGB value tuples
def generate_color_range(N, I):
    HSV_tuples = [ (x*1.0/N, 0.5, I) for x in range(N) ]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    for_conversion = []
    for RGB_tuple in RGB_tuples:
        for_conversion.append((int(RGB_tuple[0]*255), int(RGB_tuple[1]*255), int(RGB_tuple[2]*255)))
    hex_colors = [ rgb_to_hex(RGB_tuple) for RGB_tuple in for_conversion ]
    return hex_colors, for_conversion

# convert RGB tuple to hexadecimal code
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

# convert hexadecimal to RGB tuple
def hex_to_dec(hex):
    red = ''.join(hex.strip('#')[0:2])
    green = ''.join(hex.strip('#')[2:4])
    blue = ''.join(hex.strip('#')[4:6])
    return (int(red, 16), int(green, 16), int(blue,16))

# plot 1: create a color block with RGB values adjusted with sliders

# initialise a white block for the first plot
hex_color = rgb_to_hex((255, 255, 255))

# initialise the text color as black. This will be switched to white if the block color gets dark enough
text_color = '#000000'

# create a data source to enable refreshing of fill & text color
source = ColumnDataSource(data=dict(color=[hex_color], text_color=[text_color]))

# create first plot, as a rect() glyph and centered text label, with fill and text color taken from source
p1 = figure(x_range=(-8, 8), y_range=(-4, 4),
            plot_width=600, plot_height=300,
            title='move sliders to change', tools='')

p1.rect(0, 0, width=18, height=10, fill_color='color',
        line_color = 'black', source=source)

p1.text(0, 0, text='color', text_color='text_color',
        alpha=0.6667, text_font_size='48px', text_baseline='middle',
        text_align='center', source=source)

red_slider = Slider(title="R", start=0, end=255, value=255, step=1)
green_slider = Slider(title="G", start=0, end=255, value=255, step=1)
blue_slider = Slider(title="B", start=0, end=255, value=255, step=1)

# the callback function to update the color of the block and associated label text
# NOTE: the JS functions for converting RGB to hex are taken from the excellent answer
# by Tim Down at http://stackoverflow.com/questions/5623838/rgb-to-hex-and-hex-to-rgb
callback = CustomJS(args=dict(source=source, red=red_slider, blue=blue_slider, green=green_slider), code="""
    function componentToHex(c) {
        var hex = c.toString(16)
        return hex.length == 1 ? "0" + hex : hex
    }
    function rgbToHex(r, g, b) {
        return "#" + componentToHex(r) + componentToHex(g) + componentToHex(b)
    }
    function toInt(v) {
       return v | 0
    }
    const color = source.data['color']
    const text_color = source.data['text_color']
    const R = toInt(red.value)
    const G = toInt(green.value)
    const B = toInt(blue.value)
    color[0] = rgbToHex(R, G, B)
    text_color[0] = '#ffffff'
    if ((R > 127) || (G > 127) || (B > 127)) {
        text_color[0] = '#000000'
    }
    source.change.emit()
""")

red_slider.js_on_change('value', callback)
blue_slider.js_on_change('value', callback)
green_slider.js_on_change('value', callback)

# plot 2: create a color spectrum with a hover-over tool to inspect hex codes

brightness = 0.8 # change to have brighter/darker colors
crx = list(range(1,1001)) # the resolution is 1000 colors
cry = [ 5 for i in range(len(crx)) ]
crcolor, crRGBs = generate_color_range(1000,brightness) # produce spectrum

# make data source object to allow information to be displayed by hover tool
crsource = ColumnDataSource(data=dict(x=crx, y=cry, crcolor=crcolor, RGBs=crRGBs))

# create second plot
p2 = figure(x_range=(0,1000), y_range=(0,10),
            plot_width=600, plot_height=150,
            tools='hover', title='hover over color')

color_range1 = p2.rect(x='x', y='y', width=1, height=10,
                       color='crcolor', source=crsource)

# set up hover tool to show color hex code and sample swatch
p2.hover.tooltips = [
    ('color', '$color[hex, rgb, swatch]:crcolor'),
    ('RGB levels', '@RGBs')
]

# theme everything for a cleaner look
curdoc().theme = Theme(json=yaml.load("""
attrs:
    Plot:
        toolbar_location: null
    Grid:
        grid_line_color: null
    Axis:
        axis_line_color: null
        major_label_text_color: null
        major_tick_line_color: null
        minor_tick_line_color: null
""", Loader=yaml.SafeLoader))

layout = row(
    column(red_slider, green_slider, blue_slider),
    column(p1, p2)
)

output_file("color_sliders.html", title="color_sliders.py example")

show(layout)
import numpy as np

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
plot = figure(plot_height=400, plot_width=400, title="my sine wave",
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
from bokeh.io import save
from bokeh.layouts import column
from bokeh.models import Div, Paragraph
from bokeh.util.browser import view

template = """
{% block postamble %}
<style>
.bk.custom {
    border-radius: 0.5em;
    padding: 1em;
}
.bk.custom-1 {
    border: 3px solid #2397D8;
}
.bk.custom-2 {
    border: 3px solid #14999A;
    background-color: whitesmoke;
}
</style>
{% endblock %}
"""

p = Paragraph(text="The divs below were configured with additional css_classes:")

div1 = Div(text="""
<p> This Bokeh Div adds the style classes:<p>
<pre>
.bk.custom {
    border-radius: 0.5em;
    padding: 1em;
}
.bk.custom-1 {
    border: 3px solid #2397D8;
}
</pre>
""")
div1.css_classes = ["custom", "custom-1"]

div2 = Div(text="""
<p> This Bokeh Div adds the style classes:<p>
<pre>
.bk.custom {
    border-radius: 0.5em;
    padding: 1em;
}
.bk.custom-2 {
    border: 3px solid #14999A;
    background-color: whitesmoke;
}
</pre>
""")
div2.css_classes = ["custom", "custom-2"]

save(column(p, div1, div2), template=template)
show("css_classes.html")

import pandas as pd

from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.sampledata.stocks import MSFT

df = pd.DataFrame(MSFT)[:51]
inc = df.close > df.open
dec = df.open > df.close

p = figure(plot_width=1000, title="MSFT Candlestick with Custom X-Axis")

# map dataframe indices to date strings and use as label overrides
p.xaxis.major_label_overrides = {
    i: date.strftime('%b %d') for i, date in enumerate(pd.to_datetime(df["date"]))
}
p.xaxis.bounds = (0, df.index[-1])
p.x_range.range_padding = 0.05

p.segment(df.index, df.high, df.index, df.low, color="black")
p.vbar(df.index[inc], 0.5, df.open[inc], df.close[inc], fill_color="#D5E1DD", line_color="black")
p.vbar(df.index[dec], 0.5, df.open[dec], df.close[dec], fill_color="#F2583E", line_color="black")

output_file("custom_datetime_axis.html", title="custom_datetime_axis.py example")

show(p)
from bokeh.io import save
from bokeh.plotting import figure
from bokeh.util.browser import view

template = """
{% block preamble %}
<style>
* { box-sizing: border-box; }
.plots { display: flex; flex-direction: row; width: 100%; }
.p { width: 33.3%; padding: 50px; }
.p:nth-child(1) { background-color: red; }
.p:nth-child(2) { background-color: green; }
.p:nth-child(3) { background-color: blue; }
</style>
{% endblock %}
{% block body %}
<body style="background-color: lightgray;">
    {{ self.inner_body() }}
</body>
{% endblock %}
{% block contents %}
<div>
<p>This example shows how different Bokeh Document roots may be embedded in custom
templates. The individal plots were embedded in divs using the embed macro:
<pre>
    &lt;div class="p"&gt;&#123;&#123; embed(roots.p0) &#125;&#125;&lt;/div&gt;
    &lt;div class="p"&gt;&#123;&#123; embed(roots.p1) &#125;&#125;&lt;/div&gt;
    &lt;div class="p"&gt;&#123;&#123; embed(roots.p2) &#125;&#125;&lt;/div&gt;
</pre>
And the divs are styled using standard CSS in the template:
<pre>
    .p { width: 33.3%; padding: 50px; }
    .p:nth-child(1) { background-color: red; }
    .p:nth-child(2) { background-color: green; }
    .p:nth-child(3) { background-color: blue; }
</pre>
</p>
</div>
<div class="plots">
    <div class="p">{{ embed(roots.p0) }}</div>
    <div class="p">{{ embed(roots.p1) }}</div>
    <div class="p">{{ embed(roots.p2) }}</div>
</div>
</div>
{% endblock %}
"""

x = [1, 2, 3]
y = [1, 2, 3]

p0 = figure(name="p0", sizing_mode="scale_width")
p0.scatter(x, y, size=20, fill_color="red")
p1 = figure(name="p1", sizing_mode="scale_width")
p1.scatter(x, y, size=20, fill_color="green")
p2 = figure(name="p2", sizing_mode="scale_width")
p2.scatter(x, y, size=20, fill_color="blue")

save([p0, p1, p2], template=template)
view("custom_layout.html")

import pandas as pd

from bokeh.plotting import figure, show
from bokeh.sampledata.periodic_table import elements

elements = elements.copy()
elements = elements[elements.group != "-"]
elements.sort_values('metal', inplace=True)

colormap = {
    "alkali metal"         : "#a6cee3",
    "alkaline earth metal" : "#1f78b4",
    "halogen"              : "#fdbf6f",
    "metal"                : "#b2df8a",
    "metalloid"            : "#33a02c",
    "noble gas"            : "#bbbb88",
    "nonmetal"             : "#baa2a6",
    "transition metal"     : "#e08e79",
}

data=dict(
    atomic_number=elements["atomic number"],
    sym=elements["symbol"],
    name=elements["name"],
    atomic_mass = pd.to_numeric(elements['atomic mass'], errors="coerce"),
    density=elements['density'],
    metal=[x.title() for x in elements["metal"]],
    type_color=[colormap[x] for x in elements["metal"]]
)

mass_format = '{0.00}'

TOOLTIPS = """
    <div style="width: 62px; height: 62px; opacity: .8; padding: 5px; background-color: @type_color;>
    <h1 style="margin: 0; font-size: 12px;"> @atomic_number </h1>
    <h1 style="margin: 0; font-size: 24px;"><strong> @sym </strong></h1>
    <p style=" margin: 0; font-size: 8px;"><strong> @name </strong></p>
    <p style="margin: 0; font-size: 8px;"> @atomic_mass{mass_format} </p>
    </div>
""".format(mass_format=mass_format)

p = figure(plot_width=900, plot_height=450, tooltips=TOOLTIPS, title='Densities by Atomic Mass')
p.background_fill_color = "#fafafa"

p.circle('atomic_mass', 'density', size=12, source=data, color='type_color',
         line_color="black", legend_field='metal', alpha=0.9)

p.legend.glyph_width = 30
p.legend.glyph_height = 30
p.xaxis.axis_label= 'Atomic Mass'
p.yaxis.axis_label= 'Density'
p.xgrid.grid_line_color = None
p.toolbar_location = None

legend = p.legend[0]
p.add_layout(legend, 'right')
legend.border_line_color = None

show(p)
import pandas as pd

from bokeh.plotting import figure, show
from bokeh.sampledata.periodic_table import elements

elements = elements.copy()
elements = elements[elements.group != "-"]
elements.sort_values('metal', inplace=True)

colormap = {
    "alkali metal"         : "#a6cee3",
    "alkaline earth metal" : "#1f78b4",
    "halogen"              : "#fdbf6f",
    "metal"                : "#b2df8a",
    "metalloid"            : "#33a02c",
    "noble gas"            : "#bbbb88",
    "nonmetal"             : "#baa2a6",
    "transition metal"     : "#e08e79",
}

data=dict(
    atomic_number=elements["atomic number"],
    sym=elements["symbol"],
    name=elements["name"],
    atomic_mass = pd.to_numeric(elements['atomic mass'], errors="coerce"),
    density=elements['density'],
    metal=[x.title() for x in elements["metal"]],
    type_color=[colormap[x] for x in elements["metal"]]
)

mass_format = '{0.00}'

TOOLTIPS = """
    <div style="width: 62px; height: 62px; opacity: .8; padding: 5px; background-color: @type_color;>
    <h1 style="margin: 0; font-size: 12px;"> @atomic_number </h1>
    <h1 style="margin: 0; font-size: 24px;"><strong> @sym </strong></h1>
    <p style=" margin: 0; font-size: 8px;"><strong> @name </strong></p>
    <p style="margin: 0; font-size: 8px;"> @atomic_mass{mass_format} </p>
    </div>
""".format(mass_format=mass_format)

p = figure(plot_width=900, plot_height=450, tooltips=TOOLTIPS, title='Densities by Atomic Mass')
p.background_fill_color = "#fafafa"

p.circle('atomic_mass', 'density', size=12, source=data, color='type_color',
         line_color="black", legend_field='metal', alpha=0.9)

p.legend.glyph_width = 30
p.legend.glyph_height = 30
p.xaxis.axis_label= 'Atomic Mass'
p.yaxis.axis_label= 'Density'
p.xgrid.grid_line_color = None
p.toolbar_location = None

legend = p.legend[0]
p.add_layout(legend, 'right')
legend.border_line_color = None

show(p)
from bokeh.io import output_file, show
from bokeh.models import CustomJSHover, HoverTool
from bokeh.plotting import figure
from bokeh.tile_providers import CARTODBPOSITRON, get_provider

output_file("customjs_hover.html")

# range bounds supplied in web mercator coordinates
p = figure(x_range=(-2000000, 6000000), y_range=(-1000000, 7000000),
           x_axis_type="mercator", y_axis_type="mercator")
p.add_tile(get_provider(CARTODBPOSITRON))

p.circle(x=[0, 2000000, 4000000], y=[4000000, 2000000, 0], size=30)

code = """
    var projections = Bokeh.require("core/util/projections");
    var x = special_vars.x
    var y = special_vars.y
    var coords = projections.wgs84_mercator.inverse([x, y])
    return coords[%d].toFixed(2)
"""

p.add_tools(HoverTool(
    tooltips=[
        ( 'lon', '$x{custom}' ),
        ( 'lat', '$y{custom}' ),
    ],

    formatters={
        '$x' : CustomJSHover(code=code % 0),
        '$y' : CustomJSHover(code=code % 1),
    }
))

show(p)
# Based on https://www.reddit.com/r/dataisbeautiful/comments/6qnkg0/google_search_interest_follows_the_path_of_the/

import pandas as pd
import shapefile as shp

from bokeh.models import ColorBar, ColumnDataSource, Label, LinearColorMapper
from bokeh.palettes import YlOrRd5
from bokeh.plotting import figure, show
from bokeh.sampledata.us_states import data

states = pd.DataFrame.from_dict(data, orient="index")
states.drop(["AK", "HI"], inplace=True)

trends = pd.read_csv("eclipse_data/trends.csv")

states.set_index("name", inplace=True)
trends.set_index("Region", inplace=True)

states["trend"] = trends["solar eclipse"]

upath17 = shp.Reader("eclipse_data/upath17")
(totality_path,) = upath17.shapes()

p = figure(plot_width=1000, plot_height=600, background_fill_color="#333344",
           tools="", toolbar_location=None, x_axis_location=None, y_axis_location=None)

p.grid.grid_line_color = None

p.title.text = "Google Search Trends and the Path of Solar Eclipse, 21 August 2017"
p.title.align = "center"
p.title.text_font_size = "21px"
p.title.text_color = "#333344"

mapper = LinearColorMapper(palette=list(reversed(YlOrRd5)), low=0, high=100)

source = ColumnDataSource(data=dict(
    state_xs=list(states.lons),
    state_ys=list(states.lats),
    trend=states.trend,
))
us = p.patches("state_xs", "state_ys",
    fill_color=dict(field="trend", transform=mapper),
    source=source,
    line_color="#333344", line_width=1)

p.x_range.renderers = [us]
p.y_range.renderers = [us]

totality_x, totality_y = zip(*totality_path.points)
p.patch(totality_x, totality_y,
    fill_color="black", fill_alpha=0.7,
    line_color=None)

path = Label(
    x=-76.3, y=31.4,
    angle=-36.5, angle_units="deg",
    text="Solar eclipse path of totality",
    text_baseline="middle", text_font_size="11px", text_color="silver")
p.add_layout(path)

color_bar = ColorBar(
    color_mapper=mapper,
    location="bottom_left", orientation="horizontal",
    title="Popularity of \"solar eclipse\" search term",
    title_text_font_size="16px", title_text_font_style="bold",
    title_text_color="lightgrey", major_label_text_color="lightgrey",
    background_fill_alpha=0.0)
p.add_layout(color_bar)

notes = Label(
    x=0, y=0, x_units="screen", y_units="screen",
    x_offset=40, y_offset=20,
    text="Source: Google Trends, NASA Scientific Visualization Studio",
    level="overlay",
    text_font_size="11px", text_color="gray")
p.add_layout(notes)

show(p)
import pandas as pd

from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, output_file, show
from bokeh.sampledata.periodic_table import elements

elements = elements.copy()
elements = elements[elements["atomic number"] <= 82]
elements = elements[~pd.isnull(elements["melting point"])]
mass = [float(x.strip("[]")) for x in elements["atomic mass"]]
elements["atomic mass"] = mass

palette = ["#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0",
           "#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"]

melting_points = elements["melting point"]
low = min(melting_points)
high = max(melting_points)
melting_point_inds = [int(10*(x-low)/(high-low)) for x in melting_points] #gives items in colors a value from 0-10
elements['melting_colors'] = [palette[i] for i in melting_point_inds]

TITLE = "Density vs Atomic Weight of Elements (colored by melting point)"
TOOLS = "hover,pan,wheel_zoom,box_zoom,reset,save"

p = figure(tools=TOOLS, toolbar_location="above", plot_width=1200, title=TITLE)
p.toolbar.logo = "grey"
p.background_fill_color = "#dddddd"
p.xaxis.axis_label = "atomic weight (amu)"
p.yaxis.axis_label = "density (g/cm^3)"
p.grid.grid_line_color = "white"
p.hover.tooltips = [
    ("name", "@name"),
    ("symbol:", "@symbol"),
    ("density", "@density"),
    ("atomic weight", "@{atomic mass}"),
    ("melting point", "@{melting point}")
]

source = ColumnDataSource(elements)

p.circle("atomic mass", "density", size=12, source=source,
         color='melting_colors', line_color="black", fill_alpha=0.8)

labels = LabelSet(x="atomic mass", y="density", text="symbol", y_offset=8,
                  text_font_size="11px", text_color="#555555",
                  source=source, text_align='center')
p.add_layout(labels)

output_file("elements.html", title="elements.py example")

show(p)

from bokeh.layouts import column
from bokeh.models import CustomJS, Div, FileInput
from bokeh.plotting import output_file, show

# Set up widgets
file_input = FileInput(accept=".csv,.json")
para = Div(text="<h1>FileInput Values:</h1><p>filename:<p>b64 value:")

# Create CustomJS callback to display file_input attributes on change
callback = CustomJS(args=dict(para=para, file_input=file_input), code="""
    para.text = "<h1>FileInput Values:</h1><p>filename: " + file_input.filename  + "<p>b64 value: " + file_input.value
""")

# Attach callback to FileInput widget
file_input.js_on_change('change', callback)


output_file("file_input.html")

show(column(file_input, para))

import numpy as np

from bokeh.plotting import figure, output_file, show

x = np.linspace(-6, 6, 500)
y = 8*np.sin(x)*np.sinc(x)

p = figure(plot_width=800, plot_height=300, title="", tools="",
           toolbar_location=None, match_aspect=True)

p.line(x, y, color="navy", alpha=0.4, line_width=4)
p.background_fill_color = "#efefef"
p.xaxis.fixed_location = 0
p.yaxis.fixed_location = 0

output_file("fixed_axis.html", title="fixed_axis.py example")

show(p)

from bokeh.io import output_file, show
from bokeh.models import GeoJSONDataSource
from bokeh.plotting import figure
from bokeh.sampledata.sample_geojson import geojson

p = figure(tooltips=[("Organisation Name", "@OrganisationName")])

p.circle(x='x', y='y', line_color=None, fill_alpha=0.8, size=20,
         source=GeoJSONDataSource(geojson=geojson))

output_file("geojson_points.html", title="GeoJSON Points")

show(p)

import numpy as np

from bokeh.io import curdoc, output_file, show
from bokeh.layouts import gridplot
from bokeh.plotting import figure
from bokeh.themes import Theme

N = 9

x = np.linspace(-2, 2, N)
y = x**2
sizes = np.linspace(10, 20, N)

xpts = np.array([-.09, -.12, .0, .12, .09])
ypts = np.array([-.1, .02, .1, .02, -.1])

children = []

p = figure(title="annular_wedge")
p.annular_wedge(x, y, 10, 20, 0.6, 4.1, color="#8888ee",
                inner_radius_units="screen", outer_radius_units="screen")
children.append(p)

p = figure(title="annulus")
p.annulus(x, y, 10, 20, color="#7FC97F",
          inner_radius_units="screen", outer_radius_units = "screen")
children.append(p)

p = figure(title="arc")
p.arc(x, y, 20, 0.6, 4.1, radius_units="screen", color="#BEAED4", line_width=3)
children.append(p)

p = figure(title="bezier")
p.bezier(x, y, x+0.2, y, x+0.1, y+0.1, x-0.1, y-0.1, color="#D95F02", line_width=2)
children.append(p)

p = figure(title="circle")
p.circle(x, y, radius=0.1, color="#3288BD")
children.append(p)

p = figure(title="ellipse")
p.ellipse(x, y, 15, 25, angle=-0.7, color="#1D91C0",
       width_units="screen", height_units="screen")
children.append(p)

p = figure(title="Hbar")
p.hbar(y=x, height=0.5, left=0, right=y, color="#AA9348")
children.append(p)

p = figure(title="line")
p.line(x, y, color="#F46D43")
children.append(p)

p = figure(title="multi_line")
p.multi_line([xpts+xx for xx in x], [ypts+yy for yy in y],
    color="#8073AC", line_width=2)
children.append(p)

p = figure(title="multi_polygons")
p.multi_polygons(
    [[[xpts*2+xx, xpts+xx]] for xx in x],
    [[[ypts*3+yy, ypts+yy]] for yy in y],
    color="#FB9A99")
children.append(p)

p = figure(title="oval")
p.oval(x, y, 15, 25, angle=-0.7, color="#1D91C0",
       width_units="screen", height_units="screen")
children.append(p)

p = figure(title="patch")
p.patch(x, y, color="#A6CEE3")
children.append(p)

p = figure(title="patches")
p.patches([xpts+xx for xx in x], [ypts+yy for yy in y], color="#FB9A99")
children.append(p)

p = figure(title="quad")
p.quad(x, x-0.1, y, y-0.1, color="#B3DE69")
children.append(p)

p = figure(title="quadratic")
p.quadratic(x, y, x+0.2, y, x+0.3, y+1.4, color="#4DAF4A", line_width=3)
children.append(p)

p = figure(title="ray")
p.ray(x, y, 45, -0.7, color="#FB8072", line_width=2)
children.append(p)

p = figure(title="rect")
p.rect(x, y, 10, 20, color="#CAB2D6", width_units="screen", height_units="screen")
children.append(p)

p = figure(title="segment")
p.segment(x, y, x-0.1, y-0.1, color="#F4A582", line_width=3)
children.append(p)

p = figure(title="square")
p.square(x, y, size=sizes, color="#74ADD1")
children.append(p)

p = figure(title="Vbar")
p.vbar(x=x, width=0.5, bottom=0, top=y, color="#CAB2D6")
children.append(p)

p = figure(title="wedge")
p.wedge(x, y, 15, 0.6, 4.1, radius_units="screen", color="#B3DE69")
children.append(p)

p = figure(title="Marker: circle_x")
p.scatter(x, y, marker="circle_x", size=sizes, color="#DD1C77", fill_color=None)
children.append(p)

p = figure(title="Marker: triangle")
p.scatter(x, y, marker="triangle", size=sizes, color="#99D594", line_width=2)
children.append(p)

p = figure(title="Marker: circle")
p.scatter(x, y, marker="o", size=sizes, color="#80B1D3", line_width=3)
children.append(p)

p = figure(title="Marker: x")
p.scatter(x, y, marker="x", size=sizes, color="#B3DE69", line_width=3)
children.append(p)

p = figure(title="Marker: cross")
p.scatter(x, y, marker="cross", size=sizes, color="#E6550D", line_width=2)
children.append(p)

p = figure(title="Marker: dash")
p.scatter(x, y, marker="dash", angle=-0.7, size=sizes, color="#E6550D")
children.append(p)

p = figure(title="Marker: diamond")
p.scatter(x, y, marker="diamond", size=sizes, color="#1C9099", line_width=2)
children.append(p)

p = figure(title="hex")
p.scatter(x, y, marker="hex", size=sizes, color="#99D594")
children.append(p)

p = figure(title="Marker: inverted_triangle")
p.scatter(x, y, marker="inverted_triangle", size=sizes, color="#DE2D26")
children.append(p)

p = figure(title="Marker: square_x")
p.scatter(x, y, marker="square_x", size=sizes, color="#FDAE6B",
    fill_color=None, line_width=2)
children.append(p)

p = figure(title="Marker: asterisk")
p.scatter(x, y, marker="asterisk", size=sizes, color="#F0027F", line_width=2)
children.append(p)

p = figure(title="Marker: square_cross")
p.scatter(x, y, marker="square_cross", size=sizes, color="#7FC97F",
    fill_color=None, line_width=2)
children.append(p)

p = figure(title="Marker: diamond_cross")
p.scatter(x, y, marker="diamond_cross", size=sizes, color="#386CB0",
    fill_color=None, line_width=2)
children.append(p)

p = figure(title="Marker: circle_cross")
p.scatter(x, y, marker="circle_cross", size=sizes, color="#FB8072",
    fill_color=None, line_width=2)
children.append(p)

# simplify theme by turning off axes and gridlines
curdoc().theme = Theme(json={
    "attrs": {
        "Axis": {
            "visible": False
            },
        "Grid": {
            "visible": False
            }
        }
    })

output_file("glyphs.html", title="glyphs.py example")

show(gridplot(children, ncols=4, plot_width=200, plot_height=200))  # open a browser

from bokeh.io import output_file, show
from bokeh.models import GMapOptions, Label
from bokeh.plotting import gmap

output_file("gmap.html")

map_options = GMapOptions(lat=30.2861, lng=-97.7394, map_type="roadmap", zoom=13)

# replace with your google api key
p = gmap("GOOGLE_API_KEY", map_options)

if p.api_key == "GOOGLE_API_KEY":
    p.add_layout(Label(x=140, y=400, x_units='screen', y_units='screen',
                       text='Replace GOOGLE_API_KEY with your own key',
                       text_color='red'))

show(p)
import numpy as np

from bokeh.models import (ColumnDataSource, HoverTool, NodesAndLinkedEdges,
                          StaticLayoutProvider, TapTool,)
from bokeh.palettes import Set3_12
from bokeh.plotting import figure, output_file, show
from bokeh.sampledata.airport_routes import airports, routes
from bokeh.sampledata.us_states import data as us_states

output_file("graphs.html")

airports.set_index("AirportID", inplace=True)
airports.index.rename("index", inplace=True)
routes.rename(columns={"SourceID": "start", "DestinationID": "end"}, inplace=True)

lats, lons = [], []
for k, v in us_states.items():
    lats.append(np.array(v['lats']))
    lons.append(np.array(v['lons']))

source = ColumnDataSource(data=dict(lats=lats, lons=lons))

graph_layout = dict(zip(airports.index.astype(str), zip(airports.Longitude, airports.Latitude)))
layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

fig = figure(x_range=(-180, -60), y_range=(15,75),
              x_axis_label="Longitude", y_axis_label="Latitude",
              plot_width=800, plot_height=600, background_fill_color=Set3_12[4],
              background_fill_alpha=0.2, tools='box_zoom,reset')

fig.patches(xs="lons", ys="lats", line_color='grey', line_width=1.0,
             fill_color=Set3_12[10], source=source)

r = fig.graph(airports, routes, layout_provider,
              ## node style props
              node_fill_color=Set3_12[3], node_fill_alpha=0.4, node_line_color="black", node_line_alpha=0.3,
              node_nonselection_fill_color=Set3_12[3], node_nonselection_fill_alpha=0.2, node_nonselection_line_alpha=0.1,
              node_selection_fill_color=Set3_12[3], node_selection_fill_alpha=0.8, node_selection_line_alpha=0.3,
              ## edge style props
              edge_line_color="black", edge_line_alpha=0.04,
              edge_hover_line_alpha=0.6, edge_hover_line_color=Set3_12[1],
              edge_nonselection_line_color="black", edge_nonselection_line_alpha=0.01,
              edge_selection_line_alpha=0.6, edge_selection_line_color=Set3_12[1],
              ## graph policies
              inspection_policy=NodesAndLinkedEdges(), selection_policy=NodesAndLinkedEdges())

hover = HoverTool(tooltips=[("Airport", "@Name (@IATA), @City ")], renderers=[r])
tap = TapTool(renderers=[r])
fig.add_tools(hover, tap)

show(fig)
import numpy as np

from bokeh.models import (ColumnDataSource, HoverTool, NodesAndLinkedEdges,
                          StaticLayoutProvider, TapTool,)
from bokeh.palettes import Set3_12
from bokeh.plotting import figure, output_file, show
from bokeh.sampledata.airport_routes import airports, routes
from bokeh.sampledata.us_states import data as us_states

output_file("graphs.html")

airports.set_index("AirportID", inplace=True)
airports.index.rename("index", inplace=True)
routes.rename(columns={"SourceID": "start", "DestinationID": "end"}, inplace=True)

lats, lons = [], []
for k, v in us_states.items():
    lats.append(np.array(v['lats']))
    lons.append(np.array(v['lons']))

source = ColumnDataSource(data=dict(lats=lats, lons=lons))

graph_layout = dict(zip(airports.index.astype(str), zip(airports.Longitude, airports.Latitude)))
layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

fig = figure(x_range=(-180, -60), y_range=(15,75),
              x_axis_label="Longitude", y_axis_label="Latitude",
              plot_width=800, plot_height=600, background_fill_color=Set3_12[4],
              background_fill_alpha=0.2, tools='box_zoom,reset')

fig.patches(xs="lons", ys="lats", line_color='grey', line_width=1.0,
             fill_color=Set3_12[10], source=source)

r = fig.graph(airports, routes, layout_provider,
              ## node style props
              node_fill_color=Set3_12[3], node_fill_alpha=0.4, node_line_color="black", node_line_alpha=0.3,
              node_nonselection_fill_color=Set3_12[3], node_nonselection_fill_alpha=0.2, node_nonselection_line_alpha=0.1,
              node_selection_fill_color=Set3_12[3], node_selection_fill_alpha=0.8, node_selection_line_alpha=0.3,
              ## edge style props
              edge_line_color="black", edge_line_alpha=0.04,
              edge_hover_line_alpha=0.6, edge_hover_line_color=Set3_12[1],
              edge_nonselection_line_color="black", edge_nonselection_line_alpha=0.01,
              edge_selection_line_alpha=0.6, edge_selection_line_color=Set3_12[1],
              ## graph policies
              inspection_policy=NodesAndLinkedEdges(), selection_policy=NodesAndLinkedEdges())

hover = HoverTool(tooltips=[("Airport", "@Name (@IATA), @City ")], renderers=[r])
tap = TapTool(renderers=[r])
fig.add_tools(hover, tap)

show(fig)
import numpy as np

from bokeh.plotting import figure, gridplot, output_file, show

N = 50

x = np.linspace(0, 4*np.pi, N)
y = np.sin(x)

TOOLS = "pan,wheel_zoom,box_zoom,reset,save,crosshair"

l = figure(title="line", tools=TOOLS, plot_width=300, plot_height=300)
l.line(x,y, line_width=3, color="gold")

aw = figure(title="annular wedge", tools=TOOLS, plot_width=300, plot_height=300)
aw.annular_wedge(x, y, 10, 20, 0.6, 4.1, color="navy", alpha=0.5,
    inner_radius_units="screen", outer_radius_units="screen")

bez = figure(title="bezier", tools=TOOLS, plot_width=300, plot_height=300)
bez.bezier(x, y, x+0.4, y, x+0.1, y+0.2, x-0.1, y-0.2,
    line_width=2, color="olive")

q = figure(title="quad", tools=TOOLS, plot_width=300, plot_height=300)
q.quad(x, x-0.2, y, y-0.2, color="tomato", alpha=0.4)

# specify "empty" grid cells with None
p = gridplot([[l, None, aw], [bez, q, None]])

output_file("grid.html", title="grid.py example")

show(p)
from bokeh.io import output_file, show
from bokeh.models import ImageURLTexture
from bokeh.plotting import figure

output_file("hatch_custom_image.html")

clips = [
    'https://static.bokeh.org/clipart/clipart-colorful-circles-64x64.png',
    'https://static.bokeh.org/clipart/clipart-celtic-repeating-64x64.png',
    'https://static.bokeh.org/clipart/clipart-tie-dye-64x64.png',
    'https://static.bokeh.org/clipart/clipart-ferns-64x64.png',
    'https://static.bokeh.org/clipart/clipart-diamond-tiles-64x64.png',
    'https://static.bokeh.org/clipart/clipart-abstract-squares-64x64.png',
    'https://static.bokeh.org/clipart/clipart-interlaced-pattern-64x64.png',
    'https://static.bokeh.org/clipart/clipart-mosaic-64x64.png',
    'https://static.bokeh.org/clipart/clipart-gold-stars-64x64.png',
    'https://static.bokeh.org/clipart/clipart-voronoi-2d-64x64.png',
    'https://static.bokeh.org/clipart/clipart-victorian-background-64x64.png',
    'https://static.bokeh.org/clipart/clipart-wallpaper-circles-64x64.png',
    'https://static.bokeh.org/clipart/clipart-beads-64x64.png',
]

p = figure(plot_width=900, plot_height=450, toolbar_location=None, tools="")
p.x_range.range_padding = p.y_range.range_padding = 0

for i, url in enumerate(clips):
    p.vbar(x=i+0.5, top=5, width=0.9, fill_color=None, line_color="black",
           hatch_pattern=dict(value='image'), hatch_extra={"image": ImageURLTexture(url=url)})

show(p)

import numpy as np

from bokeh.io import show
from bokeh.plotting import figure

x = np.linspace(0, 6*np.pi, 100)
y = np.sin(x)

p = figure(plot_height=250, sizing_mode="stretch_width", x_range=(0, 6*np.pi), tools="", toolbar_location=None)
p.line(x, y)

ticks = np.linspace(0, 6*np.pi, 13)

labels = dict(zip(ticks[1:], ["π/2", "π", "3π/2", "2π", "5π/2", "3π", "7π/2", "4π", "9π/2", "5π",  "11π/2", "6π",]))
p.xaxis.ticker = ticks
p.xgrid.ticker = ticks[1::2]
p.xaxis.major_label_overrides = labels

p.ygrid.grid_line_color = None

p.xgrid.band_hatch_pattern = "/"
p.xgrid.band_hatch_alpha = 0.6
p.xgrid.band_hatch_color = "lightgrey"
p.xgrid.band_hatch_weight = 0.5
p.xgrid.band_hatch_scale = 10

show(p)
from bokeh.core.enums import HatchPattern
from bokeh.io import output_file, show
from bokeh.plotting import figure

output_file("hatch_patterns.html")

pats = list(HatchPattern)
lefts  = [3,  4, 6,  5, 3, 7, 4,  5, 3,  4,  7,  5, 6,  4, 5, 6, 8]
scales = [12, 6, 12, 4, 8, 4, 10, 8, 18, 16, 12, 8, 12, 8, 6, 8, 12]

p = figure(y_range=pats, plot_height=900, plot_width=600, title="Built-in Hatch Patterns",
           toolbar_location=None, tools="", y_axis_location="right")

r = p.hbar(y=pats, left=lefts, right=10, height=0.9, fill_color="#fafafa", line_color="grey",
       hatch_pattern=pats, hatch_scale=scales, hatch_color="black", hatch_weight=0.5, hatch_alpha=0.5)

p.ygrid.grid_line_color = None
p.x_range.end = 10

show(p)

import numpy as np

from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
from bokeh.util.hex import hexbin

n = 5000
x = np.random.standard_normal(n)
y = np.random.standard_normal(n)

bins = hexbin(x, y, 0.1)

p = figure(title="Manual hex bin for 5000 points", tools="wheel_zoom,pan,reset",
           match_aspect=True, background_fill_color='#FFC0CB')
p.grid.visible = False

p.hex_tile(q="q", r="r", size=0.2, line_color=None, source=bins,
           fill_color=linear_cmap('counts', 'Viridis256', 0, max(bins.counts)))

show(p)
