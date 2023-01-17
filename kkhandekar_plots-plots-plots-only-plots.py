#Libraries



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import mpld3

from mpld3 import plugins
np.random.seed(9615)



# generate df

N = 100

df = pd.DataFrame((.1 * (np.random.random((N, 2)) - .5)).cumsum(0),

                  columns=['a', 'b'],)



# plot line + confidence interval

fig, ax = plt.subplots()

ax.grid(True, alpha=0.3)



for key, val in df.iteritems():

    l, = ax.plot(val.index, val.values, label=key)

    ax.fill_between(val.index,

                    val.values * .5, val.values * 1.5,

                    color=l.get_color(), alpha=.4)



# define interactive legend



handles, labels = ax.get_legend_handles_labels() # return lines and labels

interactive_legend = plugins.InteractiveLegendPlugin(zip(handles,

                                                         ax.collections),

                                                     labels,

                                                     alpha_unsel=0.5,

                                                     alpha_over=1.5, 

                                                     start_visible=True)

plugins.connect(fig, interactive_legend)



ax.set_xlabel('X')

ax.set_ylabel('Y')

ax.set_title('Interactive legend', size=15)



mpld3.display()
fig, ax = plt.subplots()

N = 50



scatter = ax.scatter(np.random.normal(size=N),

                     np.random.normal(size=N),

                     c=np.random.random(size=N),

                     s=1000 * np.random.random(size=N),

                     alpha=0.3,

                     cmap=plt.cm.jet)



ax.grid(color='white', linestyle='dotted')



ax.set_title("Scatter Plot", size=15)



labels = ['point {0}'.format(i + 1) for i in range(N)]

tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)

mpld3.plugins.connect(fig, tooltip)



mpld3.display()

fig, ax = plt.subplots()



x = np.linspace(-2, 2, 20)

y = x[:, None]

X = np.zeros((20, 20, 4))



X[:, :, 0] = np.exp(- (x - 1) ** 2 - (y) ** 2)

X[:, :, 1] = np.exp(- (x + 0.71) ** 2 - (y - 0.71) ** 2)

X[:, :, 2] = np.exp(- (x + 0.71) ** 2 - (y + 0.71) ** 2)

X[:, :, 3] = np.exp(-0.25 * (x ** 2 + y ** 2))



im = ax.imshow(X, extent=(10, 20, 10, 20),

               origin='lower', zorder=1, interpolation='nearest')

fig.colorbar(im, ax=ax)



ax.set_title('Image', size=20)



plugins.connect(fig, plugins.MousePosition(fontsize=14))



mpld3.display()
fig = plt.figure()



ax = fig.add_subplot(111)

ax.grid(color='gray', linestyle='dotted')



x = np.random.normal(size=1000)

ax.hist(x, 30, histtype='stepfilled', fc='lightblue', alpha=0.5);
fig, ax = plt.subplots()

x = np.linspace(-5, 15, 1000)

for offset in np.linspace(0, 3, 3):

    ax.plot(x, 0.9 * np.sin(x - offset), lw=2, alpha=0.4,

            label="Offset: {0}".format(offset))

ax.set_xlim(0, 10)

ax.set_ylim(-1.2, 1.0)

ax.text(5, -1.5, "Line Plot", size=18, ha='center')

ax.grid(color='lightgray', alpha=0.7, linestyle='dotted')

ax.legend(loc=4, fontsize='x-small')
fig, ax = plt.subplots(2, 2, figsize=(8, 6),sharex='col', sharey='row')

fig.subplots_adjust(hspace=0.3)



np.random.seed(0)



for axi in ax.flat:

    color = np.random.random(4)

    axi.plot(np.random.random(50), lw=1.5, c=color)

    axi.set_title("RGB = ({0:.2f}, {1:.2f}, {2:.2f})".format(*color),

                  size=14)

    axi.grid(color='lightgray', alpha=0.7, linestyle='dotted')
#Install PyGal Library

!pip install pygal -q
# Libraries

import pygal

from IPython.display import SVG, HTML
html_pygal = """

<!DOCTYPE html>

<html>

  <head>

  <script type="text/javascript" src="http://kozea.github.com/pygal.js/javascripts/svg.jquery.js"></script>

  <script type="text/javascript" src="http://kozea.github.com/pygal.js/javascripts/pygal-tooltips.js"></script>

    <!-- ... -->

  </head>

  <body>

    <figure>

      {pygal_render}

    </figure>

  </body>

</html>

"""
line_chart = pygal.Line()

line_chart.title = 'Line Plot'

line_chart.x_labels = map(str, range(100, 115))

line_chart.add('X', [0, 15.5, 18, 16.6,   25,   31, 36.4, 45.5, 46.3, 42.8])

line_chart.add('Y',  [0, 2, 5, 10.6, 15.8, 20.2,30, 39, 10.8, 23.8])

HTML(html_pygal.format(pygal_render=line_chart.render()))

line_chart = pygal.StackedLine(fill=True)

line_chart.title = 'Stacked Line'

line_chart.x_labels = map(str, range(100, 115))

line_chart.add('X', [None, None, 0, 16.6,   25,   31, 36.4, 45.5, 46.3, 42.8, 37.1])

line_chart.add('Y',  [None, None, None, None, None, None,    0,  3.9, 10.8, 23.8, 35.3])

HTML(html_pygal.format(pygal_render=line_chart.render()))
line_chart = pygal.Bar()

line_chart.title = 'Bar Plot'

line_chart.x_labels = map(str, range(100, 115))

line_chart.add('X', [None, None, 0, 16.6,   25,   31, 36.4, 45.5, 46.3, 42.8, 37.1])

line_chart.add('Y',  [None, None, None, None, None, None,    0,  3.9, 10.8, 23.8, 35.3])

HTML(html_pygal.format(pygal_render=line_chart.render()))
line_chart = pygal.StackedBar()

line_chart.title = 'Stacked Bar Plot'

line_chart.x_labels = map(str, range(100, 116))

line_chart.add('X', [None, None, 0, 16.6,   25,   31, 36.4, 45.5, 46.3, 42.8, 37.1])

line_chart.add('Y',  [14.2, 15.4, 15.3,  8.9,    9, 10.4,  8.9,  5.8,  6.7,  6.8,  7.5])

HTML(html_pygal.format(pygal_render=line_chart.render()))
line_chart = pygal.HorizontalBar()

line_chart.title = 'Horizontal Bar Plot'

line_chart.add('X', 4.5)

line_chart.add('Y', 2.3)

HTML(html_pygal.format(pygal_render=line_chart.render()))
hist = pygal.Histogram()

hist.add('Histogram Bar',  [(5, 1, 5), (6, 4, 10), (4, 8, 13)])

hist.render()

HTML(html_pygal.format(pygal_render=hist.render()))
xy_chart = pygal.XY(stroke=False)

xy_chart.title = 'Scatter Plot'

xy_chart.add('X', [(0, 0), (.1, .2), (.3, .1), (.5, 1), (.8, .6), (1, 1.08), (1.3, 1.1), (2, 3.23), (2.43, 2)])

xy_chart.add('Y', [(.1, .15), (.12, .23), (.4, .3), (.6, .4), (.21, .21), (.5, .3), (.6, .8), (.7, .8)])

HTML(html_pygal.format(pygal_render=xy_chart.render()))
pie_chart = pygal.Pie()

pie_chart.title = 'Pie Plot'

pie_chart.add('A', 5)

pie_chart.add('B', 30)

pie_chart.add('C', 20)

pie_chart.add('D', 30)

pie_chart.add('E', 15)

HTML(html_pygal.format(pygal_render=pie_chart.render()))
box_plot = pygal.Box()

box_plot.title = 'Box Plot'

box_plot.add('A', [6395, 8212, 7520, 7218, 12464, 1660, 2123, 8607])

box_plot.add('B', [7473, 8099, 11700, 2651, 6361, 1044, 3797, 9450])

box_plot.add('C', [3472, 2933, 4203, 5229, 5810, 1828, 9013, 4669])

HTML(html_pygal.format(pygal_render=box_plot.render()))

#Libraries

import bokeh

from bokeh.io import output_notebook, show

output_notebook()

import scipy.special

from bokeh.palettes import brewer



from bokeh.models import ColumnDataSource

from bokeh.palettes import Spectral6

from bokeh.plotting import figure

from bokeh.transform import factor_cmap

from bokeh.models import FactorRange

from bokeh.layouts import row
groups = ['A', 'B', 'C', 'D']

counts = [5, 3, 4, 2]



source = ColumnDataSource(data=dict(groups=groups, counts=counts))



p = figure(x_range=groups, plot_height=350, toolbar_location=None, title="Bar PLot")

p.vbar(x='groups', top='counts', width=0.9, source=source, legend_field="groups",

       line_color='white', fill_color=factor_cmap('groups', palette=Spectral6, factors=groups))



p.xgrid.grid_line_color = None

p.y_range.start = 0

p.y_range.end = 9

p.legend.orientation = "horizontal"

p.legend.location = "top_center"



show(p)
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
factors = ["A", "B", "C"]

x = ["A", "A", "A", "B", "B", "B", "C",  "C",  "C"]

y = ["A", "B", "C",  "A", "B", "C",  "A", "B", "C"]



colors = [

    "#0B486B", "#79BD9A", "#CFF09E",

    "#79BD9A", "#0B486B", "#79BD9A",

    "#CFF09E", "#79BD9A", "#0B486B"

]



hm = figure(title="Heatmap", tools="hover", toolbar_location=None,

            x_range=factors, y_range=factors)



hm.rect(x, y, color=colors, width=1, height=1)



show(hm)
groups = ['A', 'B', 'C', 'D', 'E', 'F']

years = ["1", "2", "3"]

colors = ["#EC4067", "#A01A7D", "#311847"]



data = {'groups' : groups,

        '1'   : [2, 1, 4, 3, 2, 4],

        '2'   : [5, 3, 4, 2, 4, 6],

        '3'   : [3, 2, 4, 4, 5, 3]}



p = figure(x_range=groups, plot_height=250, title="Bar(stacked)",

           toolbar_location=None, tools="hover", tooltips="$name @groups: @$name")



p.vbar_stack(years, x='groups', width=0.9, color=colors, source=data,

             legend_label=years)



p.y_range.start = 0

p.x_range.range_padding = 0.1

p.xgrid.grid_line_color = None

p.axis.minor_tick_line_color = None

p.outline_line_color = None

p.legend.location = "top_left"

p.legend.orientation = "horizontal"



show(p)
def make_plot(title, hist, edges, x, pdf):

    p = figure(title=title, tools='', background_fill_color="#fafafa")

    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],

           fill_color="navy", line_color="white", alpha=0.5)

    p.line(x, pdf, line_color="#ff8888", line_width=4, alpha=0.7, legend_label="PDF")

    #p.line(x, cdf, line_color="orange", line_width=2, alpha=0.7, legend_label="CDF")



    p.y_range.start = 0

    p.legend.location = "center_right"

    p.legend.background_fill_color = "#fefefe"

    p.xaxis.axis_label = 'x'

    p.yaxis.axis_label = 'Pr(x)'

    p.grid.grid_line_color="white"

    return p



# Normal Distribution

mu, sigma = 0, 0.5

measured = np.random.normal(mu, sigma, 1000)

hist, edges = np.histogram(measured, density=True, bins=50)

x = np.linspace(-2, 2, 1000)

pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))

#cdf = (1+scipy.special.erf((x-mu)/np.sqrt(2*sigma**2)))/2



p1 = make_plot("Normal Distribution", hist, edges, x, pdf)





show(p1)
N = 5

df = pd.DataFrame(np.random.randint(10, 70, size=(15, N))).add_prefix('y')



p = figure(x_range=(0, len(df)-3), y_range=(0, 800))

p.grid.minor_grid_line_color = '#eeeeee'



names = ["y%d" % i for i in range(N)]

p.varea_stack(stackers=names, x='index', color=brewer['Spectral'][N], legend_label=names, source=df)



# reverse the legend entries to match the stacked order

p.legend.items.reverse()



show(p)
# generate some synthetic time series for four different categories

cats = list("abcd")

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



show(p)
x = np.linspace(0.1, 5, 80)



p = figure(title="log axis example", y_axis_type="log",

           x_range=(0, 5), y_range=(0.001, 10**22),

           background_fill_color="#fafafa")



p.line(x, np.sqrt(x), legend_label="y=sqrt(x)",

       line_color="tomato", line_dash="dashed")



p.line(x, x**x, legend_label="y=x^x",

       line_dash="dotted", line_color="indigo", line_width=2)



p.line(x, 10**(x**2), legend_label="y=10^(x^2)",

       line_color="coral", line_dash="dotdash", line_width=2)



p.legend.location = "top_left"



show(p)
#Libraries

import plotly.graph_objects as go

import plotly.express as px

import plotly.figure_factory as ff
df = px.data.iris()  #Iris Dataset

fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",

                 size='petal_length', hover_data=['petal_width'])

fig.show()
df = px.data.gapminder().query("continent == 'Oceania'")

fig = px.line(df, x='year', y='lifeExp', color='country')

fig.show()
colors = ['lightsteelblue',]*5

fig = go.Figure(data=[go.Bar(

    x=['A', 'B', 'C',

       'D', 'E'],

    y=[20, 14, 23, 25, 22],

    marker_color=colors

)])

fig.update_layout(title_text='Bar Plot')
groups=['A', 'B', 'C']

col1 = ['lightsteelblue',]*5

col2 = ['floralwhite',] * 5



fig = go.Figure(data=[

    go.Bar(name='XX', x=groups, y=[20, 14, 23], marker_color=col1),

    go.Bar(name='YY', x=groups, y=[12, 18, 29], marker_color=col2),

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.show()
groups=['A', 'B', 'C']

col1 = ['lightsteelblue',]*5

col2 = ['floralwhite',] * 5



fig = go.Figure(data=[

    go.Bar(name='XX', x=groups, y=[20, 14, 23], marker_color=col1),

    go.Bar(name='YY', x=groups, y=[12, 18, 29], marker_color=col2)

])

# Change the bar mode

fig.update_layout(barmode='stack')

fig.show()
df = px.data.tips()

fig = px.pie(df, values='tip', names='day', color_discrete_sequence=px.colors.sequential.BuGn)

fig.show()
df = px.data.gapminder()



fig = px.scatter(df.query("year==2007"), x="gdpPercap", y="lifeExp",

                 size="pop", color="continent",

                 hover_name="country", log_x=True, size_max=60)

fig.show()
df = px.data.iris()

fig = px.box(df, x="sepal_width", y="sepal_length")

fig.show()
df = px.data.iris()

fig = px.histogram(df, x="sepal_length", 

                   title = 'Histogram',opacity=0.8,

                   color = 'species', nbins = 30

                   )

fig.show()
df.columns
df = px.data.iris()



# Add histogram data

x1 = df.sepal_length

x2 = df.sepal_width

x3 = df.petal_length

x4 = df.petal_width



# Group data together

hist_data = [x1, x2, x3, x4]



group_labels = ['Sepal-Length', 'Sepal-Width', 'Petal-Length', 'Petal-Width']



# Create distplot with custom bin_size

fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)

fig.show()
#Libraries

import seaborn as sns
sns.set(style="white", context="talk")

rs = np.random.RandomState(8)



f, (ax1) = plt.subplots(1, 1, figsize=(7, 5), sharex=True)



x1 = np.array(list("ABCDEFGHIJ"))

y1 = np.arange(1, 11)



y = rs.choice(y1, len(y1), replace=False)

sns.barplot(x=x1, y=y, palette="deep", ax=ax1)

ax1.axhline(0, color="k", clip_on=False)

ax1.set_ylabel("Qualitative")



sns.despine(bottom=True)

plt.setp(f.axes, yticks=[])

plt.tight_layout(h_pad=2)

sns.set(style="whitegrid")



# Load the example Titanic dataset

titanic = sns.load_dataset("titanic")



# Draw a nested barplot to show survival for class and sex

g = sns.catplot(x="class", y="survived", hue="sex", data=titanic,

                height=6, kind="bar", palette="pastel")

g.despine(left=True)

g.set_ylabels("survival probability")
sns.set(style="whitegrid", palette="pastel")



# Load the example iris dataset

iris = sns.load_dataset("iris")



# "Melt" the dataset to "long-form" or "tidy" representation

iris = pd.melt(iris, "species", var_name="measurement")



# Draw a categorical scatterplot to show each observation

sns.swarmplot(x="measurement", y="value", hue="species",

              palette=["r", "g", "b"], data=iris)
sns.set(style="white", palette="muted", color_codes=True)

rs = np.random.RandomState(10)



# Set up the matplotlib figure

f, (ax1,ax2) = plt.subplots(1, 2, figsize=(8, 8), sharex=False)

sns.despine(left=True)



# Generate a random univariate dataset

d = rs.normal(size=100)



# Plot a simple histogram with binsize determined automatically

sns.distplot(d, kde=False, color="b", ax = ax1)



# Plot a histogram and kernel density estimate

sns.distplot(d, color="m", ax = ax2)



plt.setp(axes, yticks=[])

plt.tight_layout()
sns.set(style="darkgrid")



# Load an example dataset with long-form data

fmri = sns.load_dataset("fmri")



# Plot the responses for different events and regions

sns.lineplot(x="timepoint", y="signal",

             hue="region", style="event",

             data=fmri)
sns.set(style="ticks", palette="pastel")



# Load the example tips dataset

tips = sns.load_dataset("tips")



# Draw a nested boxplot to show bills by day and time

sns.boxplot(x="day", y="total_bill",

            hue="smoker", palette=["m", "g"],

            data=tips)

sns.despine(offset=10, trim=True)
sns.set(style="ticks", palette="pastel")



# Load the example flights dataset and convert to long-form

flights_long = sns.load_dataset("flights")

flights = flights_long.pivot("month", "year", "passengers")



# Draw a heatmap with the numeric values in each cell

f, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(flights, annot=True, fmt="d", linewidths=.5, ax=ax, cmap='pink_r')
labels = ['G1', 'G2', 'G3', 'G4', 'G5']

men_means = [20, 35, 30, 35, 27]

women_means = [25, 32, 34, 20, 25]

men_std = [2, 3, 4, 1, 2]

women_std = [3, 5, 2, 3, 3]

width = 0.35       # the width of the bars: can also be len(x) sequence



fig, ax = plt.subplots(figsize=(8,8))



ax.bar(labels, men_means, width, yerr=men_std, label='Men')

ax.bar(labels, women_means, width, yerr=women_std, bottom=men_means,

       label='Women')



ax.set_ylabel('Scores')

ax.set_title('Scores by group and gender')

ax.legend()



plt.show()
np.random.seed(19680801)





N = 100

r0 = 0.6

x = 0.9 * np.random.rand(N)

y = 0.9 * np.random.rand(N)

area = (20 * np.random.rand(N))**2  # 0 to 10 point radii

c = np.sqrt(area)

r = np.sqrt(x ** 2 + y ** 2)

area1 = np.ma.masked_where(r < r0, area)

area2 = np.ma.masked_where(r >= r0, area)

plt.scatter(x, y, s=area1, marker='.', c=c)

plt.scatter(x, y, s=area2, marker='x', c=c)

# Show the boundary between the regions:

theta = np.arange(0, np.pi / 2, 0.01)

plt.plot(r0 * np.cos(theta), r0 * np.sin(theta))



plt.show()
#Libraries

import altair as alt
source = pd.DataFrame({

    'a': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],

    'b': [28, 55, 43, 91, 81, 53, 19, 87, 52]

})



alt.Chart(source).mark_bar().encode(

    x='a',

    y='b'

)
x = np.arange(100)

source = pd.DataFrame({

  'x': x,

  'f(x)': np.sin(x / 5)

})



alt.Chart(source).mark_line().encode(

    x='x',

    y='f(x)'

)