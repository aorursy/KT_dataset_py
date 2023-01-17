import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/lgcovid19hotp/lg-covid19-hotp-article-info.csv', encoding = "ISO-8859-1")
df.head()
df.describe()
years = df['year']
fig = plt.figure(figsize=(10,8))
plt.xlabel('year')
plt.ylabel('#articles (logscale)')
years.hist(range=(1950, 2020), bins=70, log=True)
plt.show()
from scipy.io import mmread
A = mmread('/kaggle/input/lgcovid19hotp/lg-covid19-hotp-literature-graph.mtx')
inCitations = np.asarray( A.sum(axis=1) ).squeeze()
sortedIndices = np.argsort(inCitations)
sortedIndices = sortedIndices[::-1]   # Default sorting is ascending so convert it to descenting
top = df.iloc[sortedIndices[0:50]]
titles = top['title'].tolist()
topInCitations = inCitations[sortedIndices[0:50]].squeeze()
# Get unique elements and unique counts
u, uniqueCounts = np.unique(topInCitations, return_counts=True)
uniqueCounts = uniqueCounts[::-1]
u = u[::-1]

# 50x1 vector to hold each rank
ranks = np.zeros((len(topInCitations), 1)).squeeze()

i = 0  # Loop iterator
last = 0  # Index to last position of ranks array
rank = 1  # Current rank
while i<len(u):
    width = uniqueCounts[i]  # Number of articles with the same rank at each repetition
    ranks[last:last+width] = rank
    rank += 1
    i += 1
    last += width
# Bokeh Libraries
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.models import HoverTool
from bokeh.models import ColumnDataSource

# Output the visualization directly in the notebook
output_notebook()

# Create a figure with no toolbar
fig = figure(y_axis_label='#citations (logscale)', y_axis_type='log',
             x_axis_label='rank',
             plot_height=600, plot_width=800,
             x_range=(0, 50), y_range=(250, 1000),
             toolbar_location=None)

data={
    'x': ranks,
    'y': topInCitations,
    'title': titles
}
source = ColumnDataSource(data=data)

# Draw the coordinates as circles
fig.circle(x='x', y='y', source=source,
           color='blue', size=10, alpha=0.5)

# Format the tooltip
tooltips = [
            ('x', '@x'),
            ('y', '@y'),
            ('title', '@title')
           ]


# Add the HoverTool to the figure
fig.add_tools(HoverTool(tooltips=tooltips))

# Show plot
show(fig)
# Keep only generation 1
gen1 = df[df['generation'] == 1]
gen1Indices = gen1.index.tolist()
gen1Citations = inCitations[gen1Indices].squeeze()
gen1Citations.sort()
gen1Citations = gen1Citations[::-1]

gen1titles = df.iloc[gen1Indices]
gen1titles = gen1titles['title'].tolist()

# Get unique elements and unique counts
u, uniqueCounts = np.unique(gen1Citations, return_counts=True)
uniqueCounts = uniqueCounts[::-1]
u = u[::-1]

# vector to hold each rank
ranks = np.zeros((len(gen1Citations), 1)).squeeze()

i = 0  # Loop iterator
last = 0  # Index to last position of ranks array
rank = 1  # Current rank
while i<len(u):
    width = uniqueCounts[i]  # Number of articles with the same rank at each repetition
    ranks[last:last+width] = rank
    rank += 1
    i += 1
    last += width
# Output the visualization directly in the notebook
output_notebook()

# Create a figure with no toolbar
fig = figure(y_axis_type='log',
             y_axis_label='#citations (log scale)', x_axis_label='rank',
             plot_height=800, plot_width=1000,
             x_range=(0, 150), y_range=(0, 1000),
             toolbar_location=None)

data={
    'x': ranks,
    'y': gen1Citations,
    'title': gen1titles
}
source = ColumnDataSource(data=data)

# Draw the coordinates as circles
fig.circle(x='x', y='y', source=source,
           color='blue', size=8, alpha=0.5)

custom_hover = HoverTool()

custom_hover.tooltips = """
    <style>
        .bk-tooltip>div:not(:first-child) {display:none;}
    </style>

    <b>X: </b> @x <br>
    <b>Y: </b> @y <br>
    <b>Title: </b> @title 
"""

# Add the HoverTool to the figure
fig.add_tools(custom_hover)

# Show plot
show(fig)
