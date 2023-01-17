import pandas as pd

from bokeh.plotting import figure, show, reset_output

from bokeh.io import output_notebook, push_notebook

from bokeh.models import ColumnDataSource, HoverTool

from ipywidgets import interact
# Loading and briefly processing the data

df = pd.read_csv('../input/Pokemon.csv')



# Renaming columns for clarity

columns = df.columns.tolist()

columns[0] = 'id'

for i, col in enumerate(columns):

    columns[i] = col.replace(' ','') # remove spaces

df.columns = columns



# Selecting columns to consider

cols = ['HP', 'Attack', 'Defense', 'Sp.Atk', 'Sp.Def', 'Speed', 'Total']
output_notebook()
# Preparing plot

df['x'] = df['HP']

df['y'] = df['Attack']

source = ColumnDataSource(data=df)



p = figure(title='Pokemon Data')

r = p.scatter('x', 'y', source=source, size=8)

p.xaxis.axis_label = 'HP'

p.yaxis.axis_label = 'Attack'



# Setting on mouse-over tooltips

p.add_tools(HoverTool(tooltips=[

    ("Name", "@Name"),

    ("Type", "@Type1 - @Type2"),

    ("Total", "@Total")

]))
# Function to update x and y values

def update(setx, sety):

    r.data_source.data['x'] = df[setx].tolist()

    r.data_source.data['y'] = df[sety].tolist()

    p.xaxis.axis_label = setx

    p.yaxis.axis_label = sety

    push_notebook()
# Display the plot, the next cell will add the interaction

show(p)
interact(update, setx=cols, sety=cols)