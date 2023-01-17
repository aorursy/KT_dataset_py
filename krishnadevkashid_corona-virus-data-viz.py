import pandas as pd

patient = pd.read_csv("../input/coronavirusdataset/patient.csv")

route = pd.read_csv("../input/coronavirusdataset/route.csv", date_parser = 'date')

time = pd.read_csv("../input/coronavirusdataset/time.csv")
# plotly standard imports

import plotly.graph_objs as go

import chart_studio.plotly as py



# Cufflinks wrapper on plotly

import cufflinks



# Data science imports

import pandas as pd

import numpy as np



# Options for pandas

pd.options.display.max_columns = 30



# Display all cell outputs

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'
from plotly.offline import iplot

cufflinks.go_offline()



# Set global theme

cufflinks.set_config_file(world_readable=True, theme='ggplot')
patient['birth_year'].iplot(

    kind='hist',

    bins=30,

    xTitle='Price',

    linecolor='black',

    yTitle='count',

    title='Infection by birth year')
patient.iloc[:,1:].iplot(

    kind='scatter',

    bins=30,

    xTitle='Price',

    linecolor='black',

    yTitle='count',

    title='Overlapping scatter plot')
time[['acc_confirmed','acc_deceased']].iplot(

    kind='spread', title='Death vs infection spread Plot')
time.iplot(

    x='new_test',

    y='new_confirmed',

    categories='date',

    xTitle='Patient tested',

    yTitle='+ve Cases',

    title='Day wise infection rate')