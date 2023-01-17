# imports
import numpy as np
import pandas as pd
# read in the data
inmates = pd.read_csv('../input/daily-inmates-in-custody.csv')
# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
# re-create graph of age
data = [go.Histogram(x=inmates.AGE)]

# specify the layout of our figure
layout = dict(title = 'Counts of Inmates by Age',
              xaxis= dict(title= 'Age',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)
# re-create graph of age
data = [go.Bar(
            x=inmates.RACE.unique(),
            y=inmates.RACE.value_counts()
    )]
layout = dict(title = 'Counts of Inmates by Race',
              xaxis= dict(title= 'Race'))
fig = dict(data = data, layout = layout)
iplot(fig)
# re-create YEARS_IN
inmates['ADMITTED_DT'] = pd.to_datetime(inmates['ADMITTED_DT'])
inmates['YEARS_IN'] = (pd.to_datetime('today') - inmates['ADMITTED_DT']) / pd.Timedelta('365.25 days')
# re-create graph of YEARS_IN
data = [go.Histogram(x=inmates.YEARS_IN)]
layout = dict(title = 'Counts of Inmates by Years Incarcerated',
              xaxis= dict(title= 'Years Incarcerated',ticklen= 5,zeroline= False))
fig = dict(data = data, layout = layout)
iplot(fig)