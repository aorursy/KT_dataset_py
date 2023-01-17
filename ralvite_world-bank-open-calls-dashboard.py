# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
from datetime import datetime


df = pd.read_csv('../input/procurement-notices.csv')
df.columns = [col.lower().replace(' ', '_') for col in df.columns]

# number of calls currently out
# cells with NA deadline are currently out
df['deadline_date'] = pd.to_datetime(df['deadline_date'])
print("Number of current calls:")
print((df[df['deadline_date'] > datetime.now()] | df[df['deadline_date'].isna()]).count()['deadline_date'])

current_calls = df[df['deadline_date'] > datetime.now()]
current_calls.head()
# current_calls.groupby(current_calls['procurement_type']).size()
# import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools # for sub figures

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# dist. of due dates
due_dates = df[df['deadline_date'] > datetime.now()].groupby('deadline_date').count().dropna()
due_dates.rename(columns={'id': 'n'}, inplace=True)

# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis
data_dates = [go.Scatter(x=due_dates.index, y=due_dates.n)]

# specify the layout of our figure
layout_dates = dict(title = "Number of Calls Due (by Date)",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data_dates, layout = layout_dates)
iplot(fig)

# dist. by country
calls_by_country = pd.DataFrame(current_calls.groupby('country_name').count()['id'])
calls_by_country.rename(columns={'id': 'number_of_bids'}, inplace=True)

# specify what we want our map to look like
data_map = [dict(
        type='choropleth',
        autocolorscale = False,
        locations = calls_by_country.index,
        z = calls_by_country['number_of_bids'],
        locationmode = 'country names'
       ) ]

# chart information
layout_map = dict(
        title = 'Number of Open Calls by Country',
        geo = dict(
                showframe = False,
                showcoastlines = True
            )
        )
   
# actually show our figure
fig = dict( data=data_map, layout=layout_map )
iplot( fig, filename='d3-cloropleth-map' )