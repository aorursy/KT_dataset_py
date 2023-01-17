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
print("number of current calls:")
print((df[df['deadline_date'] > datetime.now()] | df[df['deadline_date'].isna()]).count()['deadline_date'])

current_calls = df[df['deadline_date'] > datetime.now()]
    



import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
# dist. of due dates
due_dates = df[df['deadline_date'] > datetime.now()].groupby('deadline_date').count().dropna()
due_dates.rename(columns={'id': 'n'}, inplace=True)
# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis
data = [go.Scatter(x=due_dates.index, y=due_dates.n)]

# specify the layout of our figure
layout = dict(title = "Number of  Deadline per Month",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)

# dist. of due dates
#due_dates = df[df['deadline_date'] > datetime.now()].groupby('deadline_date').count().dropna()
# plot number of bids due per date
#due_dates.rename(columns={'id': 'n'}, inplace=True)
#plt.plot(due_dates.index, due_dates['n'])
# specify what we want our map to look like
calls_by_country = pd.DataFrame(current_calls.groupby('country_name').count()['id'])
calls_by_country.rename(columns={'id': 'number_of_bids'}, inplace=True)
data = [ dict(
        type='choropleth',
        autocolorscale = False,
        locations = calls_by_country.index,
        z = calls_by_country['number_of_bids'],
        locationmode = 'country names'
       ) ]


# chart information
layout = dict(
        title = 'Number of Bids  per country',
        geo = dict(
                  showframe = False,
                  showcoastlines = False,
                  projection = dict(
                  type = 'mercator'
                                    )
                   )
              )
   
# actually show our figure
fig = dict( data=data, layout=layout )
iplot( fig, filename='d3-cloropleth-map' )
