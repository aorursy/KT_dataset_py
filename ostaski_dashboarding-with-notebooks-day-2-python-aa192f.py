import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# read in data
data = pd.read_csv("../input/meets.csv")


## Hacky data munging
# parse dates
data['Date'] = pd.to_datetime(data['Date'], format = "%Y-%m-%d")

# count of meets per month
meets_by_month = data['Date'].groupby([data.Date.dt.year, data.Date.dt.month]).agg('count') 

# convert to dataframe
meets_by_month = meets_by_month.to_frame()

# move date month from index to column
meets_by_month['date'] = meets_by_month.index

# rename column
meets_by_month = meets_by_month.rename(columns={meets_by_month.columns[0]:"meets"})

# re-parse dates
meets_by_month['date'] = pd.to_datetime(meets_by_month['date'], format="(%Y, %m)")

# remove index
meets_by_month = meets_by_month.reset_index(drop=True)

# get month of meet
meets_by_month['month'] = meets_by_month.date.dt.month

# repeat to get number of meets per state
meet_by_state = data['MeetState'].value_counts().to_frame()
meet_by_state['state'] = meet_by_state.index
meet_by_state = meet_by_state.rename(columns={meet_by_state.columns[0]:"meets"})
# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis
data = [go.Scatter(x=meets_by_month.date, y=meets_by_month.meets)]

# specify the layout of our figure
layout = dict(title = "Number of Powerlifting Meets per Month",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)
# specify what we want our map to look like
data = [ dict(
        type='choropleth',
        autocolorscale = False,
        locations = meet_by_state['state'],
        z = meet_by_state['meets'],
        locationmode = 'USA-states'
       ) ]

# chart information
layout = dict(
        title = 'Number of Powerlifting Meets per State',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
   
# actually show our figure
fig = dict( data=data, layout=layout )
iplot( fig, filename='d3-cloropleth-map' )
# this cell does not want to hide input, this is my 4th attempt