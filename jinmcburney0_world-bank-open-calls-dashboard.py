# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

from datetime import datetime



df = pd.read_csv('../input/procurement-notices.csv')

df.columns = [col.lower().replace(' ', '_') for col in df.columns]

# df.head()



# number of calls currently out

# cells with NA deadline are currently out

df['deadline_date'] = pd.to_datetime(df['deadline_date'])

#df['deadline_date'].dtype

print("Number of current calls:")

print((df[df['deadline_date'] > datetime.now()] | df[df['deadline_date'].isna()]).count()['deadline_date'])



current_calls = df[df['deadline_date'] > datetime.now()]
# import plotly

import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools # for sub figures



# these two lines allow your code to show up in a notebook!

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()



# distribution of due dates

due_dates = df[df['deadline_date'] > datetime.now()].groupby('deadline_date').count().dropna()

due_dates.rename(columns={'id': 'n'}, inplace=True)

#due_dates.head()



# specify plot type and x and y-axis

data_dates = [go.Scatter(x=due_dates.index, y=due_dates.n)]



# specify the layout of the figure

layout_dates = dict(title='Number of Calls Due (by Date)', 

                   xaxis= dict(title='Date', ticklen=5, zeroline=False))



# create and show figure

fig = dict(data=data_dates, layout=layout_dates)

iplot(fig)

# distribution by country

calls_by_country = pd.DataFrame(current_calls.groupby('country_name').count()['id'])

calls_by_country.rename(columns={'id': 'number_of_bids'}, inplace=True)



# specify what the map will look like

data_map = [dict(

                type='choropleth',

                autocolorscale=False,

                locations=calls_by_country.index,

                z=calls_by_country['number_of_bids'],

                locationmode='country names'

            )]



# specify chart information

layout_map = dict(

                title='Number of Open Calls by Country',

                geo=dict(showframe=True, showcoastlines=True)

                )



# show map

fig = dict(data=data_map, layout=layout_map)

iplot(fig, filename='d3-choropleth-map')