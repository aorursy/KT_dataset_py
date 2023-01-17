import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import janitor
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from datetime import datetime

# Any results you write to the current directory are saved as output.

proc_data = pd.read_csv('../input/procurement-notices.csv').clean_names()
proc_data['deadline_date'] = pd.to_datetime(proc_data['deadline_date'])
proc_data.info()
# number of ccalls currently out
# cells with NA deadline are currently out, not dealt with here
current_calls = proc_data[(proc_data['deadline_date'] > datetime.now())]
print(current_calls.count())
calls_by_country = current_calls.groupby(['country_code', 'country_name'])['id'].count().reset_index()
# Converting country codes from ISO Alpha2 to ISO Alpha3
# Then initializing all other countries with a value of 0 so they show up on the map

countrymap = pd.read_csv('https://raw.githubusercontent.com/gsnaveen/plotly-worldmap-mapping-2letter-CountryCode-to-3letter-country-code/master/countryMap.txt', sep='\t')
calls_by_country = calls_by_country.merge(countrymap,how='right',left_on=['country_code'],right_on=['2let'])
calls_by_country = calls_by_country.fill_empty(columns='id', value=0)
# dist. by country

import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode()
from IPython.display import HTML

data = [ dict(
        type = 'choropleth',
        locations = calls_by_country['3let'],
        z = calls_by_country['id'],
        text = calls_by_country['Countrylet'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Current Calls<br>Out'),
      ) ]

layout = dict(
    title = 'Current Calls Around the World',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='d3-world-map' )
import plotly.graph_objs as go

# Create a trace
trace = go.Scatter(
    x = current_calls.groupby(pd.Grouper(key='deadline_date',freq='D')).count().index,
    y = current_calls.groupby(pd.Grouper(key='deadline_date',freq='D')).count()['id'],
    mode = 'lines'
)



data = [trace]

layout = dict(title = 'Due Dates by Month',
              yaxis = dict(title = 'n'),
              xaxis = dict(title = 'Date')
             )
fig = dict(data=data, layout=layout)
# Plot and embed in ipython notebook!
iplot(fig, filename='basic-scatter')