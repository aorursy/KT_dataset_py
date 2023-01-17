### Import Statements
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import geopandas

import plotly.plotly as py
import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

import os
print(os.listdir("../input"))


proc_data = pd.read_csv("../input/procurement-notices.csv")
proc_data.head()
proc_data['deadline_date'] = pd.to_datetime(proc_data['Deadline Date'], format='%Y-%m-%d')
current_calls = proc_data.loc[proc_data['deadline_date'] > pd.datetime.today()]
current_calls.head()
calls_by_country = current_calls.groupby('Country Name').size().reset_index(name='count')
calls_by_country['name'] = calls_by_country['Country Name']
calls_by_country.head()
### read in geopandas dataset for worldmap
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
world.head()
merged_calls = world.merge(calls_by_country,on='name',how='outer')
merged_calls['count'] = merged_calls['count'].fillna(value=0)
merged_calls.head()
merged_calls.plot(column='count',figsize=(15,20),cmap='YlOrRd',edgecolor='black')
## distribution by deadline_date

due_dates = current_calls.groupby('deadline_date').size().reset_index(name='count')
due_dates.head()                                          
due_dates.plot(kind='line',x='deadline_date',y='count',figsize=(10,5))
### Distribution by due dates
### Using due_dates dataframe from earlier cells
data = [go.Scatter(x=due_dates['deadline_date'],y=due_dates['count'])]

layout = dict(title="Amount of Calls out by Deadline",
              xaxis=dict(title='Date',ticklen=5,zeroline=False))
                

fig = dict(data=data, layout=layout)
iplot(fig)
### Distribution by country
### using merged_calls dataframe instead of calls_by_country so that I can render the entire map
### using the geopandas world dataset
data = [ dict(
        type = 'choropleth',
        locations = merged_calls['name'],
        z = merged_calls['count'],
        text = merged_calls['name'],
        locationmode = 'country names',
)]
          

layout = dict(
    title = 'Number of Bids Tendered by Country',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot(fig)
