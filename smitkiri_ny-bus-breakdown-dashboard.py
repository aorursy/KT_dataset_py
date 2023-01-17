import numpy as np 
import pandas as pd 
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
import plotly.graph_objs as go


data = pd.read_csv('../input/bus-breakdown-and-delays.csv', low_memory=False)


data['Occurred_On'] = pd.to_datetime(data['Occurred_On'], format = '%Y-%m-%d')

#count of breakdowns per month
breakdown_by_month = data['Occurred_On'].groupby([data['Occurred_On'].dt.year, data['Occurred_On'].dt.month]).agg('count')

#convert to dataframe
breakdown_by_month = breakdown_by_month.to_frame()

#move date month from index to column
breakdown_by_month['Occured_On'] = breakdown_by_month.index

#renaming column
breakdown_by_month = breakdown_by_month.rename(columns={breakdown_by_month.columns[0]: 'breakdowns', breakdown_by_month.columns[1]: 'date'})

#re-parse dates
breakdown_by_month['date'] = pd.to_datetime(breakdown_by_month['date'], format = '(%Y, %m)')

#remove 2020
breakdown_by_month.drop(2020, inplace=True)

#remove index
breakdown_by_month = breakdown_by_month.reset_index(drop=True)

#get month of breakdowns
breakdown_by_month['month'] = breakdown_by_month['date'].dt.month

#Breakdown reasons
breakdown_reason = data['Reason'].value_counts().to_frame()
breakdown_reason['reason'] = breakdown_reason.index
breakdown_reason = breakdown_reason.reset_index(drop = True)
breakdown_reason = breakdown_reason.rename(columns = {breakdown_reason.columns[0]: 'count'})

#Get values by boroughs
breakdown_by_boro = data['Boro'].value_counts().to_frame()
breakdown_by_boro['boro'] = breakdown_by_boro.index
breakdown_by_boro = breakdown_by_boro.reset_index(drop=True)
breakdown_by_boro = breakdown_by_boro.rename(columns={breakdown_by_boro.columns[0]: 'Breakdowns'})
#Breakdowns per month
d = [go.Scatter(x = breakdown_by_month['date'], y = breakdown_by_month['breakdowns'])]

layout = dict(title = "Number of breakdowns per month",
             xaxis = dict(title = 'Date', ticklen = 10, zeroline=False))

fig = dict(data = d, layout = layout)
iplot(fig)
#Breakdown by Boroughs
d = [go.Bar(
    x = list(breakdown_by_boro['Breakdowns']),
    y = list(breakdown_by_boro['boro']),
    orientation = 'h')]

layout = go.Layout(title = "Breakdowns by Boroughs")

fig = go.Figure(data = d, layout=layout)
iplot(fig)
#Breakdown reasons
trace = [go.Pie(
    values = list(breakdown_reason['count']),
    labels = list(breakdown_reason['reason']))]

layout = go.Layout(title = "Breakdown reasons")

fig = go.Figure(data = trace, layout=layout)
iplot(fig)