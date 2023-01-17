#import libraries
import pandas as pd
pd.set_option("display.max_columns", 200)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime as dt
import yellowbrick as yb
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
import pandas_profiling

# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/use-of-force.csv')

data.head()
data['Incident_Type'] = data['Incident_Type'].map(lambda x: x.split(' ')[1].split("-")[0].strip())
data['Occured_date_time'] = pd.to_datetime(data['Occured_date_time'], format= '%Y-%m-%dT%H:%M:%S')
data['Officer_ID'] = data['Officer_ID'].astype(str)
data['Subject_ID'] = data['Subject_ID'].astype(str)
date_group = data.groupby('Occured_date_time')['Incident_Num'].count()

#Basic line plot of uses of force over time
x0 = date_group.index
y0 = date_group.values

# specify that we want a scatter plot with, with date on the x axis and meet on the y axis
trace0 = [go.Scatter(x=x0, y=y0, mode='lines')]

# specify the layout of our figure
layout = dict(title = "Reported Uses of Force by Seattle PD Over Time",
              xaxis= dict(title= 'Date',ticklen= 15,zeroline= False))

# create and show our figure
fig = dict(data = trace0, layout = layout)
iplot(fig)
precinct = data.pivot_table(index='Precinct', columns='Incident_Type', values='Incident_Num', aggfunc='count')

x0 = ['Incident Type 1', 'Incident Type 2', 'Incident Type 3']


y0 = list(precinct.loc['-'].values)
y1 = list(precinct.loc['E'].values)
y2 = list(precinct.loc['N'].values)
y3 = list(precinct.loc['S'].values)
y4 = list(precinct.loc['SW'].values)
y5 = list(precinct.loc['W'].values)
y6 = list(precinct.loc['X'].values)


trace0 = go.Bar(x=x0, y=y0, name= 'Precinct -',
                marker=dict(color='rgb(60, 71, 216)') 
                 )
trace1 = go.Bar(x=x0, y=y1, name= 'Precinct E',
                marker=dict(color='rgb(109, 130, 109)')
                 )
trace2 = go.Bar(x=x0, y=y2, name= 'Precinct N', 
                 marker=dict(color='rgb(96, 44, 96)')
               )
trace3 = go.Bar(x=x0, y=y3, name= 'Precinct S',
                marker=dict(color='rgb(45, 91, 39)')
                 )         
trace4 = go.Bar(x=x0, y=y4, name= 'Precinct SW',
                marker=dict(color='rgb(204, 8, 31)')
                 )
trace5 = go.Bar(x=x0, y=y5, name= 'Precinct W',
                marker=dict(color='rgb(162, 201, 46)')
                 )          
trace6 = go.Bar(x=x0, y=y6, name= 'Precinct X',
                marker=dict(color='rgb(136, 149, 216)')
                 )
          
          
graph_data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6]
layout = go.Layout(barmode='group', title= "Frequency of Force Per Precinct")
fig = go.Figure(data=graph_data, layout=layout)
iplot(fig)
