import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from IPython.core import display as ICD
init_notebook_mode(connected=True)
tickets = pd.read_csv('../input/parking-citations.csv')
tickets = tickets.sort_values(by='Issue Date')
tickets2 = tickets
tickets2['oldDate'] = tickets2['Issue Date']
tickets2['newDate'] = pd.DatetimeIndex(tickets2.oldDate).normalize()
dateCounts = tickets2['newDate'].value_counts()
dateCountsDf = pd.DataFrame(dateCounts)
dateCountsDf['date2'] = dateCountsDf.index
dateCountsDf = dateCountsDf.sort_values(by='date2')
dateCountsDf = dateCountsDf.resample('W', on='date2').sum()
dateCountsDf = dateCountsDf.reset_index(level='date2')
dateCountsDf = dateCountsDf[(dateCountsDf['date2'].dt.year > 2014)] 
df = dateCountsDf
trace1 = go.Scatter(
                    x = df.date2,
                    y = df.newDate,
                    mode = "lines+markers",
                    name = "gettingStarted",
                    marker = dict(color = 'rgba(0, 0, 255, 0.8)'),)
data = [trace1]
layout = dict(title = 'Weekly Count of Parking Tickets in Los Angeles',
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),yaxis= dict(title= '# of Parking Tickets',ticklen= 5,zeroline= False),legend=dict(orientation= "h",x=0, y= 1.13)
             )
fig = dict(data = data, layout = layout)
iplot(fig)