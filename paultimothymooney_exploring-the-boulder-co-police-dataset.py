# **Police Calls in Boulder, CO: Dashboard (Updated Daily)**
# * Note that the dataset updates daily: ([Link #1](https://www.kaggle.com/product-feedback/75341#449911)), ([Link #2](https://i.imgur.com/d0poO80.png))
# * And likewise the kernel updates daily as well: ([Link #3](https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-4))
import numpy as np
import pandas as pd 
import os
import time
from datetime import date, datetime
from dateutil import parser
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

bpd_call_log_data_dictionary = pd.read_csv('../input/bpd_call_log_data_dictionary.csv')
BPD_Call_Log = pd.read_csv('../input/BPD_Call_Log.csv')
BPD_Call_Log['Date'] = np.where(BPD_Call_Log['ResponseDate']=='1', BPD_Call_Log['Incident_ID'], BPD_Call_Log['ResponseDate'])

#BPD_Call_Log2 = BPD_Call_Log[:-2]
BPD_Call_Log2 = BPD_Call_Log.loc[-2:]

BPD_Call_Log2['Date'] =  pd.to_datetime(BPD_Call_Log2['Date'],errors='coerce')
BPD_Call_Log2['oldDate'] = BPD_Call_Log2['Date']
BPD_Call_Log2['newDate'] = pd.DatetimeIndex(BPD_Call_Log2.oldDate).normalize()
dateCounts = BPD_Call_Log2['newDate'].value_counts()
dateCountsDf = pd.DataFrame(dateCounts)
dateCountsDf['date2'] = dateCountsDf.index
dateCountsDf = dateCountsDf.sort_values(by='date2')
dateCountsDf = dateCountsDf.resample('D', on='date2').sum()
dateCountsDf = dateCountsDf.reset_index(level='date2')
dateCountsDf = dateCountsDf[(dateCountsDf['date2'].dt.year > 2018)] 



df = dateCountsDf
trace1 = go.Scatter(
                    x = df.date2,
                    y = df.newDate,
                    mode = "lines+markers",
                    name = "gettingStarted",
                    marker = dict(color = 'rgba(0, 0, 255, 0.8)'),)
data = [trace1]
layout = dict(title = 'Daily Count of Police Calls in Boulder, CO',
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),yaxis= dict(title= '# of Police Calls',ticklen= 5,zeroline= False),legend=dict(orientation= "h",x=0, y= 1.13)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
counts = BPD_Call_Log['CaseNumber'].value_counts()
counts = pd.DataFrame(counts)[0:10]
trace1 = go.Bar(
                x = counts.index,
                y = counts.CaseNumber,
                name = "Kaggle",
                marker = dict(color = 'blue',
                             line=dict(color='black',width=1.5)),
                text = counts.CaseNumber)
data = [trace1]
layout = go.Layout(barmode = "group",title='Total # of Calls by Type')
fig = go.Figure(data = data, layout = layout)
iplot(fig)
