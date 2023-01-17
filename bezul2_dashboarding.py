# libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

import os
print(os.listdir("../input"))

# read data
df = pd.read_csv("../input/fire-department-calls-for-service.csv")

# no of callls in last 14 days - separation by Watch Time 
# Watch date -. when the call is received. Watch date starts at 0800 each morning and ends at 0800 the next day.

#additional variable - date od WatchDate
df["WatchDay"] = pd.to_datetime(df["Watch Date"]).dt.date

#subset - calls from last 14 days
df14 = df[pd.to_datetime(df["Watch Date"]) > datetime.datetime.now() - pd.to_timedelta('14day')]

gr = df14[df14['Call Type'] == 'Alarms'].WatchDay.value_counts().sort_index()
gr = pd.DataFrame(gr)
gr.index.name = 'Date'
gr.reset_index(inplace = True)

# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis
data = [go.Line(x=gr.Date, y=gr.WatchDay)]

# specify the layout of our figure
layout = dict(title = "Number of alarms per day in last two weeks in SF",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)