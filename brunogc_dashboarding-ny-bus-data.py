import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})

# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()


import os
import datetime
print(os.listdir("../input"))
today = datetime.datetime.today().date()
df = pd.read_csv("../input/bus-breakdown-and-delays.csv")
time_cols = ["Occurred_On", "Created_On", "Informed_On", "Last_Updated_On"]
for col in time_cols:
    df[f"{col}_date"] = pd.to_datetime(df[col]).dt.date
    df[f"{col}"] = pd.to_datetime(df[col])

# Consider valid all data from the start to tomorrow (due to time zone differences, we don't use today)
valid_df = df[df["Occurred_On_date"] < datetime.date.today() + datetime.timedelta(days=1)]
temp_data = valid_df[["Occurred_On"]].resample('W-Mon', on='Occurred_On').agg('count')
temp_data["date"] = temp_data.index

# re-parse dates
temp_data['date'] = pd.to_datetime(temp_data['date'], format="(%Y-%m-%d)")
# remove index
temp_data = temp_data.reset_index(drop=True)

# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis
data = [go.Scatter(x=temp_data.date, y=temp_data.Occurred_On)]

# specify the layout of our figure
layout = dict(title = "Number of Accidents per Week",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)
data = []
temp = None
for name , group in valid_df[["Reason", "Occurred_On", "Busbreakdown_ID"]].groupby("Reason"):
    temp = group.resample('W-Mon', on='Occurred_On').agg('count')
    temp["date"] = temp.index
    
    # re-parse dates
    temp['date'] = pd.to_datetime(temp['date'], format="(%Y-%m-%d)")
    # remove index
    temp = temp.reset_index(drop=True)
    
    data.append(go.Scatter(x=temp.date, y=temp.Busbreakdown_ID, name=name))

# specify the layout of our figure
layout = dict(title = "Number of Accidents by Type",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)
