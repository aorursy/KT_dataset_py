# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns

# Configuring visualizations
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
# Reading data
df_requests = pd.read_csv('../input/service-requests-received-by-the-oakland-call-center.csv')
# Change date format
df_requests["DATETIMEINIT"] = pd.to_datetime(df_requests["DATETIMEINIT"])
df_requests["DATETIMECLOSED"] = pd.to_datetime(df_requests["DATETIMECLOSED"])
# Check columns
df_requests.info()
df_requests.sample(5)
# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
# Requests by category
series = df_requests.REQCATEGORY.value_counts()

# specify what we want in the scatter plot
data = [go.Bar(x=series.index, y=series)]

# specify the layout of our figure
layout = dict(title = "Total requests by category")

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)
df_requests_illdump = df_requests[df_requests.REQCATEGORY == "ILLDUMP"]
df_requests_illdump.head(5)
# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis
series = df_requests_illdump.DATETIMEINIT.dt.hour.value_counts().sort_index()

# specify what we want in the scatter plot
data = [go.Scatter(x=series.index, y=series)]

# specify the layout of our figure
layout = dict(title = "ILLDUMP Request by hour of day",
              xaxis= dict(title= 'Hour of day', zeroline= True, 
                          tickmode='linear',
                            ticks='outside',
                            tick0=0,
                            dtick=1,
                            ticklen=8,
                            tickwidth=4,
                            tickcolor='#000'
    ))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)
