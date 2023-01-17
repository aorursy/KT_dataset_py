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
vehicle_collisions_df = pd.read_csv('../input/nypd-motor-vehicle-collisions.csv')
vehicle_collisions_df.head()
# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
collisions_by_month = pd.DataFrame()
vehicle_collisions_df['DATE'] = pd.to_datetime(vehicle_collisions_df['DATE'], format = "%Y-%m-%d")
collisions_by_month['collisions'] = vehicle_collisions_df['DATE'].groupby([vehicle_collisions_df.DATE.dt.year, vehicle_collisions_df.DATE.dt.month]).agg('count') 
collisions_by_month['DATE'] = collisions_by_month.index
collisions_by_month['DATE'] = pd.to_datetime(collisions_by_month['DATE'], format="(%Y, %m)")
collisions_by_month.head()


# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis
data = [go.Scatter(x=collisions_by_month.DATE, y=collisions_by_month.collisions)]

# specify the layout of our figure
layout = dict(title = "Number of Collisions per Month",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data , layout = layout)
iplot(fig)