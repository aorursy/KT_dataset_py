# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

data = pd.read_csv('../input/fire-department-calls-for-service.csv')


# Missing Values
per = (data.isnull().sum()/data.shape[0])*100
percents = per.iloc[per.nonzero()[0]]


percents.plot.barh()
plt.title('Missing Values')
plt.xlabel('Percentage')
plt.show()


# data munging

# parse dates
data['Call Date'] = pd.to_datetime(data['Call Date'], format = "%Y-%m-%d")

# count of meets per month
calls_by_month = data['Call Date'].groupby([data['Call Date'].dt.year, data['Call Date'].dt.month]).agg('count') 

# convert to dataframe
calls_by_month = calls_by_month.to_frame()

# move date month from index to column
calls_by_month['date'] = calls_by_month.index

# rename column
calls_by_month = calls_by_month.rename(columns={calls_by_month.columns[0]:"calls"})

# re-parse dates
calls_by_month['date'] = pd.to_datetime(calls_by_month['date'], format="(%Y, %m)")

# remove index
calls_by_month = calls_by_month.reset_index(drop=True)

# get month of meet
calls_by_month['month'] = calls_by_month.date.dt.month

# import plotly

import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow the code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis
new_data = [go.Scatter(x=calls_by_month.date, y=calls_by_month.calls)]

# specify the layout of our figure
layout = dict(title = "Number of Calls to Fire Department per Month",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = new_data, layout = layout)
iplot(fig)

# count of calls by type
calls_by_type = data['Call Type'].value_counts().to_frame()

# move call type to column
calls_by_type['Call Types'] = calls_by_type.index

# rename column
calls_by_type = calls_by_type.rename(columns={calls_by_type.columns[0]:"calls"})

# remove index
calls_by_type = calls_by_type.reset_index(drop=True)


init_notebook_mode()

# sepcify that we want a bar plot with, with call type on the x axis and calls on the y axis
type_data = [go.Bar(x=calls_by_type['Call Types'], y=calls_by_type['calls'])]

# specify the layout of our figure
layout = dict(title = "Number of Calls to Fire Department by Type",
              xaxis= dict(title= 'Call Type',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = type_data, layout = layout)
iplot(fig)
# count of calls by unit id
calls_by_UnitID = data['Unit ID'].value_counts().to_frame()

# move call unit id to column
calls_by_UnitID['Unit IDs'] = calls_by_UnitID.index

# rename column
calls_by_UnitID = calls_by_UnitID.rename(columns={calls_by_UnitID.columns[0]:"calls"})

# remove index
calls_by_UnitID = calls_by_UnitID.reset_index(drop=True)
calls_by_UnitID.head()
init_notebook_mode()
# sepcify that we want a bar plot with, with call unit id on the x axis and calls on the y axis
unitID_data = [go.Bar(x=calls_by_UnitID['Unit IDs'], y=calls_by_UnitID['calls'])]

# specify the layout of our figure
layout = dict(title = "Number of Calls to Fire Department by Unit ID",
              xaxis= dict(title= 'Call Unit ID',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = unitID_data, layout = layout)
iplot(fig)