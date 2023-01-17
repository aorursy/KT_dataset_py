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
raw_data = pd.read_csv("../input/nypd-motor-vehicle-collisions.csv")
raw_data['DATE'] = pd.to_datetime(raw_data['DATE'], format="%Y-%m-%d")
day_data = raw_data\
                .filter(items=['DATE','NUMBER OF PERSONS INJURED','NUMBER OF PERSONS KILLED'])\
                .groupby([raw_data.DATE.dt.year, raw_data.DATE.dt.month, raw_data.DATE.dt.day])\
                .sum()
day_data['date'] = day_data.index
day_data['date'] = pd.to_datetime(day_data['date'], format="(%Y, %m, %d)")
day_data = day_data.reset_index(drop=True)
day_data=day_data.tail(100)
# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()


# making a chart
trace1 = go.Scatter(
    x=day_data.date, # assign x as the dataframe column 'x'
    y=day_data['NUMBER OF PERSONS INJURED'],
    name='INJURED'
)
trace2 = go.Scatter(
    x=day_data.date,
    y=day_data['NUMBER OF PERSONS KILLED'],
    name='KILLED'
)

plotted_data = [trace1, trace2]
layout = dict(title="Number of persons injured/killed per month",
              xaxis=dict(title='Date', ticklen=10, zeroline=False))
fig = dict(data=plotted_data, layout=layout)
iplot(fig)