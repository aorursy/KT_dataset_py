# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime


# import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools # for sub figures

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_ny_bus_delay = pd.read_csv("../input/bus-breakdown-and-delays.csv", index_col=0)
df_ny_bus_delay.head()
df_ny_bus_delay['Occurred_On'] = pd.to_datetime(df_ny_bus_delay['Occurred_On'])
# breakdown_dates = df_ny_bus_delay.loc[(df_ny_bus_delay['Occurred_On'] < datetime.now()), ['Occurred_On','Busbreakdown_ID']].groupby('Occurred_On').count().dropna()
breakdown_dates = df_ny_bus_delay.loc[(df_ny_bus_delay['Occurred_On'] < datetime.now()), ['Occurred_On','Busbreakdown_ID']].groupby(['Occurred_On']).count().dropna()
breakdown_dates.rename(columns={'Busbreakdown_ID' : 'n'}, inplace=True)
breakdown_dates.head()

data_dates = [go.Scatter(x=breakdown_dates.index, y=breakdown_dates.n)]

# specify the layout of our figure
layout_dates = dict(title = "Number of bus breakdowns",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data_dates, layout = layout_dates)
iplot(fig)
delay_by_borough = df_ny_bus_delay.loc[(df_ny_bus_delay['Occurred_On'] < datetime.now()), ['Boro','Busbreakdown_ID']].groupby(['Boro']).count().dropna()
delay_by_borough.rename(columns={'Busbreakdown_ID' : 'n'}, inplace=True)
data = [go.Bar(
            x=delay_by_borough.index,
            y=delay_by_borough.n
    )]

layout = dict(title = "Bus delays by borough",
              xaxis= dict(title= 'Borough',ticklen= 5,zeroline= False))
iplot(data, layout)
