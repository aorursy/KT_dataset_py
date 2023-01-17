# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/road-weather-information-stations.csv")
temp_df = data[['DateTime', 'AirTemperature']]
temp_df['DateTime'] = pd.to_datetime(temp_df['DateTime'])
temp_df = temp_df.set_index('DateTime')
pv = pd.pivot_table(temp_df, index=temp_df.index.week, columns=temp_df.index.year,
                    values='AirTemperature') #, aggfunc='sum')

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(pv)
# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

list_plots = []

# specify that we want a scatter plot with, with date on the x axis and meet on the y axis
data1 = go.Scatter(x=pv.index, y=pv[2017])
list_plots.append(data1)
data2 = go.Scatter(x=pv.index, y=pv[2018])
list_plots.append(data2)

# specify the layout of our figure
layout = dict(title = "Average temperature (F) per week",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = list_plots, layout = layout)
iplot(fig)
