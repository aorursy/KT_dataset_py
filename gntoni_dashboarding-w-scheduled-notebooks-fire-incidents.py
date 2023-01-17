# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os



print("Reading dataset...")

data = pd.read_csv('../input/fire-department-calls-for-service.csv')

print("Done.")



# Let's plot the number of calls during the last 18years

yearMonth = data['Call Date'].apply(lambda x: x[:7])

callsByDate = yearMonth.value_counts()

callsByDate = callsByDate.sort_index()

dates = callsByDate.keys()

calls = callsByDate.values



# Get Numbers by trimester

datesTrim = dates[::3].get_values()

callsTrim = calls.reshape(-1,3).sum(1)



data_plot = [go.Scatter(x=datesTrim, y=callsTrim)]

layout = dict(title = "Number of Calls per Trimester",

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))



# create and show our figure

fig = dict(data = data_plot, layout = layout)

iplot(fig)

#plt.xticks(range(len(datesTrim))[::3], rotation='vertical')

#plt.bar(datesTrim, callsTrim)
