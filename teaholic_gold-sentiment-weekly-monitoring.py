# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

date_polarity_df = pd.read_csv('../input/weekly_polarities.csv', sep='\t')
date_polarity_df['MondayWeek'] = pd.to_datetime(date_polarity_df['MondayWeek'])
# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

def allmondays(year):
    return pd.date_range(start=str(year), end=str(year+1), 
                         freq='W-MON').strftime('%m/%d/%Y').tolist()

monday_list = pd.to_datetime(allmondays(2018)[-15:])
neutral = [date_polarity_df[date_polarity_df['MondayWeek']==date]['neutral'].item() if date in date_polarity_df['MondayWeek'].tolist() else None for date in monday_list]
positive = [date_polarity_df[date_polarity_df['MondayWeek']==date]['positive'].item() if date in date_polarity_df['MondayWeek'].tolist() else None for date in monday_list]
negative = [date_polarity_df[date_polarity_df['MondayWeek']==date]['negative'].item() if date in date_polarity_df['MondayWeek'].tolist() else None for date in monday_list]

# specify that we want a scatter plot with, with date on the x axis and meet on the y axis
positive = go.Scatter(x=monday_list, y=positive, name='positive', line = dict(
        color = ('rgb(22, 96, 167)'),
        width = 4,
        dash = 'dot'), connectgaps=False)
neutral = go.Scatter(x=monday_list, y=neutral, name='neutral', line = dict(
        color = ('rgb(22, 96, 167)')), connectgaps=False)
negative = go.Scatter(x=monday_list, y=negative, name='negative', line = dict(
        color = ('rgb(205, 12, 24)'),
        width = 4,
        dash = 'dot'), connectgaps=False)

# specify the layout of our figure
layout = dict(title = "Reuters Gold Market Report",
              xaxis= dict(title= 'Week (Monday)',ticklen= 5,zeroline= False), yaxis= dict(title= 'Signal'))
data = [positive, neutral, negative]

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)
