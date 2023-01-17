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
date_polarity_df = pd.read_csv('../input/daily_polarities.csv', sep='\t')

date_polarity_df.columns
date_polarity_df['date'] = [pd.to_datetime(day) for day in date_polarity_df['date']]

day_range = [pd.to_datetime(day) for day in pd.date_range(start=date_polarity_df['date'].min(), end=date_polarity_df['date'].max())]
import seaborn as sns

import matplotlib.pyplot as plt

% matplotlib inline 



sns.tsplot(date_polarity_df['neutral'], time=date_polarity_df['date'], color='gold', legend=True)

sns.tsplot(date_polarity_df['positive'], time=date_polarity_df['date'], color='blue', legend=True)

sns.tsplot(date_polarity_df['negative'], time=date_polarity_df['date'], color='red', legend=True)

plt.title('Gold dataset')

plt.ylabel('Signals (paragraphs)')

plt.xlabel('Day')

#plt.legend()
neutral_signal_count = [date_polarity_df[date_polarity_df['date']==date]['neutral'].item() if date in date_polarity_df['date'].tolist() else 0 for date in day_range]

positive_signal_count = [date_polarity_df[date_polarity_df['date']==date]['positive'].item() if date in date_polarity_df['date'].tolist() else 0 for date in day_range]

negative_signal_count = [date_polarity_df[date_polarity_df['date']==date]['negative'].item() if date in date_polarity_df['date'].tolist() else 0 for date in day_range]



sns.tsplot(neutral_signal_count, time=day_range, color='gold', legend=True)

sns.tsplot(positive_signal_count, time=day_range, color='red', legend=True)

sns.tsplot(negative_signal_count, time=day_range, color='blue', legend=True)

plt.title('Gold dataset')

plt.ylabel('Signals (paragraphs)')

plt.xlabel('Day')

#plt.legend()
# Interactive plot

# import plotly

import plotly.plotly as py

import plotly.graph_objs as go



# these two lines are what allow your code to show up in a notebook!

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()



# specify that we want a scatter plot with, with date on the x axis and meet on the y axis

positive = go.Scatter(x=day_range, y=positive_signal_count, name='positive', line = dict(

        color = ('rgb(22, 96, 167)'),

        width = 4,

        dash = 'dot'))

neutral = go.Scatter(x=day_range, y=neutral_signal_count, name='neutral', line = dict(

        color = ('rgb(22, 96, 167)')))

negative = go.Scatter(x=day_range, y=negative_signal_count, name='negative', line = dict(

        color = ('rgb(205, 12, 24)'),

        width = 4,

        dash = 'dash'))

# specify the layout of our figure

layout = dict(title = "Reuters Gold Market Report",

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False), yaxis= dict(title= 'Signal'))

data = [positive, neutral, negative]



# create and show our figure

fig = dict(data = data, layout = layout)
# Zeroing no-signal

iplot(fig)
# Deleting zero-signal

neutral_signal_count = [date_polarity_df[date_polarity_df['date']==date]['neutral'].item() if date in date_polarity_df['date'].tolist() else None for date in day_range]

positive_signal_count = [date_polarity_df[date_polarity_df['date']==date]['positive'].item() if date in date_polarity_df['date'].tolist() else None for date in day_range]

negative_signal_count = [date_polarity_df[date_polarity_df['date']==date]['negative'].item() if date in date_polarity_df['date'].tolist() else None for date in day_range]



# specify that we want a scatter plot with, with date on the x axis and meet on the y axis

positive = go.Scatter(x=day_range, y=positive_signal_count, name='positive', line = dict(

        color = ('rgb(22, 96, 167)'),

        width = 4,

        dash = 'dot'), connectgaps=False)

neutral = go.Scatter(x=day_range, y=neutral_signal_count, name='neutral', line = dict(

        color = ('rgb(22, 96, 167)')), connectgaps=False)

negative = go.Scatter(x=day_range, y=negative_signal_count, name='negative', line = dict(

        color = ('rgb(205, 12, 24)'),

        width = 4,

        dash = 'dot'), connectgaps=False)



# specify the layout of our figure

layout = dict(title = "Reuters Gold Market Report",

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False), yaxis= dict(title= 'Signal'))

data = [positive, neutral, negative]



# create and show our figure

fig = dict(data = data, layout = layout)
# Hiding no-signal

iplot(fig)