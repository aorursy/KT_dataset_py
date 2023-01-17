# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

from datetime import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
N = 1
dfs = pd.read_csv('/kaggle/input/covid19-in-spain/nacional_covid19.csv', header=0, names=['Date', 'Cases', 'Recovered', 'Deaths', 'ICU', 'Hospitalized'])

dfs.head(10)
dfsn = dfs.loc[dfs['ICU'] >= N].reset_index(drop=True)

dfsn.head()
dfi = pd.read_csv('/kaggle/input/covid19-in-italy/covid19_italy_region.csv')

dfi = dfi.drop(['SNo', 'RegionCode', 'Latitude', 'Longitude'], axis=1)

dfi = dfi.groupby('Date').sum()

dfi.head()
dfin = dfi.loc[dfi['IntensiveCarePatients'] >= N].reset_index('Date', drop=True)

# dfin = dfin.set_index(dfin.groupby('IntensiveCarePatients').cumcount().rename('Day'), append=True)#.reset_index() # date_of_N_cases

dfin.head()
fig = go.Figure(data=[

    go.Bar(name='Spain', y=dfsn['ICU']),

    go.Bar(name='Italy', y=dfin['IntensiveCarePatients'])

])

# Change the bar mode

fig.update_layout(title='Spain and Italy ICU vs. day',

                  xaxis_title="Days since %d case"%N,

                  yaxis_title="Total ICU patients",

                  barmode='group')

fig.show()