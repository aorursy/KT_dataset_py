# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import plotly.graph_objects as go 

import seaborn as sns

import plotly

import plotly.express as px

from fbprophet.plot import plot_plotly

from fbprophet import Prophet
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot, plot_mpl

import plotly.offline as py

init_notebook_mode(connected=True)
# Reading dataset

#data = pd.read_csv('Soma:/corona/covid19-in-india/covid_19_india.csv')

data = pd.read_csv('../input/covid-19-india1/covid_19_india.csv')
data
new = px.bar(data, x='Date', y='ConfirmedIndianNational', hover_data=['State/UnionTerritory','ConfirmedForeignNational', 'Deaths','Cured'], color='State/UnionTerritory')

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Confirmed covid-19 case plot for India',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

new.update_layout(annotations=annotations)

new.show()
#Confirmed ALL

import plotly.graph_objects as go

new1 = go.Figure()

new1.update_layout(template='plotly_dark')

new1.add_trace(go.Scatter(x=data['Date'], 

                         y=data['ConfirmedIndianNational'],

                         mode='lines+markers',

                         name='Confirmed',

                         line=dict(color='Blue', width=2)))

new1.add_trace(go.Scatter(x=data['Date'], 

                         y=data['Deaths'],

                         mode='lines+markers',

                         name='Deaths',

                         line=dict(color='Red', width=2)))

new1.add_trace(go.Scatter(x=data['Date'], 

                         y=data['Cured'],

                         mode='lines+markers',

                         name='Recovered',

                         line=dict(color='Green', width=2)))

new1.show()