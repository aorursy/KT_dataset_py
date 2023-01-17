###VERSION 3
import numpy as np

import pandas as pd 

from matplotlib import pyplot as plt

import seaborn as sb

import datetime 

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, plot, iplot



%matplotlib inline

init_notebook_mode(connected = True) ##Print plotly graphs in notebook

import warnings; warnings.simplefilter('ignore')###Hide deprecation warning for sb.tsplot
data = pd.read_csv("../input/candy_production.csv")

data.head()
##Create month and year columns by converting timestamps to datetime objects

data['Date'] = pd.to_datetime(data['observation_date'])

data['Month'] = ''

data['Year'] = ''

for i in range(0,len(data)):

    data.loc[i,('Month')] = data['Date'][i].month

    data.loc[i,('Year')] = data['Date'][i].year

data.head()
plot1 = [go.Scatter(

          x=data['Date'],

          y=data['IPG3113N'])]

iplot(plot1)
year_sum = data.groupby(['Year'])['IPG3113N'].sum()

year_sum = year_sum.drop(2017)

plot2 = [go.Scatter(

          x=year_sum.index,

          y=year_sum)] 

iplot(plot2)
fig, ax = plt.subplots()

fig.set_size_inches(12.94427191, 8)##golden ratio

sb.tsplot(data=data, time = 'Year',condition="Month", unit = 'Month', value="IPG3113N")   

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,title = 'Month')
plt.plot(data.groupby(['Month'])['IPG3113N'].mean())

trace = go.Pie(labels=data['Month'], values=data.groupby(['Month'])['IPG3113N'].mean())

iplot([trace])