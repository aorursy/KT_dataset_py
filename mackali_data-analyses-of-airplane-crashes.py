# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter

# plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import seaborn as sns

# word cloud library

from wordcloud import WordCloud

import matplotlib.pyplot as plt

import datetime as dt



# matplotlib



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/planecrashinfo_20181121001952.csv')

data.info()



data.head()
date = []

for each in data.date:

    x = pd.to_datetime(each)

    date.append(x)

    

data.date = date

seperate = data.aboard.str.split()

a,b,c = zip(*seperate)

seperate = data.fatalities.str.split()

d,e,f = zip(*seperate)

data.aboard = a

data.fatalities = d

#data.dropna(inplace = True)

data.aboard.replace(['?'],0,inplace = True)

data.ac_type.replace(['?'],'a b',inplace = True)

data.fatalities.replace(['?'],0,inplace = True)
data.aboard = data.aboard.astype(float)

data.fatalities = data.fatalities.astype(float)
data.head()
sum_fatalities = []

sum_aboard = []

year = data.date.dt.year.unique()

for each in year:

    x = sum(data.fatalities[data.date.dt.year==each])

    y = sum(data.aboard[data.date.dt.year==each])

    sum_fatalities.append(x)

    sum_aboard.append(y)

    
trace1 = go.Bar(

    x = year,

    y = sum_fatalities,

    marker = dict(color = 'blue'),

    name = 'death :'

    )

trace2 = go.Scatter(

    x = year,

    y = sum_aboard,

    mode = 'markers+lines',

    marker = dict(color = 'red'),

    name = 'aboard :'

    )

layout = dict(title = 'DEATHS AND ABOARD',

              xaxis= dict(title= 'YEAR',ticklen= 15,zeroline= False),yaxis = dict(title = 'NUMBERS',ticklen= 15,zeroline= False))

data1 = [trace1,trace2]

fig = dict(data=data1,layout=layout)

iplot(fig)
data['time'].replace(['?'],'00:00',inplace = True)

import re

time = []

for each in data.time:

    x = re.sub('[^0-9]','',each)

    x = re.sub(' ','',x)

    if len(x)!=4:

        x = '0000'

    a = list(x)

    a.insert(2,':')

    a = ''.join(a)

    time.append(a)

   

data['time'] = time

sep = data.ac_type.str.split() 

brand = []

for each in sep:

  

    brand.append( each[0])

data.ac_type = brand
data['death_rate'] = data.fatalities/data.aboard





data['dead_or_alive'] = ['alive' if each<0.5 else 'dead' for each in data.death_rate]



data.head()
x = Counter(data.ac_type)



y = x.most_common(7)



a,b = zip(*y)



trace1 = go.Pie(

    values = b,

    labels = a)



data2 = [trace1]



layout = dict(title = 'TOP 7 DEADLIEST AIRCRAFTS IN HISTORY')



fig = dict(data = data2,layout = layout)



iplot(fig)
a = Counter(data.ac_type)



b = a.most_common(5)



x,y = zip(*b)



a = data[data.ac_type==x[0]]



b = data[data.ac_type==x[1]]



c = data[data.ac_type==x[2]]



d = data[data.ac_type==x[3]]



e = data[data.ac_type==x[4]]



data1 = pd.concat([a,b,c,d,e],axis=0)
plt.figure(figsize=(10,10))



sns.swarmplot(data1.ac_type,data1.date.dt.year,hue=data.dead_or_alive)
len(data)
a = data[data.time == '00:00'].index

data.drop(a,inplace = True)

a = pd.to_datetime(data.time)



b = a.dt.hour.values



data['time'] = b



len(data)
data.head()
trace1 = go.Histogram(

    x = data.time,

    opacity = 0.5,

    marker = dict(color = 'blue'))

layout = dict(title = 'AC-CRASHES PER HOUR',xaxis = dict(title = 'HOURS'),yaxis = dict(title = 'CRASH NUMBER'))

dat = [trace1]

fig = {'data':dat,'layout':layout}

iplot(fig)
plt.figure(figsize=(10,10))



data.groupby(data.time).death_rate.mean().plot()

plt.title('time-death rate correlation')

plt.xlabel('time')

plt.ylabel('death rate')
