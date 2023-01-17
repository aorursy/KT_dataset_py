

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline





import plotly.plotly as py

import plotly.graph_objs as go

from scipy import stats

from plotly.offline import iplot, init_notebook_mode

import cufflinks



# Using plotly + cufflinks in offline mode

init_notebook_mode(connected=True)

cufflinks.go_offline(connected=True)



# To interactive buttons

import ipywidgets as widgets

from ipywidgets import interact, interact_manual

import plotly.plotly as py

import plotly.graph_objs as go

import plotly.tools as tls



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/world-happiness-report-2019.csv")
data.head()
data.shape
data.isnull().sum()
data.groupby(['Country (region)']).sum()['Positive affect'].nlargest(20).iplot(kind='bar',

                                                                     xTitle='Country', yTitle='Positive Emotion',

                                                                     title='Countries with most positive emotion',colors='blue')
data.groupby(['Country (region)']).sum()['Negative affect'].nlargest(20).iplot(kind='bar',

                                                                     xTitle='Country', yTitle='Negative Emotion',

                                                                     title='Countries with most negative emotion',colors = 'red')
trace2 = go.Line(

    x=data.sort_values(['Social support'],ascending=False)['Country (region)'][:30],

    y=data.sort_values(['Social support'],ascending=False)['Social support'][:30],

    name='Social Support'

)







d = [trace2]

layout = go.Layout(

    barmode='group'

)



fig = go.Figure(data=d, layout=layout)

iplot(fig)
trace2 = go.Bar(

    x=data.sort_values(['Freedom'],ascending=False)['Country (region)'][:30],

    y=data.sort_values(['Freedom'],ascending=False)['Freedom'][:30],

    name='Freedom'

)







d = [trace2]

layout = go.Layout(

    barmode='group'

)



fig = go.Figure(data=d, layout=layout)

iplot(fig)
plt.figure(figsize = (19,15))

sns.heatmap(data.corr(),annot = True)
trace2 = go.Scatter(

    x = data['Positive affect'],

    y = data['Negative affect'],

    mode = 'markers'

)







d = [trace2]

layout = go.Layout(

    barmode='group'

)



fig = go.Figure(data=d, layout=layout)

iplot(fig)
data['Positive affect'].corr( data['Negative affect'])
trace0 = go.Scatter(

    x = data['Positive affect'],

    y = data['Freedom'],

    mode = 'markers',

    name = 'markers'

)



d = [trace0]

layout = go.Layout(

    barmode='group'

)



fig = go.Figure(data=d, layout=layout)

iplot(fig)
data['Positive affect'].corr( data['Freedom'])
plt.figure(figsize = (19,15))

sns.regplot(x = 'Healthy life\nexpectancy', y ='Log of GDP\nper capita', data = data)
data['Healthy life\nexpectancy'].corr( data['Log of GDP\nper capita'])