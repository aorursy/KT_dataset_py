import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import plotly.graph_objs as go
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

import os
print(os.listdir("../input"))
df = pd.read_csv("../input/StudentsPerformance.csv")
df.head()
df.info()
df.describe()
df.isnull().any()
plt.figure(figsize=(8, 6))
sns.heatmap(df.isnull())
trace1 = go.Bar(
            x=df['gender'].value_counts().index,
            y=df['gender'].value_counts().values,
            marker = dict(
                  line=dict(color='rgb(0,0,255)',width=2)),
            name = 'Weapons Acquired'
    )

data = [trace1]

layout = dict(title = 'Gender Distribution',
              xaxis= dict(title= 'Gender',ticklen= 5,zeroline= False),
              yaxis = dict(title = "Count")
             )
fig = dict(data = data, layout=layout)
iplot(fig)
df['gender'].value_counts()/len(df)*100
trace1 = go.Bar(
            x=df['race/ethnicity'].value_counts().index,
            y=df['race/ethnicity'].value_counts().values,
            marker = dict(
                  line=dict(color='rgb(0,0,255)',width=2)),
            name = 'Cast and Creed'
    )

data = [trace1]

layout = dict(title = 'Cast/Races Distribution',
              xaxis= dict(title= 'Races',ticklen= 5,zeroline= False),
              yaxis = dict(title = "Count")
             )
fig = dict(data = data, layout=layout)
iplot(fig)
temp = df['race/ethnicity'].value_counts()/len(df)*100
labels = list(temp.index)
values = list(temp.values)
colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1', '#379A56']

trace = go.Pie(labels=labels, values=values,
              marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)))
data = [trace]
layout = dict(title = 'Cast/Races Percentage Distribution',
              xaxis= dict(title= 'Races',ticklen= 5),
              yaxis = dict(title = "Count")
             )
fig = dict(data = data, layout=layout)
iplot(fig)
colors = ['#379A56', '#D0F9B1']

trace1 = go.Bar(
            x=df['test preparation course'].value_counts().index,
            y=df['test preparation course'].value_counts().values,
            marker = dict(
                color=colors,
                  line=dict(color='rgb(0,0,0)',width=2)),
            name = 'Test Prepration'
    )

data = [trace1]

layout = dict(title = 'Test Prepration',
              xaxis= dict(title= 'Test Prepration',ticklen= 5,zeroline= False),
              yaxis = dict(title = "Count")
             )
fig = dict(data = data, layout=layout)
iplot(fig)
df.columns
df['math score'].unique()
fig, axarr = plt.subplots(2, 2)
ax1, ax2, ax3, ax4 = axarr[0, 0], axarr[0, 1], axarr[1, 0], axarr[1, 1]
fig.set_figheight(10)
fig.set_figwidth(10)

sns.pointplot(y='math score', x='race/ethnicity', data=df, ax=ax1)

sns.pointplot(y='math score', x='gender', data=df, ax=ax2)

sns.pointplot(y='math score', x='race/ethnicity', data=df, hue='test preparation course', ax=ax3)

sns.pointplot(y='math score', x='gender', hue='test preparation course', data=df, ax=ax4)
fig, axarr = plt.subplots(1, 2)
ax1, ax2 = axarr[0], axarr[1]
fig.set_figheight(10)
fig.set_figwidth(18)

sns.pointplot(y='math score', x='parental level of education', data=df, ax=ax1)

sns.pointplot(y='math score', x='lunch', data=df, ax=ax2)
fig, axarr = plt.subplots(2, 2)
ax1, ax2, ax3, ax4 = axarr[0, 0], axarr[0, 1], axarr[1, 0], axarr[1, 1]
fig.set_figheight(10)
fig.set_figwidth(10)

sns.pointplot(y='reading score', x='race/ethnicity', data=df, ax=ax1)

sns.pointplot(y='reading score', x='gender', data=df, ax=ax2)

sns.pointplot(y='reading score', x='race/ethnicity', data=df, hue='test preparation course', ax=ax3)

sns.pointplot(y='reading score', x='gender', hue='test preparation course', data=df, ax=ax4)
fig, axarr = plt.subplots(1, 2)
ax1, ax2 = axarr[0], axarr[1]
fig.set_figheight(10)
fig.set_figwidth(18)

sns.pointplot(y='reading score', x='parental level of education', data=df, ax=ax1)

sns.pointplot(y='reading score', x='lunch', data=df, ax=ax2)
fig, axarr = plt.subplots(2, 2)
ax1, ax2, ax3, ax4 = axarr[0, 0], axarr[0, 1], axarr[1, 0], axarr[1, 1]
fig.set_figheight(10)
fig.set_figwidth(10)

sns.pointplot(y='writing score', x='race/ethnicity', data=df, ax=ax1)

sns.pointplot(y='writing score', x='gender', data=df, ax=ax2)

sns.pointplot(y='writing score', x='race/ethnicity', data=df, hue='test preparation course', ax=ax3)

sns.pointplot(y='writing score', x='gender', hue='test preparation course', data=df, ax=ax4)
fig, axarr = plt.subplots(1, 2)
ax1, ax2 = axarr[0], axarr[1]
fig.set_figheight(10)
fig.set_figwidth(18)

sns.pointplot(y='writing score', x='parental level of education', data=df, ax=ax1)

sns.pointplot(y='writing score', x='lunch', data=df, ax=ax2)