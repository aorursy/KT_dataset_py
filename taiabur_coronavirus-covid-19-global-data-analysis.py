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
#plotly
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
file = "../input/coronavirus-covid19-dataset/corona_latest.csv"
df = pd.read_csv(file)
df
df = pd.read_csv(file,index_col=[0])
df.head()
df.tail()
df.describe()
pd.isnull(df)
fig = go.Figure()

fig.add_trace(go.Bar(
    x=df["Country,Other"][df.TotalCases>=10000],
    y=df.TotalCases[df.TotalCases>=10000],
    name='TotalCases',
    marker_color='#1b81c2',
    text=df.TotalCases[df.TotalCases>=10000],
    textposition='auto'
))
fig.add_trace(go.Bar(
    x=df["Country,Other"][df.TotalCases>=10000],
    y=df["TotalRecovered"][df.TotalCases>=10000],
    name='TotalRecovered',
    marker_color='#2ca02c',
    text=df.TotalRecovered[df.TotalCases>=10000],
    textposition='auto'
))

fig.add_trace(go.Bar(
    x=df["Country,Other"][df.TotalCases>=10000],
    y=df.TotalDeaths[df.TotalCases>=10000],
    name='TotalDeaths',
    marker_color='red',
    text=df.TotalDeaths[df.TotalCases>=10000],
    textposition='auto'
))
fig.update_layout(
    title={
        'text': "Corona virus TotalCase greater than 10,000",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<10000 )&(df.TotalCases>5000))],
    y=df.TotalCases[((df.TotalCases<10000 )&(df.TotalCases>5000))],
    name='TotalCases',
    marker_color='#1b81c2',
    text=df.TotalCases[((df.TotalCases<10000 )&(df.TotalCases>5000))],
    textposition='auto'
))
fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<10000 )&(df.TotalCases>5000))],
    y=df["TotalRecovered"][((df.TotalCases<10000 )&(df.TotalCases>5000))],
    name='TotalRecovered',
    marker_color='#2ca02c',
    text=df.TotalRecovered[((df.TotalCases<10000 )&(df.TotalCases>5000))],
    textposition='auto'
))

fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<10000 )&(df.TotalCases>5000))],
    y=df.TotalDeaths[((df.TotalCases<10000 )&(df.TotalCases>5000))],
    name='TotalDeaths',
    marker_color='red',
    text=df.TotalDeaths[((df.TotalCases<10000 )&(df.TotalCases>5000))],
    textposition='auto'
))
fig.update_layout(
    title={
        'text': "Corona virus TotalCase greater than 5000 and less than 10000",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<5000 )&(df.TotalCases>1000))],
    y=df.TotalCases[((df.TotalCases<5000 )&(df.TotalCases>1000))],
    name='TotalCases',
    marker_color='#1b81c2',
    text=df.TotalCases[((df.TotalCases<5000 )&(df.TotalCases>1000))],
    textposition='auto'
))
fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<5000 )&(df.TotalCases>1000))],
    y=df["TotalRecovered"][((df.TotalCases<5000 )&(df.TotalCases>1000))],
    name='TotalRecovered',
    marker_color='#2ca02c',
    text=df.TotalRecovered[((df.TotalCases<5000 )&(df.TotalCases>1000))],
    textposition='auto'
))

fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<5000 )&(df.TotalCases>1000))],
    y=df.TotalDeaths[((df.TotalCases<5000 )&(df.TotalCases>1000))],
    name='TotalDeaths',
    marker_color='red',
    text=df.TotalDeaths[((df.TotalCases<5000 )&(df.TotalCases>1000))],
    textposition='auto'
))
fig.update_layout(
    title={
        'text': "Corona virus TotalCase greater than 1000 and less than 5000",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<1000 )&(df.TotalCases>500))],
    y=df.TotalCases[((df.TotalCases<1000 )&(df.TotalCases>500))],
    name='TotalCases',
    marker_color='#1b81c2',
    text=df.TotalCases[((df.TotalCases<1000 )&(df.TotalCases>500))],
    textposition='auto'
))
fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<1000 )&(df.TotalCases>500))],
    y=df["TotalRecovered"][((df.TotalCases<1000 )&(df.TotalCases>500))],
    name='TotalRecovered',
    marker_color='#2ca02c',
    text=df.TotalRecovered[((df.TotalCases<1000 )&(df.TotalCases>500))],
    textposition='auto'
))

fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<1000 )&(df.TotalCases>500))],
    y=df.TotalDeaths[((df.TotalCases<1000 )&(df.TotalCases>500))],
    name='TotalDeaths',
    marker_color='red',
    text=df.TotalDeaths[((df.TotalCases<1000 )&(df.TotalCases>500))],
    textposition='auto'
))
fig.update_layout(
    title={
        'text': "Corona virus TotalCase greater than 500 and less than 1000",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<500 )&(df.TotalCases>250))],
    y=df.TotalCases[((df.TotalCases<500 )&(df.TotalCases>250))],
    name='TotalCases',
    marker_color='#1b81c2',
    text=df.TotalCases[((df.TotalCases<500 )&(df.TotalCases>250))],
    textposition='auto'
))
fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<500 )&(df.TotalCases>250))],
    y=df["TotalRecovered"][((df.TotalCases<500 )&(df.TotalCases>250))],
    name='TotalRecovered',
    marker_color='#2ca02c',
    text=df.TotalRecovered[((df.TotalCases<500 )&(df.TotalCases>2500))],
    textposition='auto'
))

fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<500 )&(df.TotalCases>250))],
    y=df.TotalDeaths[((df.TotalCases<500 )&(df.TotalCases>0))],
    name='TotalDeaths',
    marker_color='red',
    text=df.TotalDeaths[((df.TotalCases<500 )&(df.TotalCases>250))],
    textposition='auto'
))
fig.update_layout(
    title={
        'text': "Corona virus TotalCase greater than 250 and less than 500",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<250 )&(df.TotalCases>100))],
    y=df.TotalCases[((df.TotalCases<250 )&(df.TotalCases>100))],
    name='TotalCases',
    marker_color='#1b81c2',
    text=df.TotalCases[((df.TotalCases<250 )&(df.TotalCases>100))],
    textposition='auto'
))
fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<250 )&(df.TotalCases>100))],
    y=df["TotalRecovered"][((df.TotalCases<250 )&(df.TotalCases>100))],
    name='TotalRecovered',
    marker_color='#2ca02c',
    text=df.TotalRecovered[((df.TotalCases<250 )&(df.TotalCases>100))],
    textposition='auto'
))

fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<250 )&(df.TotalCases>100))],
    y=df.TotalDeaths[((df.TotalCases<250 )&(df.TotalCases>100))],
    name='TotalDeaths',
    marker_color='red',
    text=df.TotalDeaths[((df.TotalCases<250 )&(df.TotalCases>100))],
    textposition='auto'
))
fig.update_layout(
    title={
        'text': "Corona virus TotalCase greater than 100 and less than 250",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<100 )&(df.TotalCases>50))],
    y=df.TotalCases[((df.TotalCases<100 )&(df.TotalCases>50))],
    name='TotalCases',
    marker_color='#1b81c2',
    text=df.TotalCases[((df.TotalCases<100 )&(df.TotalCases>50))],
    textposition='auto'
))
fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<100 )&(df.TotalCases>50))],
    y=df["TotalRecovered"][((df.TotalCases<100 )&(df.TotalCases>50))],
    name='TotalRecovered',
    marker_color='#2ca02c',
    text=df.TotalRecovered[((df.TotalCases<100 )&(df.TotalCases>50))],
    textposition='auto'
))

fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<100 )&(df.TotalCases>50))],
    y=df.TotalDeaths[((df.TotalCases<100 )&(df.TotalCases>50))],
    name='TotalDeaths',
    marker_color='red',
    text=df.TotalDeaths[((df.TotalCases<100 )&(df.TotalCases>50))],
    textposition='auto'
))
fig.update_layout(
    title={
        'text': "Corona virus TotalCase greater than 50 and less than 100",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<50 )&(df.TotalCases>0))],
    y=df.TotalCases[((df.TotalCases<50 )&(df.TotalCases>0))],
    name='TotalCases',
    marker_color='#1b81c2',
    text=df.TotalCases[((df.TotalCases<50 )&(df.TotalCases>0))],
    textposition='auto'
))
fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<50 )&(df.TotalCases>0))],
    y=df["TotalRecovered"][((df.TotalCases<50 )&(df.TotalCases>0))],
    name='TotalRecovered',
    marker_color='#2ca02c',
    text=df.TotalRecovered[((df.TotalCases<50 )&(df.TotalCases>0))],
    textposition='auto'
))

fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<50 )&(df.TotalCases>0))],
    y=df.TotalDeaths[((df.TotalCases<50 )&(df.TotalCases>0))],
    name='TotalDeaths',
    marker_color='red',
    text=df.TotalDeaths[((df.TotalCases<50 )&(df.TotalCases>0))],
    textposition='auto'
))
fig.update_layout(
    title={
        'text': "Corona virus TotalCase greater than 0 and less than 50",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()
top_deaths = df.nlargest(5, ['TotalDeaths'])
top_deaths
fig = go.Figure()

fig.add_trace(go.Bar(
    x=top_deaths["Country,Other"],
    y=top_deaths.TotalDeaths,
    name='TotalDeaths',
    marker_color='#1b81c2',
    text=top_deaths.TotalDeaths,
    textposition='auto'
))

fig.update_layout(
    title={
        'text': "Deaths by Country(Top 5)",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()
zero_death = df[df["TotalDeaths"] == 0]
pd.set_option('display.max_rows', None)
def highlight_col(x):
    r = 'background-color: red'
    df1 = pd.DataFrame('', index=x.index, columns=x.columns)
    df1.iloc[:, 3] = r
    return df1    
zero_death.style.apply(highlight_col, axis=None)
val = len(zero_death.index)
print("There are",val, "countries with no death")