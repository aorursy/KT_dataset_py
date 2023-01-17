# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
df=pd.read_csv('../input/police-department-calls-for-service.csv')

df=df.head(50000)
report=df.loc[:,'Report Date'].values
call=df.loc[:,'Call Date'].values
offense=df.loc[:,'Offense Date'].values
print(np.array_equal(report,call))
print(np.array_equal(report,offense))
df=df.drop(['Call Date','Offense Date'],axis=1)
print(df.State.unique())
df=df.drop(['State'],axis=1)
df.head()
df=df.drop(['Agency Id'],axis=1)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools
from subprocess import check_output
city = df['City'].value_counts()
label = city.index
size = city.values

colors = ['skyblue', 'orange', '#96D38C', '#D0F9B1']
trace = go.Pie(labels=label, values=size, marker=dict(colors=colors))
layout = go.Layout(
    title='City distribution'
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="train")
Address = df['Address Type'].value_counts()
label = Address.index
size = Address.values

colors = ['skyblue', 'orange', '#96D38C', '#D0F9B1']
trace = go.Pie(labels=label, values=size, marker=dict(colors=colors))
layout = go.Layout(
    title='Address Type '
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="train")
Common = df['Common Location'].value_counts()
label = Common.index
size = Common.values

colors = ['skyblue', 'orange', '#96D38C', '#D0F9B1']
trace = go.Pie(labels=label, values=size, marker=dict(colors=colors))
layout = go.Layout(
    title='Common Location'
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="train")
Type = df['Original Crime Type Name'].value_counts()
label = Type.index
size = Type.values

colors = ['skyblue', 'orange', '#96D38C', '#D0F9B1']
trace = go.Pie(labels=label, values=size, marker=dict(colors=colors))
layout = go.Layout(
    title='Original Crime Type Name'
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="train")
df2=df.drop(['Common Location','Original Crime Type Name','Address','Call Date Time'],axis=1)
df2.head()
df2['year']=df2['Report Date'].str[0:4]
df2.year = pd.to_numeric(df2.year)
df2['month']=df2['Report Date'].str[5:7]
df2.month = pd.to_numeric(df2.month)
df2['day']=df2['Report Date'].str[8:10]
df2.day = pd.to_numeric(df2.day)
df2['hour']=df2['Call Time'].str[0:2]
df2.hour = pd.to_numeric(df2.hour)
df2['minute']=df2['Call Time'].str[3:5]
df2.minute = pd.to_numeric(df2.minute)
df3=df2.drop(['Report Date','Call Time'],axis=1)
df3.head()
df3_dummy = pd.get_dummies(df3)
df3_dummy.head()
df3_dummy['date']=df3_dummy['year']+(1/12)*df3_dummy['month']+(1/360)*df3_dummy['day']
df3_dummy.date = pd.to_numeric(df3_dummy.date)
df3_dummy['time']=df3_dummy['hour']+(1/60)*df3_dummy['minute']
df3_dummy.time = pd.to_numeric(df3_dummy.time)
df3_dummy_1=df3_dummy.drop(['year','month','day','hour','minute'],axis=1)
import seaborn as sns
import matplotlib.pyplot as pl

f, ax = pl.subplots(figsize=(10, 8))
corr = df3_dummy_1.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
dff=pd.read_csv('../input/police-department-incidents.csv')
dff.info()
dff=dff.head(20000)
dff.head()
dff=dff.drop(['Location'],axis=1)
Category = dff['Category'].value_counts()
label = Category.index
size = Category.values

colors = ['skyblue', 'orange', '#96D38C', '#D0F9B1']
trace = go.Pie(labels=label, values=size, marker=dict(colors=colors))
layout = go.Layout(
    title='Category'
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="train")
Descript = dff['Descript'].value_counts()
label = Descript.index
size = Descript.values

trace = go.Pie(labels=label, values=size)
layout = go.Layout(
    title='Descript'
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="train")
DayOfWeek = dff['DayOfWeek'].value_counts()
label = DayOfWeek.index
size = DayOfWeek.values

trace = go.Pie(labels=label, values=size)
layout = go.Layout(
    title='DayOfWeek'
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="train")
Resolution = dff['Resolution'].value_counts()
label = Resolution.index
size = Resolution.values

trace = go.Pie(labels=label, values=size)
layout = go.Layout(
    title='Resolution'
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="train")
dff.head()
dff1=dff[(dff.X>-123)&(dff.X<-122)]
plt.figure(figsize=(15,20))
ax=sns.scatterplot(x="X", y="Y", hue="Resolution",data=dff1)
mapbox_access_token='pk.eyJ1IjoiYW1tb24xIiwiYSI6ImNqbGhtdDNtNzFjNzQzd3J2aDFndDNmbmgifQ.-dt3pKGSvkBaSQ17qXVq3A'
data = [
    go.Scattermapbox(
        lat=dff['Y'],
        lon=dff['X'],
        mode='markers',
        marker=dict(
            size=5,
            color='red',
            opacity=0.3
        )),
    ]
layout = go.Layout(
    autosize=True,
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=37.75,
            lon=-122.412102
        ),
        pitch=0,
        zoom=10,
        style='light'
    ),
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='Crime')
dff.head()
dff['date'] = pd.to_datetime(dff['Date'])
dff['year'] = dff['date'].dt.year
dff.year = pd.to_numeric(df2.year)
dff['hour']=dff['Time'].str[0:2]
dff.hour = pd.to_numeric(df2.hour)
dff_dum=dff.drop(['Descript','Date','Time','Address','date'],axis=1)
dff_dum.head()
df3_dummy = pd.get_dummies(dff_dum)
f, ax = pl.subplots(figsize=(10, 8))
corr = df3_dummy.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
ctdf = (dff.reset_index()
          .groupby(['hour','Category'], as_index=False)
          .count()
          # rename isn't strictly necessary here, it's just for readability
          .rename(columns={'index':'ct'})
       )
from matplotlib import pyplot as plt
#plt.figure(figsize=(15,20))
fig, ax = plt.subplots(figsize=(20,20))
# key gives the group name (i.e. category), data gives the actual values
for key, data in ctdf.groupby('Category'):
    data.plot(x='hour', y='ct', ax=ax, label=key)
ctdf = (dff.reset_index()
          .groupby(['hour','DayOfWeek'], as_index=False)
          .count()
          # rename isn't strictly necessary here, it's just for readability
          .rename(columns={'index':'ct'})
       )
fig, ax = plt.subplots(figsize=(20,20))
# key gives the group name (i.e. category), data gives the actual values
for key, data in ctdf.groupby('DayOfWeek'):
    data.plot(x='hour', y='ct', ax=ax, label=key)
ctdf = (dff.reset_index()
          .groupby(['hour','Resolution'], as_index=False)
          .count()
          # rename isn't strictly necessary here, it's just for readability
          .rename(columns={'index':'ct'})
       )
fig, ax = plt.subplots(figsize=(20,20))
# key gives the group name (i.e. category), data gives the actual values
for key, data in ctdf.groupby('Resolution'):
    data.plot(x='hour', y='ct', ax=ax, label=key)