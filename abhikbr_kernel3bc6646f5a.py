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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#plotly

import plotly.offline as py

from plotly.offline import iplot,init_notebook_mode,iplot

import plotly.graph_objs as go

py.init_notebook_mode(connected=True)



#cufflinks

import cufflinks as cf

import plotly.offline

cf.go_offline

cf.set_config_file(offline=False,world_readable=True)
data=pd.read_csv('../input/netflix-shows/netflix_titles_nov_2019.csv')

data.head()
data.info()
col="type"

grouped=data[col].value_counts().reset_index()



grouped=grouped.rename(columns={"index":col,col:"count"})

grouped
trace=go.Pie(labels=grouped[col],values=grouped['count'],pull=[0.05,0])

layout=go.Layout(title="TV Shows vs Movie",height=400, legend=dict(x=0.1, y=1))

fig=go.Figure(data=[trace],layout=layout)

fig.show()
data['date_added']=pd.to_datetime(data['date_added'])



data['year_added'] = data['date_added'].dt.year

data['month_added'] = data['date_added'].dt.month

data.head()
data['season_count']=data.apply(lambda x:x['duration'].split(" ")[0] if "Season" in x['duration'] else "",axis=1)

data['duration']=data.apply(lambda x:x['duration'].split(" ")[0] if "Season" not in x['duration'] else "",axis=1)

data.head()
d1=data[data['type']=="TV Show"]

d2=data[data['type']=='Movie']



col="year_added"



vc1=d1[col].value_counts().reset_index()

vc1=vc1.rename(columns={"index":col,col:"count"})

vc1['percent']=vc1['count'].apply(lambda x:x*100/sum(vc1['count']))

vc1=vc1.sort_values(col)

vc1
vc2=d2[col].value_counts().reset_index()

vc2=vc2.rename(columns={"index":col,col:"count"})

vc2['percent']=vc2['count'].apply(lambda x:x*100/sum(vc2['count']))

vc2=vc2.sort_values(col)

vc2
trace1=go.Scatter(x=vc1[col],y=vc1['count'],name="TV Show")

trace2=go.Scatter(x=vc2[col],y=vc2['count'],name="Movie")

data1=[trace1,trace2]

layout=go.Layout(title="Content added over the years",legend=dict(x=0.1,y=1.1,orientation='h'))

fig=go.Figure(data1,layout=layout)

fig.show()
col="release_year"



vc1=d1[col].value_counts().reset_index()

vc1=vc1.rename(columns={"index":col,col:"count"})

vc1['percent']=vc1['count'].apply(lambda x:x*100/sum(vc1['count']))

vc1=vc1.sort_values(col)



vc2=d2[col].value_counts().reset_index()

vc2=vc2.rename(columns={"index":col,col:"count"})

vc2['percent']=vc2['count'].apply(lambda x:x*100/sum(vc2['count']))

vc2=vc2.sort_values(col)



trace1=go.Bar(x=vc1[col],y=vc1['count'],name="TV Show")

trace2=go.Bar(x=vc2[col],y=vc2['count'],name="Movie")

data1=[trace1,trace2]

layout=go.Layout(title="Content added over the years",legend=dict(x=0.1,y=1.1,orientation='h'))

fig=go.Figure(data1,layout=layout)

fig.show()
col="month_added"



vc1=d1[col].value_counts().reset_index()

vc1=vc1.rename(columns={"index":col,col:"count"})

vc1['percent']=vc1['count'].apply(lambda x:x*100/sum(vc1['count']))

vc1=vc1.sort_values(col)



vc2=d2[col].value_counts().reset_index()

vc2=vc2.rename(columns={"index":col,col:"count"})

vc2['percent']=vc2['count'].apply(lambda x:x*100/sum(vc2['count']))

vc2=vc2.sort_values(col)



trace1=go.Bar(x=vc1[col],y=vc1['count'],name="TV Show")

trace2=go.Bar(x=vc2[col],y=vc2['count'],name="Movie")

data1=[trace1,trace2]

layout=go.Layout(title="Content added over the months",legend=dict(x=0.1,y=1.1,orientation='h'))

fig=go.Figure(data1,layout=layout)

fig.show()
small=d1[['title','release_year']]

small.sort_values('release_year',ascending=True)[:15]
small=d2[['title','release_year']]

small.sort_values('release_year',ascending=True)[:15]
import plotly.figure_factory as ff

x1=d2['duration'].astype(float)

fig=ff.create_distplot([x1],['b'],bin_size=0.7,curve_type='normal')

fig.update_layout(title_text="Dist Plot with normal distribution")

fig.show()
col="season_count"



s1=d1[col].value_counts().reset_index()

s1=s1.rename(columns={"index":col,col:"count"})

s1['percent']=s1['count'].apply(lambda x:x*100/sum(s1['count']))

s1=s1.sort_values(col)





trace1=go.Bar(x=s1[col],y=s1['count'],name="TV Shows")

layout=go.Layout(title="Seasons",legend=dict(x=0.1,y=1.1))

data1=[trace1]

fig=go.Figure(data1,layout)

fig.show()
col = "rating"



vc1 = d1[col].value_counts().reset_index()

vc1 = vc1.rename(columns = {col : "count", "index" : col})

vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))

vc1 = vc1.sort_values(col)



vc2 = d2[col].value_counts().reset_index()

vc2 = vc2.rename(columns = {col : "count", "index" : col})

vc2['percent'] = vc2['count'].apply(lambda x : 100*x/sum(vc2['count']))

vc2 = vc2.sort_values(col)



trace1 = go.Bar(x=vc1[col], y=vc1["count"], name="TV Shows", marker=dict(color="#a678de"))

trace2 = go.Bar(x=vc2[col], y=vc2["count"], name="Movies", marker=dict(color="#6ad49b"))

data1 = [trace1, trace2]

layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data1, layout=layout)

fig.show()
from collections import Counter

col = "listed_in"

categories = ", ".join(d2['listed_in']).split(", ")

counter_list = Counter(categories).most_common(25)

labels = [_[0] for _ in counter_list][::-1]

values = [_[1] for _ in counter_list][::-1]

trace1 = go.Bar(y=labels, x=values, orientation="h", marker=dict(color="#a678de"))



data1 = [trace1]

layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data1, layout=layout)

fig.show()
cols="director"

a1=d2[d2['country']=="India"]

categories = ", ".join(a1['director'].fillna("")).split(", ")

counter_list = Counter(categories).most_common(12)

labels = [_[0] for _ in counter_list][::-1]

values = [_[1] for _ in counter_list][::-1]

trace1 = go.Bar(y=labels, x=values, orientation="h", marker=dict(color="#a678de"))



data1 = [trace1]

layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data1, layout=layout)

fig.show()
cols="director"

a1=d2[d2['country']=="United States"]

categories = ", ".join(a1['director'].fillna("")).split(", ")

counter_list = Counter(categories).most_common(12)

labels = [_[0] for _ in counter_list][::-1]

values = [_[1] for _ in counter_list][::-1]

trace1 = go.Bar(y=labels, x=values, orientation="h", marker=dict(color="#a678de"))



data1 = [trace1]

layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data1, layout=layout)

fig.show()