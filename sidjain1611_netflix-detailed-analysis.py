import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/netflix-shows/netflix_titles_nov_2019.csv')
df.head()
df.describe(include='all')
df.info()
plt.figure(1, figsize=(15, 7))

plt.title("Country with maximum content creation")

sns.countplot(x = "country", order=df['country'].value_counts().index[0:15] ,data=df,palette='Accent')
plt.figure(1, figsize=(15, 7))

plt.title("Frequency")

sns.countplot(x = "rating", order=df['rating'].value_counts().index[0:15] ,data=df,palette='Accent')
df['rating'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(17,7))
plt.figure(1, figsize=(15, 7))

plt.title("Frequency")

sns.countplot(x = "release_year", order=df['release_year'].value_counts().index[0:15] ,data=df,palette='Accent')
import plotly.graph_objects as go

d1 = df[df["type"] == "TV Show"]

d2 = df[df["type"] == "Movie"]



col = "release_year"



vc1 = d1[col].value_counts().reset_index()

vc1 = vc1.rename(columns = {col : "count", "index" : col})

vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))

vc1 = vc1.sort_values(col)



vc2 = d2[col].value_counts().reset_index()

vc2 = vc2.rename(columns = {col : "count", "index" : col})

vc2['percent'] = vc2['count'].apply(lambda x : 100*x/sum(vc2['count']))

vc2 = vc2.sort_values(col)



trace1 = go.Scatter(

                    x=vc1[col], 

                    y=vc1["count"], 

                    name="TV Shows", 

                    marker=dict(color = 'rgb(249, 6, 6)',

                             line=dict(color='rgb(0,0,0)',width=1.5)))



trace2 = go.Scatter(

                    x=vc2[col], 

                    y=vc2["count"], 

                    name="Movies", 

                    marker= dict(color = 'rgb(26, 118, 255)',

                              line=dict(color='rgb(0,0,0)',width=1.5)))

layout = go.Layout(hovermode= 'closest', title = 'Content added over the years' , xaxis = dict(title = 'Year'), yaxis = dict(title = 'Count'),template= "plotly_dark")

fig = go.Figure(data = [trace1, trace2], layout=layout)

fig.show()
plt.figure(1, figsize=(4, 4))

plt.title("TV v/s Movies")

sns.countplot(x = "type", order=df['type'].value_counts().index[0:15] ,data=df,palette='Accent')
df['type'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(17,7))
movie=df[df['type']=='Movie']

tv=df[df['type']=='TV Show']
plt.figure(1, figsize=(20, 7))

plt.title("Director with most movies")

sns.countplot(x = "director", order=movie['director'].value_counts().index[0:10] ,data=movie,palette='Accent')
from collections import Counter

col = "listed_in"

categories = ", ".join(movie['listed_in']).split(", ")

counter_list = Counter(categories).most_common(50)

labels = [_[0] for _ in counter_list][::-1]

values = [_[1] for _ in counter_list][::-1]

trace1 = go.Bar(y=labels, x=values, orientation="h", name="Movie")

data = [trace1]

layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
dur=[]

for i in movie['duration']:

    dur.append(int(i.strip('min')))

plt.figure(1, figsize=(20, 7))

plt.title("Comparing the length of Movies")

sns.distplot(dur,rug=True, rug_kws={"color": "g"},

                   kde_kws={"color": "k", "lw": 3, "label": "KDE"},

                  hist_kws={"histtype": "step", "linewidth": 3,

                            "alpha": 1, "color": "g"})
plt.figure(1, figsize=(20, 7))

plt.title("Comparing the length of Movies")

sns.countplot(x = "duration", order=movie['duration'].value_counts().index[0:15] ,data=df,palette='Accent')
plt.figure(1, figsize=(20, 7))

plt.title("Director with most movies")

sns.countplot(x = "director", order=tv['director'].value_counts().index[0:10] ,data=tv,palette='Accent')
from collections import Counter

col = "listed_in"

categories = ", ".join(tv['listed_in']).split(", ")

counter_list = Counter(categories).most_common(50)

labels = [_[0] for _ in counter_list][::-1]

values = [_[1] for _ in counter_list][::-1]

trace1 = go.Bar(y=labels, x=values, orientation="h", name="TV Shows")

data = [trace1]

layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
dur=[]

for i in tv['duration']:

    if 'Seasons' in i:

        dur.append(int(i.strip('Seasons')))

    else:

        dur.append(int(i.strip('Season')))

plt.figure(1, figsize=(20, 7))

plt.title("Comparing the length of TV")

sns.distplot(dur,rug=True, rug_kws={"color": "g"},

                   kde_kws={"color": "k", "lw": 3, "label": "KDE"},

                  hist_kws={"histtype": "step", "linewidth": 3,

                            "alpha": 1, "color": "g"})
tv['dur']=dur

top=tv.nlargest(15,['dur'])

plt.figure(1, figsize=(20, 7))

sns.barplot(x="title", y="dur", data=top, ci="sd")