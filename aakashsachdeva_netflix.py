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
import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

import os
print(os.listdir('../input/netflix-shows'))
df=pd.read_csv('../input/netflix-shows/netflix_titles.csv')
df.head()
df.info()
df.duplicated().value_counts()
df.show_id.duplicated().value_counts()
df.type.unique()
sns.countplot(df.type)
df['type'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(15,7))
df[df.duplicated(subset=['type','title','director','cast','country','date_added','release_year','rating','duration','listed_in','description'])]
df[df.title=='Sarkar']
df.drop_duplicates(subset=['type','title','director','cast','country','date_added','release_year','rating','duration','listed_in','description'],inplace=True)
df[df['title']=='Sarkar']
df[df['title'].duplicated()]
df['director'].unique()
plt.figure(figsize=(20,5))

plt.title('Directors with most movies or tv shows')

sns.countplot(df['director'],order=df['director'].value_counts().index[0:10])
plt.figure(figsize=(20,5))

plt.title('Directors with most movies or tv shows')

sns.countplot(df.cast,order=df.cast.value_counts().index[0:10])
df.country.nunique()
plt.figure(figsize=(15,5))

plt.title('Top countries of movies or tv shows')

sns.countplot(df.country,order=df.country.value_counts().index[0:10])
plt.figure(figsize=(15,5))

plt.title('Frequency of movies released in a year')

sns.countplot(df.release_year,order=df.release_year.value_counts().index[0:15])
df.rating.unique()


plt.figure(figsize=(12,7))

sns.countplot(df.rating)
plt.figure(figsize=(35,5))

sns.countplot(df.listed_in,order=df.listed_in.value_counts().index[0:15])
plt.figure(figsize=(15,5))

plt.title('Top countries of movies')

sns.countplot(df.loc[df['type']=='Movie']['country'],order=df.country.value_counts().index[0:10])
plt.figure(figsize=(15,5))

plt.title('Top countries of tv shows')

sns.countplot(df.loc[df['type']=='TV Show']['country'],order=df.country.value_counts().index[0:10])
from collections import Counter

categories=", ".join(df.loc[df['type']=='Movie']['listed_in']).split(", ")

count_list=Counter(categories).most_common(50)
labels=[_[0] for _ in count_list]

values=[_[1] for _ in count_list]
plt.figure(figsize=(15,7))

plt.title('Content of movies added')

sns.barplot(x=values,y=labels)
cat=", ".join(df.loc[df['type']=='TV Show']['listed_in']).split(", ")

count_list_tv=Counter(cat).most_common(50)

labels_tv=[_[0] for _ in count_list_tv]

values_tv=[_[1] for _ in count_list_tv]

plt.figure(figsize=(15,7))

plt.title('Content of TV Shows added')

sns.barplot(x=values_tv,y=labels_tv)
cat_actor=", ".join(df.loc[df['type']=='Movie']['cast'].fillna("")).split(", ")

count_list_actor=Counter(cat_actor).most_common(50)

count_list_actor=[_ for _ in count_list_actor if "" != _[0]]

labels_actor=[_[0] for _ in count_list_actor]

values_actor=[_[1] for _ in count_list_actor]

plt.figure(figsize=(15,15))

plt.title('Top actors in movies')

sns.barplot(x=values_actor,y=labels_actor)
cat_actor_tv=", ".join(df.loc[df['type']=='TV Show']['cast'].fillna("")).split(", ")

count_list_actor_tv=Counter(cat_actor_tv).most_common(50)

count_list_actor_tv=[_ for _ in count_list_actor_tv if "" != _[0]]

labels_actor_tv=[_[0] for _ in count_list_actor_tv]

values_actor_tv=[_[1] for _ in count_list_actor_tv]

plt.figure(figsize=(15,15))

plt.title('Top actors in movies')

sns.barplot(x=values_actor_tv,y=labels_actor_tv)
categories
df['date_added']=pd.to_datetime(df['date_added'])

df['year_added']=df.date_added.dt.year
d1=df[df['type']=='TV Show']

d2=df[df['type']=='Movie']



vc1=d1['year_added'].value_counts().reset_index()

vc1=vc1.rename(columns={'year_added':"count","index":'year_added'})

vc1
vc1['percent']=vc1['count'].apply(lambda x: 100*x/sum(vc1['count']))

vc1
vc1=vc1.sort_values('year_added')

vc1
vc2=d2['year_added'].value_counts().reset_index()

vc2=vc2.rename(columns={'year_added':'count','index':'year_added'})

vc2['percent']=vc2['count'].apply(lambda x: 100*x/sum(vc2['count']))

vc2.sort_values('year_added')

vc2
type(vc1['year_added'])
vc1.drop(vc1[vc1['year_added']==2020].index,inplace=True)

vc2.drop(vc2[vc2['year_added']==2020].index,inplace=True)
import plotly.graph_objs as go



trace1 = go.Scatter(

                    x=vc1['year_added'], 

                    y=vc1["count"], 

                    name="TV Shows", 

                    marker=dict(color = 'rgb(249, 6, 6)',

                             line=dict(color='rgb(0,0,0)',width=1.5)))



trace2 = go.Scatter(

                    x=vc2['year_added'], 

                    y=vc2["count"], 

                    name="Movies", 

                    marker= dict(color = 'rgb(26, 118, 255)',

                              line=dict(color='rgb(0,0,0)',width=1.5)))

fig = go.Figure(data = [trace1, trace2])

fig.show()
temp_df1=df['release_year'].value_counts().reset_index()

temp_df1
trace1=go.Bar(x=temp_df1['index'],y=temp_df1['release_year'])

fig=go.Figure(data=[trace1])

fig.show()