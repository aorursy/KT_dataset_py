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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode,iplot

%matplotlib inline
net_data= pd.read_csv("../input/netflix-shows/netflix_titles_nov_2019.csv")
net_data.head()
net_data.index
net_data.columns
net_data['date_added'] = pd.to_datetime(net_data['date_added'])

net_data['year_added'] = net_data['date_added'].dt.year

net_data['month_added'] = net_data['date_added'].dt.month
net_data.head()
net_data['season_count'] = net_data.apply(lambda x:x['duration'].split("")[0] if "season" in x['duration'] else "" ,axis=1)



net_data['duration'] = net_data.apply(lambda x:x['duration'].split(" ")[0] if "season" not in ['duration'] else "" , axis =1)
data =net_data.groupby(['country','release_year'])

data.first()
net_data['release_year'].value_counts().head(10).plot.bar()
net_data['country'].value_counts().head(10).plot.bar()
net_data['type'].value_counts()
sns.countplot(x='type',data=net_data)
net_data['rating'].value_counts()
plt.figure(figsize=[10,10])

sns.countplot(x='rating',data=net_data, hue="type")
d1 = net_data[net_data['type']=="TV Show"]

d2 = net_data[net_data['type']=="Movie"]



col="year_added"



t1 = d1[col].value_counts().reset_index()

t1 = t1.rename(columns={col:"count","index":col})

t1['percent']=t1['count'].apply(lambda x:100*x/sum(t1['count']))

t1=t1.sort_values(col)



t2 = d2[col].value_counts().reset_index()

t2 = t2.rename(columns={col:"count","index":col})

t2['percent']=t1['count'].apply(lambda x:100*x/sum(t1['count']))

t2=t2.sort_values(col)



trace1 = go.Scatter(x=t1[col],y=t1["count"],name='TV Show', marker=dict(color="#a678de"))

trace2 = go.Scatter(x=t2[col],y=t2["count"],name='Movie', marker=dict(color="#6ad49b"))

data = [trace1,trace2]

layout = go.Layout(title="content over the year",legend=dict(x=0.1,y=1.1,orientation='h'))

fig = go.Figure(data,layout=layout)

fig.show()
s =net_data.sort_values("release_year",ascending=True)

s = s[s['duration']!=" "]

s[['title',"release_year"]][:10]
tag = "india"

net_data['relevent']=net_data['country'].fillna("").apply(lambda x : 1 if tag.lower() in x.lower() else 0 )

small = net_data[net_data['relevent']==1]

small[["title","director","cast","country","date_added","release_year","rating"]]
small = net_data.sort_values("release_year",ascending=True)

small = small[small['season_count']!=" "]

small[["title","release_year"]][:15]
col="release_year"



t1 = d1[col].value_counts().reset_index()

t1 = t1.rename(columns={col:"count","index":col})

t1['percent']=t1['count'].apply(lambda x:100*x/sum(t1['count']))

t1=t1.sort_values(col)



t2 = d2[col].value_counts().reset_index()

t2 = t2.rename(columns={col:"count","index":col})

t2['percent']=t1['count'].apply(lambda x:100*x/sum(t1['count']))

t2=t2.sort_values(col)



trace1 = go.Scatter(x=t1[col],y=t1["count"],name='TV Show', marker=dict(color="#a678de"))

trace2 = go.Scatter(x=t2[col],y=t2["count"],name='Movie', marker=dict(color="#6ad49b"))

data = [trace1,trace2]

layout = go.Layout(title="content over the year",legend=dict(x=0.1,y=1.1,orientation='h'))

fig = go.Figure(data,layout=layout)

fig.show()
tag = "Stand-Up Comedy"

net_data["relevent"] = net_data["listed_in"].fillna("").apply(lambda x:1if tag.lower() in x.lower() else 0 )

small=net_data[net_data['relevent']==1]

small[small["country"]=='India'][["title","country","release_year"]].head(10)