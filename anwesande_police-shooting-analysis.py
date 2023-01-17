# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv")



df.head()
df.info()
missingp=df.isna().sum()*100/df.shape[0]

missingp
df.dropna(inplace=True)
df.shape
df['month'] = pd.to_datetime(df['date']).dt.month

df['year'] = pd.to_datetime(df['date']).dt.year

df["day"]=pd.to_datetime(df['date']).dt.day
df.info()
import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

plt.figure(num=None, figsize=(10, 6), dpi=120, facecolor='w', edgecolor='k')

g = sns.FacetGrid(df, col="race", margin_titles=True)

g.map(plt.hist, "gender")



# Add a title to the figure

g.fig.suptitle("race and gender")

plt.hist(data=df,x="manner_of_death")
g = sns.FacetGrid(df, col="gender", margin_titles=True)

g.map(plt.hist, "manner_of_death")
q = df[df['armed']=='unarmed']['race']

fig = px.histogram(q,x='race',color='race')

fig.show()
g = sns.FacetGrid(df, col='signs_of_mental_illness', margin_titles=True)

g.map(plt.hist, "manner_of_death")
df.race.value_counts().plot(kind='bar')
df.gender.value_counts().plot(kind="bar")
unarmed=df[df['armed']=='unarmed']

unarmed.race.value_counts().plot(kind="bar")
df.body_camera.value_counts().plot(kind="bar")
df.threat_level.value_counts().plot(kind="bar")
t=df[df["body_camera"]==False]

t.threat_level.value_counts().plot(kind="bar")
df.columns

df.threat_level.unique()
k=df[df["flee"]=="Not fleeing"]

l=k[k["armed"]=="unarmed"]

o=l[l["threat_level"]=="undetermined"]

q=o[o["body_camera"]==False]

fig = px.histogram(q,x='race',color='race')

fig.show()


sns.set_style("whitegrid", {'axes.grid' : False})

plt.figure(num=None, figsize=(10, 6), dpi=120, facecolor='w', edgecolor='k')

plt.hist(data=df,x="state",bins=50)



plt.tight_layout()

plt.xticks(rotation=90)

plt.show()
df.state.value_counts().plot(kind="bar")

plt.figure(num=None, figsize=(10, 6), dpi=120, facecolor='w', edgecolor='k')
import plotly.express as px

 

shootout_by_states = df['state'].value_counts()[:10]

shootout_by_states = pd.DataFrame(shootout_by_states)

shootout_by_states=shootout_by_states.reset_index()#to arrange in descending order

fig = px.pie(shootout_by_states, values='state', names='index', color_discrete_sequence=px.colors.sequential.RdBu)

fig.show()
shootout_by_states

df.year.value_counts().plot(kind="bar")
df[df.year==2020].date
df[df.year!=2020].year.value_counts()
l=df[df.day<=15]

j=l[l.month<=6]

j.year.value_counts().plot(kind="bar")
g = sns.FacetGrid(df, col='year', margin_titles=True)

g.map(plt.hist, "race")
g = sns.FacetGrid(df, col="month", margin_titles=True)

g.map(plt.hist, "race")

plt.figure(figsize=(120,120))
df.city.value_counts()[:5]
d=df[["year","city"]]

d["death"]=1

d=d.groupby(["year","city"]).sum()

d= d.reset_index()

d=d.sort_values(by=['year','death'],ascending=False)

d
t=pd.DataFrame(columns=["year","city","death"])

for i in range(2015,2021,1):

  

    t=t.append(d[d.year==i].iloc[0:1,:])



t    