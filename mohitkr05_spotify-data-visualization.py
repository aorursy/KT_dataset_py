# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # this is used for the plot the graph 

import seaborn as sns # used for plot interactive graph.

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import warnings

warnings.filterwarnings('ignore')

from pylab import rcParams

# figure size in inches

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/spotify-dataset-19212020-160k-tracks/data.csv',index_col=0)

df_genre = pd.read_csv('/kaggle/input/spotify-dataset-19212020-160k-tracks/data_by_genres.csv')

df_artist = pd.read_csv('/kaggle/input/spotify-dataset-19212020-160k-tracks/data_by_artist.csv')

df_year = pd.read_csv('/kaggle/input/spotify-dataset-19212020-160k-tracks/data_by_year.csv')

df_genre2 = pd.read_csv('/kaggle/input/spotify-dataset-19212020-160k-tracks/data_w_genres.csv')

df_super_genres = pd.read_json('/kaggle/input/spotify-dataset-19212020-160k-tracks/super_genres.json')
df_genre.head()
df.info()
df.isna().sum()
plt.figure(figsize=(16, 8))

sns.distplot(df.popularity,bins=20)
plt.figure(figsize=(16, 8))

ax = sns.jointplot(x=df.popularity,y=df["duration_ms"],data=df)


df["duration_ms"].describe()
df["duration_minutes"] = df["duration_ms"] / 60000
df[df["duration_ms"] == df["duration_ms"].max()]
df["popularity"].describe()
df[df["popularity"] > 95]["name"]
plt.figure(figsize=(16, 8))

sns.set(style="whitegrid")

corr = df.corr()

sns.heatmap(corr,annot=True,cmap="coolwarm")
sns.clustermap(corr,cmap="coolwarm")
plt.figure(figsize=(16, 8))

sns.set(style="whitegrid")

x = df.groupby("name")["popularity"].mean().sort_values(ascending=False).head(10)

ax = sns.barplot(x.index, x)

ax.set_title('Top Tracks with Popularity')

ax.set_ylabel('Popularity')

ax.set_xlabel('Tracks')

plt.xticks(rotation = 90)
plt.figure(figsize=(16, 8))

sns.set(style="whitegrid")

x = df.groupby("artists")["popularity"].sum().sort_values(ascending=False).head(10)

ax = sns.barplot(x.index, x)

ax.set_title('Top Artists with Popularity')

ax.set_ylabel('Popularity')

ax.set_xlabel('Artists')

plt.xticks(rotation = 90)
plt.figure(figsize=(16, 8))

sns.set(style="whitegrid")

x = df.groupby("year")["id"].count()

ax= sns.lineplot(x.index,x)

ax.set_title('Count of Tracks added')

ax.set_ylabel('Count')

ax.set_xlabel('Year')
plt.figure(figsize=(16, 8))

sns.set(style="whitegrid")

columns = ["acousticness","danceability","energy","speechiness","liveness","valence"]

for col in columns:

    x = df.groupby("year")[col].mean()

    ax= sns.lineplot(x=x.index,y=x,label=col)

ax.set_title('Audio characteristics over year')

ax.set_ylabel('Measure')

ax.set_xlabel('Year')
plt.figure(figsize=(16, 8))

sns.set(style="whitegrid")

columns = ["loudness"]

for col in columns:

    x = df.groupby("year")[col].mean()

    ax= sns.lineplot(x=x.index,y=x,label=col)

ax.set_title('tempo')

ax.set_ylabel('Count')

ax.set_xlabel('Year')
plt.figure(figsize=(16, 8))

sns.set(style="whitegrid")

columns = ["tempo"]

for col in columns:

    x = df.groupby("year")[col].mean()

    ax= sns.lineplot(x=x.index,y=x,label=col)

ax.set_title('tempo')

ax.set_ylabel('Count')

ax.set_xlabel('Year')
df.duration_ms.sum()/(1000*60*60*24*365)
df.columns
# Popularity of Genres with respect to the various features

plt.figure(figsize=(16, 8))

sns.set(style="whitegrid")

cols = ["valence","popularity","acousticness","instrumentalness","speechiness","danceability" ]

sns.pairplot(df_genre[cols], height = 2.5 )

plt.show();