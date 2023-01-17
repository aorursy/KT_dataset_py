# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/top50spotify2019/top50.csv',encoding='ISO-8859-1')
data.head()
s_songs = data.drop('Unnamed: 0',1)
s_songs.head()
x= s_songs.sort_values(by='Popularity',ascending=False)
top_10_data = x.head(10)
top_10_data
top_10_data.Genre
top_10_data['Artist.Name']
import seaborn as sns
sns.set(rc={'figure.figsize':(10,10)})
grid  = sns.pairplot(top_10_data, x_vars=['Genre','Artist.Name'], y_vars=(['Popularity']),height=5)
for ax in grid.axes.flat[:2]: ## for rotating x labels
    ax.tick_params(axis='x', labelrotation=90)
#plt.xticks(rotation='vertical')
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
sns.countplot(x='Artist.Name',data=s_songs,color='c')
plt.xlabel('Artist.Name in given data',fontsize=12)
plt.ylabel('count of song sung',fontsize=12)
plt.xticks(rotation='vertical')
plt.title('Artist.Name and no.of song count',fontsize=12)
plt.show()
plt.figure(figsize=(12,8))
sns.countplot(x='Genre',data=s_songs,color='c')
plt.xlabel('Genre in given data',fontsize=12)
plt.ylabel('count of song under gener',fontsize=12)
plt.xticks(rotation='vertical')
plt.title('Genre and no.of song count',fontsize=12)
plt.show()
Genre_counts = top_10_data["Genre"].value_counts()
Genre_counts_index = Genre_counts.index
Genre_counts, Genre_counts_index = zip(*sorted(zip(Genre_counts, Genre_counts_index)))
%matplotlib inline
import plotly
import plotly.graph_objects as go
plotly.offline.init_notebook_mode(connected=True)
fig = go.Figure(
    go.Treemap(
        labels = ["Number of Tracks by Genre of the Spotify Top 10 Music List"] + list(Genre_counts_index),
        parents = [""] + ["Number of Tracks by Genre of the Spotify Top 10 Music List"] * len(Genre_counts_index),
        values = [0] + list(Genre_counts),
        textposition='middle center', # center the text
        textinfo = "label+percent parent", # show label and its percentage among the whole treemap
        textfont=dict(
            size=15 # adjust small text to larger text
        )
    )
)
fig.show()

data[data.Popularity > 90].groupby(by=['Genre','Artist.Name']).agg('count')['Track.Name'].sort_values(ascending=False)[:10]
#plot data
fig, ax = plt.subplots(figsize=(15,7))
data[data.Popularity > 90].groupby(by=['Genre','Artist.Name']).agg('count')['Track.Name'].sort_values(ascending=False)[:10].plot(ax=ax)
# plot data
fig, ax = plt.subplots(figsize=(15,7))
# use unstack()
data[data.Popularity > 90].groupby(by=['Genre','Artist.Name']).agg('count')['Track.Name'].sort_values(ascending=False)[:10].unstack().plot(ax=ax)
sns.set(rc={'figure.figsize':(8,5)})
sns.heatmap(pd.crosstab(top_10_data.Genre, top_10_data['Artist.Name']))