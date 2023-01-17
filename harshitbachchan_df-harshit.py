# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('../input/spotifydata-19212020/data.csv'):

    for filename in filenames:

        print(os.path.join(input/spotifydata-19212020, data.csv))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv('../input/spotifydata-19212020/data.csv')

df.head()
df.shape
df.info()
categorical_features = [features for features in df if df[features].dtypes=='O']

categorical_features
for features in df[categorical_features]:

    print(features ,len(df[features].value_counts()))

    

# No. of values of each variable in Categorical features. 
# Songs are repeated
continuous_features = [features for features in df if df[features].dtypes==('float64')]

continuous_features
discrete_features = [features for features in df if df[features].dtypes==('int64')]

discrete_features
for features in df[discrete_features]:

    print(features ,len(df[features].value_counts()))

    

# No. of values of each variable in Discrete features. 
fig , axes = plt.subplots(2,2,figsize=(15,10))

sns.countplot(df['mode'],ax=axes[0,0])

sns.countplot(df['explicit'],ax=axes[0,1])

sns.countplot(df['key'],ax=axes[1,0])

sns.distplot(df['popularity'],ax=axes[1,1],color='violet')

plt.show()
for features in df[categorical_features]:

    print(features , len(df[features].value_counts()))

    

# No. of Values of each variable in Categorical features
for features in df[continuous_features]:

    sns.distplot(df[features],color='green')

    plt.show()
for features in df[continuous_features]:

    print(features , df[features].skew())
#highly skewed features --> instrumentalness , liveness , loudness , speechiness
for features in df[continuous_features]:

    sns.boxplot(df[features],color='red')

    plt.show()
for features in df[continuous_features]:

    if(df[features]==0).any():

        print(features , len(df[df[features]==0][features]))

# outliers --> instrumentalness , liveness , loudness , speechiness , tempo
sns.barplot(df['explicit'] , df['popularity'])

plt.show()
sns.barplot(df['key'] , df['popularity'])

plt.show()
sns.barplot(df['mode'] , df['popularity'])

plt.show()
sns.scatterplot(df['year'] , df['popularity'])

plt.show()

# With time the popularity of songs get increased.
fig , axes = plt.subplots(3,3,figsize=(15,15))

sns.lineplot(df['year'],df['energy'],color='g',ax=axes[0,0])

sns.lineplot(df['year'],df['loudness'],color='g',ax=axes[0,1])

sns.lineplot(df['year'],df['tempo'],color='g',ax=axes[0,2])

sns.lineplot(df['year'],df['acousticness'],color='r',ax=axes[1,0])

sns.lineplot(df['year'],df['liveness'],color='r',ax=axes[1,1])

sns.lineplot(df['year'],df['instrumentalness'],color='r',ax=axes[1,2])

sns.lineplot(df['year'],df['valence'],color='b',ax=axes[2,0])

sns.lineplot(df['year'],df['danceability'],color='b',ax=axes[2,1])

sns.lineplot(df['year'],df['speechiness'],color='b',ax=axes[2,2])

plt.show()
artist_popularity = (df.groupby('artists').sum()['popularity'].sort_values(ascending=False).head(10))

# Top 10 artist



ax = sns.barplot(artist_popularity.index, artist_popularity)

ax.set_title('Top Artists with Popularity')

ax.set_ylabel('Popularity')

ax.set_xlabel('Tracks')

plt.xticks(rotation = 90)
df[['artists','energy','acousticness']].groupby('artists').mean().sort_values(by='energy',ascending=False).head(10)
# mostly artists with high energy have less acousticness
df[['name','acousticness','popularity']].groupby('name').mean().sort_values(by='popularity',ascending=False).head(10)

# With few acception , mostly popular songs have low acousticness
cmap = sns.diverging_palette(330, 40, sep=40, as_cmap=True)

fig , axes = plt.subplots(1,1,figsize=(14,8))

sns.heatmap(df[['acousticness','danceability','duration_ms','energy','explicit','instrumentalness','key','liveness','loudness','mode','popularity','speechiness','tempo','valence','year']].corr(),annot=True, cmap=cmap)

plt.show()