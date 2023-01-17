import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df1 = pd.read_csv('../input/spotifydata-19212020/data.csv')
df1.head()
df1.tail()
df1.shape
df1.dtypes
df1.describe()
# categorical_features
categorical_features = [features for features in df1 if df1[features].dtypes=='object']
categorical_features
for features in df1[categorical_features]:
    print(features ,len(df1[features].value_counts()))
df1.name.duplicated().value_counts()

# 36969 names are duplicates
duplicates = df1[df1.duplicated(['name'])]
duplicates.head(50)
# continuous_features
continuous_features = [features for features in df1 if df1[features].dtypes==('float64')]
continuous_features
# dist plot for continuous features
for features in df1[continuous_features]:
    sns.distplot(df1[features],color='violet')
    plt.show()

# boxplot for continuous features
for features in df1[continuous_features]:
    sns.boxplot(df1[features],color='red')
    plt.show()
# discrete_features
discrete_features = [features for features in df1 if df1[features].dtypes==('int64')]
discrete_features
for features in df1[discrete_features]:
    print(features ,len(df1[features].value_counts()))
 
# count plot for discrete features
fig , axes = plt.subplots(2,2,figsize=(15,10))
sns.countplot(df1['mode'],ax=axes[0,0])
sns.countplot(df1['explicit'],ax=axes[0,1])
sns.countplot(df1['key'],ax=axes[1,0])
sns.distplot(df1['popularity'],ax=axes[1,1],color='g')
plt.show()
sns.barplot(df1['explicit'] , df1['popularity'])
plt.show()
sns.barplot(df1['key'] , df1['popularity'])
plt.show()
sns.barplot(df1['mode'] , df1['popularity'])
plt.show()
sns.scatterplot(df1['year'] , df1['popularity'])
plt.show()
fig , axes = plt.subplots(3,3,figsize=(15,15))
sns.lineplot(df1['year'],df1['energy'],color='g',ax=axes[0,0])
sns.lineplot(df1['year'],df1['loudness'],color='g',ax=axes[0,1])
sns.lineplot(df1['year'],df1['tempo'],color='g',ax=axes[0,2])
sns.lineplot(df1['year'],df1['acousticness'],color='r',ax=axes[1,0])
sns.lineplot(df1['year'],df1['liveness'],color='r',ax=axes[1,1])
sns.lineplot(df1['year'],df1['instrumentalness'],color='r',ax=axes[1,2])
sns.lineplot(df1['year'],df1['valence'],color='b',ax=axes[2,0])
sns.lineplot(df1['year'],df1['danceability'],color='b',ax=axes[2,1])
sns.lineplot(df1['year'],df1['speechiness'],color='b',ax=axes[2,2])
plt.show()
artist_popularity = (df1.groupby('artists').sum()['popularity'].sort_values(ascending=False).head(10))
# Top 10 artist

ax = sns.barplot(artist_popularity.index, artist_popularity)
ax.set_title('Top Artists with Popularity')
ax.set_ylabel('Popularity')
ax.set_xlabel('Tracks')
plt.xticks(rotation = 90)
df1[['artists','energy','acousticness']].groupby('artists').mean().sort_values(by='energy',ascending=False).head(10)
df1[['name','acousticness','popularity']].groupby('name').mean().sort_values(by='popularity',ascending=False).head(10)
# With few acception , mostly popular songs have low acousticness
# correlation matrix
plt.figure(figsize=(16, 8))
sns.heatmap(df1.corr(),annot=True)
plt.show
