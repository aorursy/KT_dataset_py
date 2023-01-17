import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
% matplotlib inline
import seaborn as sns
sns.set_style('darkgrid')

df=pd.read_csv('../input/tmdb-movies.csv')
df.head(3)
df.info()
df.describe()
#dropping the columns that I will not use this analysis
df.drop(['id','imdb_id','cast','homepage','tagline','keywords','overview','runtime','production_companies','release_date','budget_adj','revenue_adj'], axis=1, inplace=True)
#checking if there are null values
df.isnull().sum().any()
#deleting null values
df.dropna(inplace=True)
#checking if there are duplicate values
print(df.duplicated().sum())
#erasing duplicate values
df.drop_duplicates(inplace=True)
#erasing rows with zero values
df = df[(df != 0).all(1)]
#split the genres rows in mulple rows
g = df['genres'].str.split('|').apply(pd.Series, 1).stack()
g.index = g.index.droplevel(-1)
g.name = 'genres'
del df['genres']
df= df.join(g)
df
df.head(2)
df.hist(figsize=(15,15));
pd.plotting.scatter_matrix(df, figsize=(15,15));
df['genres'].value_counts().plot(kind='pie',figsize= (15,15));
df.plot(x='vote_average',y='popularity',kind='scatter',figsize=(18,10));
df['budget'].hist(figsize=(8,8));
df.groupby(['vote_average']).mean().revenue
#check unique values
df['genres'].unique()
# verify the revenue mean related with genres column
df.groupby('genres').mean().revenue
gen_rev =df.groupby('genres').mean()['revenue']

ind = np.arange(len(gen_rev))  # the x locations for the groups
width = 0.35  

plt.subplots(figsize=(18,10))
gen_bars =plt.bar(ind, gen_rev, width, color='g', alpha=.7)
plt.ylabel('Revenue',size=14) # title and labels
plt.xlabel('Genres',size=14)
plt.title('Performance of Revenue by genres',size=18)
locations = ind + width / 2  # xtick locations
labels = ['Action', 'Adventure', 'Syfy', 'Thriller', 'Fantasy',
       'Crime', 'Western', 'Drama', 'Family', 'Animation', 'Comedy',
       'Mystery', 'Romance', 'War', 'History', 'Music', 'Horror',
       'Document.', 'TV Movie', 'Foreign']  # xtick labels
plt.xticks(locations, labels);
df.groupby('genres').mean().vote_average
gen_vot =df.groupby('genres').mean()['vote_average']

ind = np.arange(len(gen_rev))  # the x locations for the groups
width = 0.35  

plt.subplots(figsize=(18,10))
gen_bars =plt.bar(ind, gen_vot, width, color='orange', alpha=.7)
#adv_bars =plt.bar(ind, adv, width, color='b', alpha=.7, label="Adventure")
plt.ylabel('Vote Average',size=14) # title and labels
plt.xlabel('Genres',size=14)
plt.title('Performance of Vote Average by genres',size=18)
locations = ind + width / 2  # xtick locations
labels = ['Action', 'Adventure', 'Syfy', 'Thriller', 'Fantasy',
       'Crime', 'Western', 'Drama', 'Family', 'Animation', 'Comedy',
       'Mystery', 'Romance', 'War', 'History', 'Music', 'Horror',
       'Document.', 'TV Movie', 'Foreign']  # xtick labels
plt.xticks(locations, labels);
df.groupby(['genres']).mean().popularity
gen_pop =df.groupby('genres').mean()['popularity']

ind = np.arange(len(gen_pop))  # the x locations for the groups
width = 0.35  

plt.subplots(figsize=(18,10))
gen_bars =plt.bar(ind, gen_pop, width, color='r', alpha=.7)
#adv_bars =plt.bar(ind, adv, width, color='b', alpha=.7, label="Adventure")
plt.ylabel('Vote Average',size=14) # title and labels
plt.xlabel('Genres',size=14)
plt.title('Performance of Vote Average by genres',size=18)
locations = ind + width / 2  # xtick locations
labels = ['Action', 'Adventure', 'Syfy', 'Thriller', 'Fantasy',
       'Crime', 'Western', 'Drama', 'Family', 'Animation', 'Comedy',
       'Mystery', 'Romance', 'War', 'History', 'Music', 'Horror',
       'Document.', 'TV Movie', 'Foreign']  # xtick labels
plt.xticks(locations, labels)
df.groupby(['genres','release_year']).mean().popularity
plt.subplots(figsize=(14,12))
plt.title('Performance of popularity by genres')
sns.pointplot(x='release_year', y='popularity', hue='genres', data=df);
# Bin edges that will be used to "cut" the data into groups
bin_edges= [1960,1965,1975,1985,1995, 2005,2015]

# Labels for the six decades groups        
bin_years= [1965,1975,1985,1995, 2005,2015]

# Creates release_yearby10 column
df['release_yearby10']= pd.cut(df['release_year'],bin_edges,labels=bin_years) 
df # Check for successful creation of this column
plt.subplots(figsize=(20,12))
plt.title('Performance Of Popularity By Genres', fontsize = 30)
ax = sns.pointplot(x='release_yearby10', y='popularity', hue='genres', 
                   data=df);
plt.legend(frameon=False, loc='upper center', ncol=5, fontsize=15)
ax.set_xlabel("Release Year By 10", fontsize=25)
ax.set_ylabel("Popularity", fontsize=25)

plt.show()
