import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from wordcloud import WordCloud
movieSet = pd.read_csv('../input/tmdb-movies-dataset/tmdb_movies_data.csv')

movieSet.head()
movieSet.describe()
movieSet.describe(include=np.object)
movieSet.info()
movieSet.duplicated().sum()
movieSet.drop_duplicates(inplace=True)
movieSet['release_date'] = pd.to_datetime(movieSet['release_date'])
movieSet.head()
movieSet.drop(['imdb_id','homepage','tagline','overview','budget_adj','revenue_adj'],axis=1,inplace=True)
movieSet.head()
movieSet.isnull().sum()
movieSet[movieSet['budget'] == 0].shape
movieSet[movieSet['revenue'] == 0].shape
movieSet['budget'].replace(0,np.nan,inplace=True)
movieSet['revenue'].replace(0,np.nan,inplace=True)
movieSet['runtime'].replace(0,np.nan,inplace=True)
movieSet['id'].nunique()
movieSet.set_index('id',inplace=True)
movieSet.columns.shape
text = ','.join(movieSet['keywords'].str.cat(sep='|').split('|'))
wc = WordCloud(max_words=50,background_color='white').generate(text)

plt.figure(figsize=(15,10))

plt.imshow(wc)

plt.show()
plt.figure(figsize=(18,10))

sns.countplot(movieSet['release_year'])

sns.set_style("darkgrid")

plt.xticks(rotation = 90)

plt.show()
movieSet['profit'] = movieSet['revenue'] - movieSet['budget']
movieSet.head()
dd = movieSet[(movieSet.profit == movieSet['profit'].max()) | (movieSet.profit == movieSet['profit'].min())][['original_title','profit']]

sns.barplot(dd['original_title'],dd['profit'])

plt.show()
db = movieSet[(movieSet.budget == movieSet['budget'].max()) | (movieSet.budget == movieSet['budget'].min())][['original_title','budget']]

db['scaled_budget'] = MinMaxScaler().fit_transform(db['budget'].values.reshape(-1,1))

db

sns.barplot(db['original_title'],db['scaled_budget'])

plt.xticks(rotation = 90)

plt.show()
dr = movieSet[(movieSet.revenue == movieSet['revenue'].max()) | (movieSet.revenue == movieSet['revenue'].min())][['original_title','revenue']]

dr['scaled_revenue'] = MinMaxScaler().fit_transform(dr['revenue'].values.reshape(-1,1))

dr

sns.barplot(dr['original_title'],dr['scaled_revenue'])

plt.xticks(rotation = 90)

plt.show()
dd = movieSet[(movieSet.runtime == movieSet['runtime'].max()) | (movieSet.runtime == movieSet['runtime'].min())][['original_title','runtime']]

sns.barplot(dd['original_title'],dd['runtime'])

plt.xticks(rotation = 90)

plt.show()
plt.figure(figsize=(20,10))

movieSet.groupby('release_year')['profit'].mean().plot(kind='bar',color='pink')

plt.show()
dd = movieSet.sort_values('popularity',ascending=False)[['popularity','runtime']].head()

sns.lineplot(dd['runtime'],dd['popularity'])

plt.show()
plt.figure(figsize=(15,6))

movieSet.groupby('release_year')['runtime'].mean().plot(color='grey')

plt.show()
sns.scatterplot(movieSet['budget'],movieSet['revenue'])

sns.set_style("whitegrid")

plt.show()
sns.regplot('runtime','revenue',data=movieSet,color='violet')

plt.show()
sns.lmplot('budget','popularity',data=movieSet)

sns.set_style("darkgrid")

plt.show()
sns.scatterplot(movieSet['popularity'],movieSet['runtime'],color='c')

plt.show()
sns.regplot(movieSet['popularity'],movieSet['profit'],color='g')

plt.show()
movieSet['release_date'].dt.month.value_counts().plot.bar(color='magenta')

plt.show()
movieSet.groupby(movieSet['release_date'].dt.month)['revenue'].mean().plot.bar(color='lime')

plt.show()
dg = movieSet['genres'].str.get_dummies(sep='|')

dg.head()
dg[dg.columns].apply(lambda x: sum(x.values)).plot.pie(figsize=(20,10),autopct='%1.1f%%',explode=[0.12]*len(dg.columns))

plt.show()
dy = movieSet.groupby('release_year')['genres'].apply(lambda x: x.str.cat(sep='|')).apply(lambda x: x.split('|'))

dy.head()
dy2 = dy.apply(lambda x:stats.mode(x)).reset_index()

dy2.head()
dy2['genre_name'] = dy2['genres'].apply(lambda x: ''.join(x[0]))

dy2['freq'] = dy2['genres'].apply(lambda x: ''.join(x[1].astype(str)))

dy2['freq'] = dy2['freq'].astype(int)

dy2.head()
dy3 = dy2.sort_values('freq').head(10)

fig,ax = plt.subplots(figsize=(15,10))

ax.barh(dy3.release_year,dy3.freq,color='c')

ax.set_yticks(dy3.release_year)

ax.set_yticklabels(dy3.genre_name)



ax2= ax.twinx()

ax2.barh(dy3.release_year,dy3.freq,color='c')

ax2.set_yticks(dy3.release_year)

ax2.set_yticklabels(dy3.release_year)

plt.show()
dc = movieSet['production_companies'].str.get_dummies(sep='|')

dc.head()
dpc = dc[dc.columns].apply(lambda x: sum(x.values))

dpc.head()
dpc.sort_values(0,ascending=False).head(20).plot.pie(autopct='%1.1f%%',frame=True,shadow=True)

plt.show()
movieSet['director'].value_counts().head(20).plot.barh(figsize=(18,10),color='brown')

plt.show()
movieSet.sort_values('revenue',ascending=False).head(3).describe()