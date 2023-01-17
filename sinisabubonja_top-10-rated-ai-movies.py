import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
!ls ../input/
!cat ../input/movie.csv | wc -l

!cat ../input/tag.csv | wc -l

!cat ../input/rating.csv | wc -l

!cat ../input/genome_scores.csv | wc -l

!cat ../input/genome_tags.csv | wc -l

!cat ../input/link.csv | wc -l
movies = pd.read_csv('../input/movie.csv')

movies.head()
movies.isnull().any()
tags = pd.read_csv('../input/tag.csv')

tags.head()
tags.isnull().any()
tags[tags.tag.isnull()]
tags=tags.dropna()

tags.shape
tags.isnull().any()
ratings = pd.read_csv('../input/rating.csv')

ratings.head()
ratings.isnull().any()
tag_rel=pd.read_csv('../input/genome_scores.csv')

tag_rel.head()
tag_rel.isnull().any()
tag_id=pd.read_csv('../input/genome_tags.csv')

tag_id.head()
tag_id.isnull().any()
links=pd.read_csv('../input/link.csv')

links.head()
links.isnull().any()
links[links.tmdbId.isnull()]
del links['tmdbId']

links.isnull().any()
links.head()
is_ai=tags['tag'].str.contains('artificial intelligence', na=False)

tags[is_ai].head()
ai_movies=tags[is_ai][['movieId','tag']]

ai_movies.head()
ai_movies.shape
del ai_movies['tag']

ai_movies.index=np.arange(0,521)

ai_movies.head()
ai_movies.duplicated().any()
ai_movies.drop_duplicates('movieId', inplace=True)

ai_movies.shape
ai_tagId=tag_id[tag_id.tag=='artificial intelligence']

ai_tagId.head()
ai_rel=tag_rel[tag_rel.tagId==77]

ai_rel.head()
del ai_rel['tagId']

ai_rel.head()
ai_movies=pd.merge(ai_movies,movies[['movieId','title']],on='movieId')

ai_movies
ai_movies=ai_movies.merge(ai_rel,on='movieId')

ai_movies.head()
ai_movies.shape
41-37
four_movies=list(set(tags[is_ai][['movieId']].iloc[:, 0].tolist())-set(tag_rel[tag_rel.tagId==77][['movieId']].iloc[:, 0].tolist()))

movies[movies.movieId.isin(four_movies)]
ai_ratings=ratings[['movieId','rating']].groupby('movieId',as_index=False).mean().rename(columns={'rating':'averageRating'})

ai_ratings.head()
ai_userRatings=ratings[['movieId','userId']].groupby('movieId',as_index=False).count().rename(columns={'userId':'numberOfUserRatings'})

ai_userRatings.head()
ai_movies=ai_movies.merge(ai_ratings, on='movieId', how='left')

ai_movies.head()
ai_movies=ai_movies.merge(ai_userRatings, on='movieId', how='left')

ai_movies
ai_movies[['relevance','averageRating','numberOfUserRatings']].describe()
ai_movies[['relevance','averageRating','numberOfUserRatings']].corr()
%matplotlib inline



sns.pairplot(ai_movies[['relevance','averageRating','numberOfUserRatings']])
ai_movies.boxplot(column='averageRating',figsize=(15,10))
ai_movies.boxplot(column='numberOfUserRatings', figsize=(15,20))
mlist=[0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]

Clist=np.arange(10,52240,10).tolist()

table=[]



for m in mlist:

    for C in Clist:

        ai_movies['rating']=(C*(0.5+(m-0.5)*ai_movies['relevance'])+ai_movies['numberOfUserRatings']*ai_movies['averageRating'])/(C+ai_movies['numberOfUserRatings'])

        table.append([m,C,np.corrcoef(ai_movies['rating'],ai_movies['relevance'])[1,0],np.corrcoef(ai_movies['rating'],ai_movies['averageRating'])[1,0],np.corrcoef(ai_movies['rating'],ai_movies['numberOfUserRatings'])[1,0]])



corr=pd.DataFrame(table,columns=['m','C','rating_relevance','rating_averageRating','rating_numberOfUserRatings'])
corr[(np.abs(corr.rating_relevance-corr.rating_numberOfUserRatings)<0.0005)].sort_values('rating_averageRating', ascending=False)
m=4.00

C=350

ai_movies['rating']=(C*(0.5+(m-0.5)*ai_movies['relevance'])+ai_movies['numberOfUserRatings']*ai_movies['averageRating'])/(C+ai_movies['numberOfUserRatings'])
ai_movies.head()
sns.heatmap(ai_movies[['relevance','averageRating','numberOfUserRatings','rating']].corr(), annot=True, xticklabels=ai_movies[['relevance','averageRating','numberOfUserRatings','rating']].columns.values, yticklabels=ai_movies[['relevance','averageRating','numberOfUserRatings','rating']].columns.values)
ai_movies=ai_movies.sort_values('rating',ascending=False)

ai_movies.head()
ai_movies=ai_movies.merge(links,on='movieId')

ai_movies.index=np.arange(1,38)

ai_movies.head()
ai_movies['linkImdb']='http://www.imdb.com/title/tt'+ai_movies['imdbId'].astype(str)

del ai_movies['movieId']

del ai_movies['imdbId']

ai_movies
top_10_ai=ai_movies[:10]

del top_10_ai['relevance']

del top_10_ai['averageRating']

del top_10_ai['numberOfUserRatings']

del top_10_ai['rating']

top_10_ai
top_10_75th_ai=ai_movies.sort_values('relevance',ascending=False)

top_10_75th_ai=top_10_75th_ai[(top_10_75th_ai.relevance>0.785000) & (top_10_75th_ai.numberOfUserRatings<=11196)].sort_values('rating',ascending=False)[:10]

del top_10_75th_ai['relevance']

del top_10_75th_ai['averageRating']

del top_10_75th_ai['numberOfUserRatings']

del top_10_75th_ai['rating']

top_10_75th_ai.index=np.arange(1,11)

top_10_75th_ai