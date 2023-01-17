import pandas as pd
import csv
import numpy
import seaborn as sns
import matplotlib.pyplot as plt
#read data into a datagrame
title_ratings=pd.read_csv("../input/movie-ratings-dataset/title.ratings.tsv/title.ratings.tsv", sep='\t')
title_ratings.head()
#number of rows in the dataframe
title_ratings.shape
#check if we have unique ratings for the titles
title_ratings.groupby(['tconst'], as_index=False).count()
title_basics=pd.read_csv("../input/movie-ratings-dataset/title.basics.tsv/title.basics.tsv", sep='\t')
title_basics=title_basics.drop_duplicates()
title_basics=title_basics[['titleType','tconst','primaryTitle', 'originalTitle', 'startYear']]
title_basics=title_basics[title_basics.titleType=='movie']
title_basics=title_basics[title_basics.startYear.apply(lambda x: str(x).isnumeric())]
title_basics.head()
title_basics.shape
grouped=title_basics.groupby(['primaryTitle', 'startYear'], as_index=False).count()
grouped.head()

ratings_and_titles=pd.merge(title_ratings.set_index('tconst'), title_basics.set_index('tconst'), left_index=True, right_index=True, how='inner')
ratings_and_titles=ratings_and_titles.drop_duplicates()
ratings_and_titles.head()
ratings_and_titles.shape
netflix_titles=pd.read_csv("../input/netflix-shows/netflix_titles.csv", index_col="show_id")
netflix_titles=netflix_titles.dropna(subset=['release_year'])
netflix_titles.release_year=netflix_titles.release_year.astype(numpy.int64)
ratings_and_titles=ratings_and_titles[ratings_and_titles.startYear.apply(lambda x: str(x).isnumeric())]
ratings_and_titles.startYear=ratings_and_titles.startYear.astype(numpy.int64)
netflix_titles['title']=netflix_titles['title'].str.lower()
ratings_and_titles['originalTitle']=ratings_and_titles['originalTitle'].str.lower()
ratings_and_titles['primaryTitle']=ratings_and_titles['primaryTitle'].str.lower()
##subset movies
netflix_titles=netflix_titles[netflix_titles.type=='Movie']
netflix_titles.shape
netflix_titles_rating=pd.merge(netflix_titles, ratings_and_titles, left_on=['title', 'release_year'], right_on=['primaryTitle', 'startYear'], how='inner')
netflix_titles_rating.sort_values(by=['averageRating', 'numVotes'], inplace=True, ascending=False)
#look at titles where we have more than 2000 votes
netflix_titles_rating_2000=netflix_titles_rating[netflix_titles_rating.numVotes>2000]
netflix_titles_rating_2000.head(10)
plt.figure(figsize=(20, 6))
sns.distplot(netflix_titles_rating['averageRating']);
plt.figure(figsize=(20, 6))
sns.distplot(netflix_titles_rating['numVotes']);
netflix_titles_rating_2000.head(10)['title']
plt.figure(figsize=(20, 6))
chart=sns.countplot(x="country", data=netflix_titles_rating_2000.head(100), order = netflix_titles_rating_2000.head(100)['country'].value_counts().index)
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)
from itertools import chain

# return list from series of comma-separated strings
def chainer(s):
    return list(chain.from_iterable(s.str.split(',')))

# calculate lengths of splits
lens = netflix_titles_rating_2000.head(100)['listed_in'].str.split(',').map(len)

# create new dataframe, repeating or chaining as appropriate
res = pd.DataFrame({'title': numpy.repeat(netflix_titles_rating_2000.head(100)['title'], lens),
                    'listed_in': chainer(netflix_titles_rating_2000.head(100)['listed_in']),
                    })
res['listed_in']=res['listed_in'].str.strip()

print(res)
top_genres=res['listed_in'].value_counts()
top_genres
plt.figure(figsize=(20, 6))
chart=sns.countplot(x="listed_in", data=res, order = res['listed_in'].value_counts().index)
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)
#save plot
chart.figure.savefig("pop_genres.png")
#save to the file
#netflix_titles_rating.to_csv('netflix_titles_rating_movies.csv')
