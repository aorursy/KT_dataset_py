# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt # data visualization library

import re

from wordcloud import WordCloud, STOPWORDS #world cloud



from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules    



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
movies=pd.read_csv("/kaggle/input/movielens/movies.csv")

movies.head()
tags=pd.read_csv("/kaggle/input/movielens/tags.csv")

tags.head()
ratings=pd.read_csv("/kaggle/input/movielens/ratings.csv")

ratings.head()
links=pd.read_csv("/kaggle/input/movielens/links.csv")

links.head()
movies.head()
movies.shape
movies.columns
movies.info()
#check null values

movies.isnull().sum()
#number of unique movies 

movies.movieId.nunique()
#number of unique title 

movies.title.nunique()
ratings.head()
ratings.shape
ratings.dtypes
ratings.info()
ratings.isnull().sum()
#number of user which are rating the movies

ratings.userId.nunique()
#number of movies which are rated by users

ratings.movieId.nunique()
ratings.rating.describe()
ratings['rating'].hist()
ratings.rating.min()
ratings.rating.max()
tags.head()
tags.isnull().sum()
def findnull(data):

        for each in data:

            print(each.isnull().sum())

    

data=[movies, tags, ratings, links]

findnull(data)
tags[tags.tag.isnull()]
tags.tag.dropna(axis=0, inplace=True)
tags.tag.isnull().sum()


movies[movies.duplicated(['title',"genres"])]
movies[movies.title=="Offside (2006)"]
movies.drop_duplicates(subset =["title"],inplace=True)
movies[movies.title=="Offside (2006)"]
#check if there are any deleted unique movie titles

movies.title.nunique()
#check if there are any user rated same movies more than one time

ratings[ratings.duplicated(["userId","movieId"])]
#film türlerini ayırmak 

#movies.genres=movies["genres"].str.split("|",n=10, expand = False) 



genre_labels = set()

for s in movies['genres'].str.split('|').values:

    genre_labels = genre_labels.union(set(s))
genre_labels=list(genre_labels)

genre_labels
def count_word(df, ref_col, liste):

    

    keyword_count = dict()

    

    for each in liste: 

        keyword_count[each] = 0

        

    for liste_keywords in df[ref_col].str.split('|'):

        if type(liste_keywords) == float and pd.isnull(liste_keywords): 

            continue

        for x in liste_keywords: 

            

            if pd.notnull(x): 

                keyword_count[x] += 1



    # convert the dictionary in a list to sort the keywords  by frequency

    keyword_occurences = []

    

    for k,v in keyword_count.items():

        keyword_occurences.append([k,v])

    keyword_occurences.sort(key = lambda x:x[1], reverse = True)

    

    

    return keyword_occurences, keyword_count
count_word(movies, 'genres', genre_labels)
drama_movies=movies[movies.genres.str.contains("Drama")]

drama_movies.head()

print("total number of drama movies {}".format(len(drama_movies)))
ratings.columns

ratings.groupby("userId").rating.agg([np.size, np.mean, np.median])
ratings.groupby("movieId").rating.agg([np.size, np.mean, np.median])
del ratings["timestamp"]
movies.columns
merged_movies_ranking=movies.merge(ratings, on="movieId", how="inner")

merged_movies_ranking.head()
merged_movies_ranking.isnull().sum()
# ratinglerin medyanına göre filmlerin sıralaması

merged_movies_ranking.groupby("title").rating.agg(np.median).reset_index().sort_values(by="rating", ascending=False)
rating_median=merged_movies_ranking.groupby("title").rating.agg(np.median).reset_index().sort_values(by="rating", ascending=False)

rating_median.head()
#4.5den büyük ratinge sahip filmler

rating_median[rating_median.rating>4.5]
#1den küçük ratinge sahip filmler

rating_median[rating_median.rating<1]
#en çok sayıda oy kullanan userlar

merged_movies_ranking.userId.value_counts().to_frame().reset_index().rename(columns={'index': 'userId','userId': 'count'})
#en çok sayıda oy alan filmler

merged_movies_ranking.title.value_counts().to_frame().reset_index().rename(columns={'index': 'title','title': 'count'})

#merged_movies_ranking.groupby('title').size().sort_values(ascending=False)
#filmlerin yıllarını column olarak oluştur

movies["year"]=movies['title'].str.extract('.*\((.*)\).*',expand = False)

movies.head()




def set_genres_matrix(genres):

    movie_genres_matrix = []

    for x in genre_labels:

        if (x in genres.split('|')):

            movie_genres_matrix.append(1)

        else:

            movie_genres_matrix.append(0) 

    return movie_genres_matrix

    

movies['genresMatrix'] = movies.apply(lambda x: list(set_genres_matrix(x['genres'])), axis=1)



            



            
movies
movies_matrix=movies.genres.str.get_dummies()

movies_matrix.head()
frequent_itemsets = apriori(movies_matrix, min_support = 0.02, use_colnames=True)

frequent_itemsets
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

frequent_itemsets
#en düşük supporta sahip olan

frequent_itemsets[frequent_itemsets.support==frequent_itemsets.support.min()]
#en yüksek supporta sahip olan ikili

frequent_itemsets[frequent_itemsets.support==frequent_itemsets[frequent_itemsets.length==2].support.max()]
frequent_itemsets[ frequent_itemsets['itemsets'] == {'Comedy', 'Drama'} ]
rules=association_rules(frequent_itemsets, metric="confidence", min_threshold=0.35)

rules
rules[  (rules["lift"]>1.5)  &  (rules["confidence"]>0.5) ]
rules[ (rules["confidence"]>0.5) ]
merged_movies_ranking.head()
movie_recommend=ratings.groupby(["userId","movieId"]).movieId.size().unstack().reset_index().fillna(0).set_index("userId")

movie_recommend
def encode_units(x):

    if x<=0:

        return 0

    else:

        return 1
apriori(movie_recommend.applymap(encode_units), min_support = 0.2, use_colnames=True)

frequent_itemsets2['length'] = frequent_itemsets2['itemsets'].apply(lambda x: len(x))