import numpy as np

import pandas as pd
credits = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv')

movies_df = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv')
print(credits.shape)

print(movies_df.shape)
credits.head()
movies_df.shape
movies_df.head()
# making the column name same for merging the two datasets

credits = credits.rename(index = str,columns = {'movie_id':'id'})
# merging the datasets

movies = movies_df.merge(credits,on = 'id')
movies.head()
# drop the unnecessary columns

movies = movies.drop(['homepage','title_x','title_y','status','production_countries'],1)
movies.info()
movies['overview'].head(1)


from sklearn.feature_extraction.text import TfidfVectorizer
tfv = TfidfVectorizer(stop_words='english',min_df = 3,max_features=None, strip_accents='unicode',

                     analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,3))

# ngram_range will take the combination of 1 to 3 different kind of words

# stopwords is used to ignore words like 'the','a','is', etc types of words

# strip_accents, analyzer and token_pattern are used to remove unnecessary letters like '!',',',''

# meaning special symbols

# fill the null values.

movies['overview'] = movies['overview'].fillna('')
tfv_matrix = tfv.fit_transform(movies['overview'])
tfv_matrix
tfv_matrix.shape
from sklearn.metrics.pairwise import sigmoid_kernel

sig = sigmoid_kernel(tfv_matrix,tfv_matrix)


sig[0]
# Lets give indexes to the movie

indices = pd.Series(movies.index, index = movies['original_title']).drop_duplicates()
indices
# index of the maze runner

indices['The Maze Runner']
def recommend(title, sig = sig):

    # Here title is in text and we will get the respective index assigned to it

    index = indices[title]

    # lets calculate the sigmoid scores of this movie with every other movie.

    sigmoid_scores = list(enumerate(sig[index]))

    # Sort it descending so that top correlated can be listed on top

    sigmoid_scores = sorted(sigmoid_scores,key = lambda x: x[1], reverse = True)

    # We will take only top 5 recommendations

    sigmoid_scores = sigmoid_scores[1:6]

    

    movie_indices = [i[0] for i in sigmoid_scores]

    

    return movies['original_title'].iloc[movie_indices]
print(' Top Recommendations for this Movie is : ')

recommend('The Dark Knight')
print('Recommendation for the Fast and furious 5 : \n',recommend('Fast Five'))
# lets import libraries

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity
# dataset

movies.head()
cv = CountVectorizer(stop_words='english')

cv_matrix = cv.fit_transform(movies['overview'])
cv_matrix.shape
cv_matrix
cos_similarity = cosine_similarity(cv_matrix)
cos_similarity
## Again lets take the indexes

indexes = pd.Series(movies.index, index = movies['original_title']).drop_duplicates()
def recommend_m2(title,cos = cos_similarity):

    index = indexes[title]

    

    similarity_score = list(enumerate(cos_similarity[index]))

    

    similarity_score = sorted(similarity_score, reverse = True,key = lambda x:x[1])

    

    similarity_score = similarity_score[1:6]

    

    movie_indexes = [i[0] for i in similarity_score]

    

    return movies['original_title'].iloc[movie_indexes]

    

    
recommend_m2('Fast Five')
recommend_m2('The Maze Runner')
recommend_m2('Star Wars')