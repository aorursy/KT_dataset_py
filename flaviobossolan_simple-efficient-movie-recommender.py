import numpy as np 

import pandas as pd
#importing movie metadata

meta= pd.read_csv("../input/movies_metadata.csv")

meta= meta[['id', 'original_title', 'original_language']]

meta= meta.rename(columns={'id':'movieId'})

meta = meta[meta['original_language']== 'en'] #just want movies in English

meta.head()
#importing movie ratings

ratings= pd.read_csv("../input/ratings.csv")

ratings= ratings[['userId', 'movieId', 'rating']]

# taking a 1MM sample because it can take too long to pivot data later on

ratings=ratings.head(1000000)
#convert data types before merging

meta.movieId =pd.to_numeric(meta.movieId, errors='coerce')

ratings.movieId = pd.to_numeric(ratings.movieId, errors= 'coerce')
#create a single dataset merging the previous 2

data= pd.merge(ratings, meta, on='movieId', how='inner')

data.head()
#movie matrix so that I can use the recommender function later

matrix= data.pivot_table(index='userId', columns='original_title', values='rating')

matrix.head()
# A simple way to compute Pearson Correlation

def pearsonR(s1, s2):

    s1_c = s1-s1.mean()

    s2_c= s2-s2.mean()

    return np.sum(s1_c*s2_c) / np.sqrt(np.sum(s1_c**2)* np.sum(s2_c**2))



# A function to make N recommendations based on Pearson Correlation.

# The parameters here are: movie name, matrix name and number of recommendations.

def recommend(movie, M, n):

    reviews=[]

    for title in M.columns:

        if title == movie:

            continue

        cor= pearsonR(M[movie], M[title])

        if np.isnan(cor):

            continue

        else:

            reviews.append((title, cor))

            

    reviews.sort(key= lambda tup: tup[1], reverse=True)

    return reviews[:n]
# List with movies I have watched:

watched = ['Jarhead', 'Die Hard', 'Donnie Darko', 'Frida', 'Men in Black II' ]
recs = recommend('Jarhead', matrix, 10) #top 10 recommendations

recs
# I call it Trimmed_rec_list for the lack of a better term...

trimmed_rec_list = [r for r in recs if r[0] not in watched]

# This is the subtle difference that doesn't get on my nerves...

trimmed_rec_list