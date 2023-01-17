import pandas as pd

import numpy as np

from time import time 



from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import CountVectorizer

movie_df = pd.read_csv('../input/tmdb-10000-dataset/Movie.csv', index_col = "Unnamed: 0") 

display(movie_df.shape,

        movie_df.head()

       )
# Quick overview

movie_df.info()
# Get an overview of data

cols = movie_df.columns 

for c in cols:

    display(pd.DataFrame(movie_df[c].value_counts()).transpose())
# Describe these numbers in more detail 

movie_df.loc[:,'Popularity':'Vote Average'].describe() #.astype("int") 
# Check any duplicated rows 

movie_df.duplicated().any()

# Find items that are duplicated 

dup = movie_df[movie_df.duplicated()]

dup.head()
# Find all duplicated overviews 

movie_df[movie_df.Overview.isin(dup.Overview)]



# Drop duplicates 

movie_df.drop_duplicates(keep='first',inplace=True) 
# Along the columns --  “axis 0” represents rows and “axis 1” represents columns.

movie_df.isnull().any(axis=0)
# Along the rows - performs function along columns

movie_df[movie_df.isnull().any(axis='columns')]
# movie_df[movie_df['Vote Average']>7].sort_values(by = 'Popularity', ascending = False)[50:100]#[movie_df['Release Date']>str(2019)]
# Remove NaN by dropping rows

movie_df.dropna(inplace=True)



# Also drop overviews that are unknown
# If you don't reset index, it can cause confusion when selecting indices of movies 

movie_df = movie_df[movie_df['Original Language']=='en'].reset_index(drop=True)
# Look at text as a Bag of words

movie_df['BagOfWords'] = movie_df[['Title','Overview']].apply(lambda col: ', '.join(col.astype(str)),axis=1)

movie_df['BagOfWords']
# Generating the counts of words for vocabulary list 

count = CountVectorizer()

vocab_counts = count.fit_transform(movie_df['BagOfWords'])

display(vocab_counts.shape, 

# calculate shape cosine similarity matrix

cosine_similarity(vocab_counts,vocab_counts).shape)
# Add stopwords, get rid of brackets, commas, lexi, etc. 

import nltk

from nltk.corpus import stopwords

def reduce_vocab(df):

    # upper to lower character

    df = df.apply(lambda x: " ".join(x.lower() for x in x.split()))

    # clear punctuations

    df = df.str.replace('[^\w\s]','')

    # clear numbers

    df = df.str.replace('\d','')

    # get rid of stopwords 

    sw = stopwords.words('english')

    df = df.apply(lambda x: " ".join(x for x in x.split() if x not in sw))

    # Faster way to get rid of Stopwords

    from collections import Counter

    sw = stopwords.words('english')

    sw_dict = Counter(sw)

    df = df.apply(lambda x: ' '.join([x for x in x.split() if x not in sw_dict]))

    #    #Delete rare characters 

    sil = pd.Series(' '.join(df).split()).value_counts()[-1000:]

    # Faster way to get rid of 

    df = df.apply(lambda x: " ".join(x for x in x.split() if x not in sil))

#     Add spaces between text

    from textblob import Word

    df = df.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) 

    return df
# # Generating the counts of words for vocabulary list 

count = CountVectorizer()

vocab_counts1 = count.fit_transform(reduce_vocab(movie_df['BagOfWords']))



# movie_df['BagOfWords'] = movie_df.loc[:,'Original Language':].apply(lambda col: ', '.join(col.astype(str)),axis=1)



c1 = cosine_similarity(vocab_counts1,vocab_counts1)

vocab_counts1.shape
# # Pearson's measure using cosine 

# Take the mean for each movie

display(vocab_counts1.mean(axis=1).shape)

vocab_counts2 = vocab_counts1 - vocab_counts1.mean(axis=1)

 

c2 = cosine_similarity(vocab_counts2,vocab_counts2)
# function that takes in movie title as input and returns the top 10 recommended movies

def content_recommender(movie_df, Title, top_n=5, cosine_sim = c1):

          

    # gettin the indices of the movie that matches the title

    idx = [Title.lower() in str(t).lower() for t in pd.Series(movie_df.Title)]



    # creating a Series with the similarity scores in descending order

    sorted_scores = pd.DataFrame(cosine_sim[np.where(idx)].T).sort_values(by = 0, ascending = False)



    # getting the indexes of the 10 most similar movies

    top_n_movies = list(movie_df.loc[sorted_scores.iloc[1:(10+1)].index].Title)

#     movie_df.Title[list(sorted_scores.iloc[1:(top_n+1)].index)]

    

     

    return  display("Selects first movie only (but here are exact titles for what you might be looking for):",

                    movie_df.Title[idx], top_n_movies)
mlist = ['avengers', 'rings', 'Bad Boys']

movie = mlist[0]

content_recommender(movie_df, movie,5)
mlist = ['avengers', 'rings', 'Bad Boys']

movie = mlist[2]

content_recommender(movie_df, movie,5,c2)