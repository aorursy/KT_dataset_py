# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df1=pd.read_csv('../input/the-movies-dataset/credits.csv')

df2=pd.read_csv('../input/the-movies-dataset/movies_metadata.csv')





df2.head()
df2['id']
df1['id']
df2 = df2[df2.id!='1997-08-20']

df2 = df2[df2.id!='2012-09-29']

df2 = df2[df2.id!='2014-01-01']

df1.id
df2['id'] = df2['id'].astype(int)
df2.id
df2=df2.merge(df1, on='id')
df2.columns
df2['overview'].head()
#Import TfIdfVectorizer from scikit-learn

from sklearn.feature_extraction.text import TfidfVectorizer



#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'

tfidf = TfidfVectorizer(stop_words='english')



#Replace NaN with an empty string

df2['overview'] = df2['overview'].fillna('')



from sklearn.metrics.pairwise import linear_kernel







import random

ran = random.randint(25000, 30000)

df3 = df2.head(ran)

tfidf_matrix = tfidf.fit_transform(df3['overview'])



#Output the shape of tfidf_matrix

tfidf_matrix.shape
cosine_sim =  linear_kernel(tfidf_matrix, tfidf_matrix, True)
indices = pd.Series(df3.index, index=df3['title']).drop_duplicates()
def get_recommendations(title, cosine_sim=cosine_sim):

    # Get the index of the movie that matches the title

    idx = indices[title]



    # Get the pairwsie similarity scores of all movies with that movie

    sim_scores = list(enumerate(cosine_sim[idx]))



    # Sort the movies based on the similarity scores

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)



    # Get the scores of the 10 most similar movies

    sim_scores = sim_scores[1:11]



    # Get the movie indices

    movie_indices = [i[0] for i in sim_scores]



    # Return the top 10 most similar movies

    return df3['title'].iloc[movie_indices]
get_recommendations('Assassins')
get_recommendations('Carrington')
df4=pd.read_csv('../input/the-movies-dataset/keywords.csv')
df4.head()
df2 = df2.merge(df4, on='id')
df2.head()

from ast import literal_eval



features = ['cast', 'crew', 'keywords', 'genres']

for feature in features:

    df2[feature] = df2[feature].apply(literal_eval)
df2.head()
#function to return director name from crew column

def get_director(x):

    for i in x:

        if i['job'] == 'Director':

            return i['name']

    return np.nan
#function to get names of genres, cast and keywords from the data

def get_list(x):

    if isinstance(x, list):

        names = [i['name'] for i in x]

        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.

        if len(names) > 3:

            names = names[:3]

        return names



    #Return empty list in case of missing/malformed data

    return []
# Define new director, cast, genres and keywords features that are in a suitable form.

df2['director'] = df2['crew'].apply(get_director)



features = ['cast', 'keywords', 'genres']

for feature in features:

    df2[feature] = df2[feature].apply(get_list)
# Print the new features of the first 3 films

df2[['title', 'cast', 'director', 'keywords', 'genres']].head(3)
# Function to convert all strings to lower case and strip names of spaces

def clean_data(x):

    if isinstance(x, list):

        return [str.lower(i.replace(" ", "")) for i in x]

    else:

        #Check if director exists. If not, return empty string

        if isinstance(x, str):

            return str.lower(x.replace(" ", ""))

        else:

            return ''
# Apply clean_data function to the features.

features = ['cast', 'keywords', 'director', 'genres']



for feature in features:

    df2[feature] = df2[feature].apply(clean_data)
def create_soup(x):

    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

df2['soup'] = df2.apply(create_soup, axis=1)

df2['soup']
# Import CountVectorizer and create the count matrix

from sklearn.feature_extraction.text import CountVectorizer



count = CountVectorizer(stop_words='english')



# again we take a part of dataset as our kernel cannot handle all the dataset together.

 

df5 = df2['soup'].head(20000)



count_matrix = count.fit_transform(df5)

 
from sklearn.metrics.pairwise import cosine_similarity



cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
# we just reset the index of main data

df2 = df2.reset_index()

indices = pd.Series(df2.index, index=df2['title'])
#finally calling the get_recommendation fucntion

get_recommendations('Assassins', cosine_sim2)
get_recommendations('Jumanji', cosine_sim2)