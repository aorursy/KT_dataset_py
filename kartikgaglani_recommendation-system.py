# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import Pandas
import pandas as pd

# Loading Data sets
full_url='/kaggle/input/tmdb-5000-credits/tmdb_5000_credits.csv'
full_url1='/kaggle/input/recomend-data/tmdb_5000_movies.csv'
credits = pd.read_csv(full_url)
movies=pd.read_csv(full_url1)
# Printing 1st 5 elements of credits dataset
credits.head()
# Printing 1st 5 elements of movies dataset
movies.head()
# Printing the shapes of both the datasets
print("Credits:",credits.shape)
print("Movies:",movies.shape)
# Renaming the column of credits data set
credits_renamed=credits.rename(index=str,columns={'movie_id':'id'})
credits_renamed.head()
# Merging both data sets
merge=movies.merge(credits_renamed,on='id')
merge.head()
# Dropping unnecessary columns 
cleaned=merge.drop(columns=['homepage','title_x','title_y','status','production_countries'])
cleaned.head()
cleaned['overview'].head()
#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english',ngram_range=(1,3),min_df=3,analyzer='word')

#Replace NaN with an empty string
cleaned['overview'] = cleaned['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(cleaned['overview'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print(cosine_sim.shape)
print(cosine_sim[1])
#Construct a reverse map of indices and movie titles
indices = pd.Series(cleaned.index, index=cleaned['original_title']).drop_duplicates()

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
    return cleaned['original_title'].iloc[movie_indices]
# Getting the recommendation
get_recommendations('Avatar')