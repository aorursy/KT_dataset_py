import pandas as pd 

import numpy as np 

import seaborn as sns

import matplotlib.pyplot as plt

import string

import re

movie_info_1=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')

movie_info_2=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')
#Get familier with structure and data of movie_info_1

movie_info_1.columns
#Get familier with structure and data of movie_info_2

movie_info_2.columns
#Merge the both Dataframe

movie_data = movie_info_2.merge(movie_info_1,left_on='id', right_on='movie_id')

#Get the shape of the data

movie_data.shape
movie_data.head(3)
plt.figure(figsize=(10,5))

barplot_data = pd.DataFrame (movie_data['original_language'].value_counts() )

barplot_data.reset_index(inplace = True)

barplot_data.columns = ['language' , 'counts']

plt.bar( x = 'language' , height = 'counts' ,data = barplot_data.head(10))

plt.xlabel('Original Language')

plt.ylabel('Movies Counts') 

plt.title('Language-Wise Movie Distribution')

plt.show()
#Movies popularity index wise histogram .

plt.figure(figsize=(10,5))

sns.distplot(movie_data['popularity'])
#Check the status of the movie whether its release or not .

plt.figure(figsize=(10,5))

barplot_2 = pd.DataFrame (movie_data['status'].value_counts() )

barplot_2.reset_index(inplace = True)

barplot_2.columns = ['status' , 'counts']

plt.bar( x = 'status' , height = 'counts' ,data = barplot_2.head(10))

plt.xlabel('status')

plt.ylabel('Movies Counts') 

plt.title('Status-Wise Movie Distribution')

plt.show()
#Lowercasing the Dataframe column

movie_data['overview'] = movie_data['overview'].apply(lambda x:str(x).lower())



#Remove the Numbers from Overview columns

movie_data['overview'] = movie_data['overview'].apply(lambda x:re.sub(r"\d+", "", x))



#Remove all the Punctuation from the string 

movie_data['overview'] = movie_data['overview'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))



#Remove the strings which are having lenght less than 3 

movie_data['overview'] = movie_data['overview'].apply(lambda x: re.sub(r'\b\w{1,3}\b', '', x))



#Replace NaN with an empty string

movie_data['overview'] = movie_data['overview'].fillna('')



#Remove all the mutliple space with singlular space

movie_data['overview'] = movie_data['overview'].apply(lambda x: re.sub('\s+', ' ', x).strip() )
# import these modules 

from nltk.stem import WordNetLemmatizer 



lemmatizer = WordNetLemmatizer() 



# a denotes adjective in "pos" 

movie_data['overview'] = movie_data['overview'].apply(lambda x: lemmatizer.lemmatize(x, pos ="a"))
#Check the Sample After all data pre-processing is done

movie_data['overview'].head()
#Import TfIdfVectorizer from scikit-learn

from sklearn.feature_extraction.text import TfidfVectorizer



#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'

tfidf = TfidfVectorizer(stop_words='english')



#Construct the required TF-IDF matrix by fitting and transforming the data

tfidf_matrix = tfidf.fit_transform(movie_data['overview'])



#Output the shape of tfidf_matrix

tfidf_matrix.shape
# Import linear_kernel

from sklearn.metrics.pairwise import linear_kernel



# Compute the cosine similarity matrix

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
#Construct a reverse map of indices and movie titles

indices = pd.Series(movie_data.index, index=movie_data['title_x']).drop_duplicates()

indices.head()
# Function that takes in movie title as input and outputs most similar movies

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

    return movie_data['title_x'].iloc[movie_indices]
get_recommendations('The Dark Knight Rises')
#Do the Relality Check

pd.options.display.max_colwidth = 1000

movie_data.loc[movie_data['title_x']=='The Dark Knight Rises' , 'overview']
movie_data.loc[movie_data['title_x']=='Batman Returns' , 'overview']
get_recommendations('The Avengers')
movie_data.loc[movie_data['title_x']=='The Avengers' , 'overview']
movie_data.loc[movie_data['title_x']=='Avengers: Age of Ultron' , 'overview']