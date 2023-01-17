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
#import required packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Read data from movies_metadata.csv file and store in data variable

data=pd.read_csv('/kaggle/input/the-movies-dataset/movies_metadata.csv')
#View data

data.info()
data.describe()
# Analyse the first 5 sample in given data

data.head()
#Find count of tr Movies

count=0 # I will find number of  tr movies

for index,value in data[['original_language']][0:].iterrows():

   if value[0]=='tr':

    count=count+1

    print(index," -->",value[0])

   

print("----------------------------------------------------------------")

print("Numbers of TR movies",count)
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
#Read data from credits.csv file and store in data variable

data1=pd.read_csv('/kaggle/input/the-movies-dataset/credits.csv')
data1.head()
data.head()
data['id']
data1['id']
#Removing some redundant entries from the movies_metadata.csv

data = data[data.id!='1997-08-20']

data = data[data.id!='2012-09-29']

data = data[data.id!='2014-01-01']
data1.id
#Converting "id" column into integer type so we can finally combine both the dataset files as a single dataset

data['id'] = data['id'].astype(int)
data.id
#Now we merge both the dataset

data=data.merge(data1, on='id')
data.columns
#So here, I aim to create a generalized solution by taking into account all the important parameters for a 

#particular movie. This will give the user a much more personalized experience and user can get a recommnendation

#according to his requirements.
#Here I use the overview data column for building this system

data['overview'].head()
#calculate the similarity scores for the overview of each movie entry in the dataset

#use a text encoding technique known as word_to_vec in order to convert each overview into numeric embeddeings.

#This technique is used to convert textual data into numeric vectors

#Then we compute Term Frequency-Inverse Document Frequency (TF-IDF) vectors for each overview.

#This will give us a relative frequency of word in the document.

#To achieve this, I use scikit-learn which gives a built-in TfIdfVectorizer class that produces the TF-IDF matrix
#Import TfIdfVectorizer from scikit-learn

from sklearn.feature_extraction.text import TfidfVectorizer



#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'

tfidf = TfidfVectorizer(stop_words='english')



#Replace NaN with an empty string

data['overview'] = data['overview'].fillna('')
#The shape of the tfid_matrix gives us the number of distinct words used to define our entries of 45538 movies. 

#We have 75827 distinct english words used to describe the overview of the movies
#To calculate similarity score,ound several techniques like euclidean, the Pearson and the cosine similarity scores. 

#I will be using cosine similarity scores to find the similarity between two movies

#One advantage of cosine similarity score is that it is independent of magnitude and is relatively easy and fast to calculate.
from sklearn.metrics.pairwise import linear_kernel
# I will be taking a random part of data in order to generate the cosin values

import random

ran = random.randint(25000, 30000)

data2 = data.head(ran)

tfidf_matrix = tfidf.fit_transform(data2['overview'])



#Output the shape of tfidf_matrix

tfidf_matrix.shape
cosine_sim =  linear_kernel(tfidf_matrix, tfidf_matrix, True)
indices = pd.Series(data2.index, index=data2['title']).drop_duplicates()
#Now I create a get_recommendation function which accpets a movie title from the user and recommends similar movies 

#to the user based on the title by taking the cosine simalarity scores of the most similar movies.

#This function will recommend the 10 most similar movies
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

    return data2['title'].iloc[movie_indices]
get_recommendations('Assassins')
get_recommendations('Carrington')
#Taking important features like movie genres, caste of the movie, keywords etc will help to build a much more accurate recommender system
data3=pd.read_csv('/kaggle/input/the-movies-dataset/keywords.csv')
data3.head()
data=data.merge(data3, on='id')
#Now that we have our merged dataset, so we can extract important features like genres, keywords, cast etc

from ast import literal_eval



features = ['cast', 'crew', 'keywords', 'genres']

for feature in features:

    data[feature] = data[feature].apply(literal_eval)
data.head()
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

data['director'] = data['crew'].apply(get_director)



features = ['cast', 'keywords', 'genres']

for feature in features:

    data[feature] = data[feature].apply(get_list)
# Print the new features of the first 3 films

data[['title', 'cast', 'director', 'keywords', 'genres']].head(3)
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

    data[feature] = data[feature].apply(clean_data)
#metadata string fucntion which will help to concat all the important features

def create_soup(x):

    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

data['soup'] = data.apply(create_soup, axis=1)

data['soup']
#One important thing which I came to know is that here, we need to CountVectorizer instead of tf-idf because tf-idf 

#only gives unique values. So if one actor has acted in many films and another actor has acted in only few, 

#then tf-idf gives same weightage which is technically not right.

# Import CountVectorizer and create the count matrix

from sklearn.feature_extraction.text import CountVectorizer



count = CountVectorizer(stop_words='english')



# again we take a part of dataset as our kernel cannot handle all the dataset together.

 

df = data['soup'].head(20000)



count_matrix = count.fit_transform(df)
from sklearn.metrics.pairwise import cosine_similarity



cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
# we just reset the index of main data

data = data.reset_index()

indices = pd.Series(data.index, index=data['title'])
#finally calling the get_recommendation fucntion

get_recommendations('Assassins', cosine_sim2)
#The system is now able to recommend movies which are really similar to the one mentioned i.e Assasins. It has taken into account all the important features like the director of the movie, cast, and the various keywords associated with the movie
get_recommendations('Jumanji', cosine_sim2)