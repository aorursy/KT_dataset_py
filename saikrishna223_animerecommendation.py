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
#Importing libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data = '../input/animeenglishdubbed/anime_data.csv'
anime = pd.read_csv(data)
anime.columns
anime = pd.read_csv(data,usecols=['name','rating','votes'], dtype={'name':'str','rating':'float32','votes':'int32'})
anime.head()
R = anime['rating']

v = anime['votes']

C = anime['rating'].mean()

m = anime['votes'].quantile(0.75)
anime['WeightedAverage'] = ((R*v)+(C*m))/(v+m)
anime.head()
anime = anime .sort_values('WeightedAverage',ascending = False)

anime  = anime [['name','rating','votes','WeightedAverage']]

anime .head()
WeightedAverage = anime.sort_values('WeightedAverage',ascending = False)

plt.figure(figsize=(10,5))

axis1 = sns.barplot(x = WeightedAverage['WeightedAverage'].head(10), y =WeightedAverage['name'].head(10))

plt.xlim(4,10)

plt.title('Best anime by Average Votes', weight='bold')

plt.xlabel('WeightedAverage Score', weight='bold')

plt.ylabel('Anime', weight='bold')
from wordcloud import WordCloud

from fuzzywuzzy import process #for matching string names
anime = pd.read_csv(data, usecols=['name','overview'], dtype={'name':'str','overview':'str'})

anime.head()
overview=[]

for overview in anime.overview:

    

    x=overview.split('|')

    for i in x:

         if i not in overview:

            overview.append(str(i))

overview=str(overview)    
wordcloud_overview=WordCloud(width=1500,height=800,background_color='cyan',min_font_size=2

                    ,min_word_length=3).generate(overview)

plt.figure(figsize=(20,6))

plt.axis('off')

plt.title('Overview',fontsize=25)

plt.imshow(wordcloud_overview)
#Import TfIdfVectorizer from scikit-learn

from sklearn.feature_extraction.text import TfidfVectorizer



#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'

tfidf = TfidfVectorizer(stop_words='english')



#Replace NaN with an empty string

anime['overview'] = anime['overview'].fillna('')



#Construct the required TF-IDF matrix by fitting and transforming the data

tfidf_matrix = tfidf.fit_transform(anime['overview'])



#Output the shape of tfidf_matrix

tfidf_matrix.shape
#Array mapping from feature integer indices to feature name.

tfidf.get_feature_names()[194:200]
# Import linear_kernel

from sklearn.metrics.pairwise import linear_kernel



# Compute the cosine similarity matrix

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

cosine_sim.shape
#Construct a reverse map of indices and movie titles

indices = pd.Series(anime.index, index=anime['name']).drop_duplicates()

indices[:10]
# Function that takes in anime title as input and outputs most similar anime

def get_recommendations(title, cosine_sim=cosine_sim):

    # Get the index of the anime that matches the title

    idx = process.extractOne(title, anime['name'])[2]

    print('Anime selected:',anime['name'][idx])

    print('Recommendations based on selected anime are:')

    # Get the pairwsie similarity scores of all anime with that anime

    sim_scores = list(enumerate(cosine_sim[idx]))



    # Sort the anime based on the similarity scores

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)



    # Get the scores of the 10 most similar anime

    sim_scores = sim_scores[1:11]



    # Get the anime indices

    anime_indices = [i[0] for i in sim_scores]



    # Return the top 10 most similar movies

    return anime['name'].iloc[anime_indices]
get_recommendations('naruto')
anime = pd.read_csv(data, usecols=['name','genre','stars'], dtype={'name':'str','genre':'str','stars':'str'})
anime.head()
def combine_features(row):

    try:

        return row['name']+' '+row['genre']+' '+row['stars']

    except:

        print("Error:",row)
anime['combined_features'] = anime.apply(combine_features,axis = 1)

anime['combined_features'].head()
#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'

tfidf = TfidfVectorizer(stop_words='english')



#Replace NaN with an empty string

anime['combined_features'] = anime['combined_features'].fillna('')



#Construct the required TF-IDF matrix by fitting and transforming the data

tfidf_matrix = tfidf.fit_transform(anime['combined_features'])



#Output the shape of tfidf_matrix

tfidf_matrix.shape
#Array mapping from feature integer indices to feature name.

tfidf.get_feature_names()[194:200]
# Compute the cosine similarity matrix

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

cosine_sim.shape
#Construct a reverse map of indices and movie titles

indices = pd.Series(anime.index, index=anime['name']).drop_duplicates()
indices[:10]
# Function that takes in anime title as input and outputs most similar anime

def get_recommendations(title, cosine_sim=cosine_sim):

    # Get the index of the anime that matches the title

    idx = process.extractOne(title, anime['name'])[2]

    print('Anime selected:',anime['name'][idx])

    print('Recommendations based on selected anime are:')

    # Get the pairwsie similarity scores of all anime with that anime

    sim_scores = list(enumerate(cosine_sim[idx]))



    # Sort the anime based on the similarity scores

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)



    # Get the scores of the 10 most similar anime

    sim_scores = sim_scores[1:11]



    # Get the anime indices

    anime_indices = [i[0] for i in sim_scores]



    # Return the top 10 most similar movies

    return anime['name'].iloc[anime_indices]
get_recommendations('death note')
get_recommendations('naruto')