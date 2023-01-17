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
import pandas as pd
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
df = pd.read_csv("../input/netflix-shows/netflix_titles.csv")

# Iam dropping all nan since that is out of scope and also i have ample dataset after the drop - 
# you can use any means to fill na - 
df = df.dropna()
df = df[['title','type','director','cast','description']]
df.head()
df['cast'] = df['cast'].map(lambda x: str(x).split(',')[:3])
df['type'] = df['type'].map(lambda x: x.lower().split(','))
df['director'] = df['director'].map(lambda x: str(x).split(' '))
for index, row in df.iterrows():
    row['cast'] = [x.lower().replace(' ','') for x in row['cast']]
    row['director'] = ''.join(row['director']).lower()
df['Key_words'] = ""

for index, row in df.iterrows():
    plot = row['description']
    
    # It uses english stopwords from NLTK ,discards punctuation
    r = Rake()

    # extracting root words
    r.extract_keywords_from_text(plot)

    # keywords as keys and scores as values in dict
    key_words_dict_scores = r.get_word_degrees()
    
    # assigning the key words to the new column for the corresponding movie
    row['Key_words'] = list(key_words_dict_scores.keys())

# dropping the description column
df.drop(columns = ['description'], inplace = True)
# set movie title as index
df.set_index('title', inplace = True)
df.head()
# Bag of words
df['final_keywords'] = ''
columns = df.columns
for index, row in df.iterrows():
    words = ''
    for col in columns:
        if col != 'director':
            words = words + ' '.join(row[col])+ ' '
        else:
            words = words + row[col]+ ' '
    row['final_keywords'] = words
    
df.drop(columns = [col for col in df.columns if col!= 'final_keywords'], inplace = True)

# Initialize the frequency counts
count = CountVectorizer()
count_matrix = count.fit_transform(df['final_keywords'])
# Creating series
indices = pd.Series(df.index)
indices[:5]
# generating the cosine similarity to find the vector coordinance
cosine_ = cosine_similarity(count_matrix, count_matrix)
cosine_
# function that takes in movie title as input and returns the top 10 recommended movies 
def recommended_netflix(title, cosine_sim = cosine_):
    
    recommended_movies = []
    
    # gettin the index of the movie that matches the title
    idx = indices[indices == title].index[0]

    # creating a Series with the similarity scores in descending order
    scores = pd.Series(cosine_[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar movies
    top_movies = list(scores.iloc[1:11].index)
    
    # populating the list with the titles of the best 10 matching movies
    for i in top_movies:
        recommended_movies.append(list(df.index)[i])
        
    return recommended_movies
# find the similar shows or movies with respect to similarity 
recommended_netflix('Norm of the North: King Sized Adventure')