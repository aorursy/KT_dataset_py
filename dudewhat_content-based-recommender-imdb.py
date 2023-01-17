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
df = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv')
df = df.head(30000)

df.head(15)
# Need to create a TFID based vectorizer to NLP

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
df['overview'] = df['overview'].fillna('')
X = vectorizer.fit_transform(df['overview'])
X.shape


# To do : Need to clean the data set.
# Import linear_kernel to compute the dot product
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(X,X)
cosine_sim.shape

#Construct a reverse mapping of indices and movie titles, and drop duplicate titles, if any
indices = pd.Series(df.index, index=df['title']).drop_duplicates()
df.columns
# plot based recommender function

def plot_based_recommender(title,indices=indices,cosine_sim=cosine_sim,df=df):
    movieID = indices[title]; # Get the movie id
    # next, find the cosine similarity
    moviesim = cosine_sim[movieID]
    moviescores = list(enumerate(moviesim))
    moviescores = sorted(moviescores,key=lambda x:x[1], reverse =True)
    moviescores = moviescores[1:6]
    similarmovieindex = [i[0] for i in moviescores]
    # with the index, now return the title of the movie
    
     
    return df['title'].iloc[similarmovieindex];
df.head()
plot_based_recommender('Toy Story')