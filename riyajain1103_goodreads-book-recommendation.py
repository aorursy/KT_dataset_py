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
import numpy as np 
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from tqdm import tqdm
import re
from scipy.cluster.vq import kmeans, vq
from pylab import plot, show
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
import re
import string
import random
from PIL import Image
import requests
from io import BytesIO
df=pd.read_csv("/kaggle/input/goodreadsbooks/books.csv",error_bad_lines=False)
df.head()
df.describe()
df.shape
df['bookID'].nunique()
df.index=df['bookID']
df.head()
most_occurent_books=df['title'].value_counts()[:15]
sns.barplot(x=most_occurent_books,y=most_occurent_books.index,palette='deep')
plt.title('Most occurent books')
plt.xlabel('Number of occurences')
plt.ylabel('Book')
plt.show()
Freq_authors=df['authors'].value_counts()[:10]
sns.barplot(x=Freq_authors,y=Freq_authors.index,palette='deep')
plt.title('Most famous authors')
plt.xlabel('No. of occurences')
plt.ylabel('Author')
plt.show()
author=Freq_authors.index
print(author)

lang=df['language_code'].value_counts()
sns.set_context('paper')
plt.figure(figsize=(15,10))
sns.barplot(x=lang,y=lang.index,palette='deep')
plt.title('Language Code')
plt.xlabel('No. of occurences')
plt.ylabel('language')
plt.show()
most_rated=df.sort_values('average_rating',ascending=False).head(10).set_index('title')
plt.figure(figsize=(15,10))
plt.xlabel('rating')
plt.ylabel('book')
sns.barplot(most_rated['average_rating'], most_rated.index, palette='rocket')
most_rated=df.sort_values('average_rating',ascending=False).head(50).set_index('title')
plt.figure(figsize=(15,10))
plt.xlabel('rating')
plt.ylabel('book')
sns.barplot(most_rated['average_rating'], most_rated.index, palette='rocket')
most_rated=df.sort_values('average_rating',ascending=False).head(50).set_index('authors')
plt.figure(figsize=(15,10))
plt.xlabel('rating')
plt.ylabel('author')
sns.barplot(most_rated['average_rating'], most_rated.index, palette='rocket')
df['publisher'].nunique()
publishers=df['publisher'].value_counts()[:20]
publishers.head(5)
sns.barplot(x=publishers,y=publishers.index,palette='rocket')
plt.xlabel('occurence')
plt.ylabel('publisher')
plt.figure(figsize=(15,10))

most_rated=df.sort_values('average_rating',ascending=False).head(50).set_index('publisher')
plt.figure(figsize=(15,10))
plt.xlabel('rating')
plt.ylabel('publisher')
sns.barplot(most_rated['average_rating'], most_rated.index, palette='rocket')
plt.figure(figsize=(15,10))

corr=df.corr()
sns.heatmap(corr,annot=True)
sns.distplot(df['average_rating'],kde=False)

sns.distplot(df['ratings_count'],kde=False)

df.head()
df1=df[['title','authors','average_rating']]
df1.head()
#Generalised or a demographic filter
demo=df1.sort_values('average_rating',ascending=False).set_index('title')

demo1=df1.sort_values('average_rating',ascending=False).set_index('authors')

#Recommending top 10 books to the user

rec=demo[:10]
rec
rec2=demo1[:10]
#Recommendingg top authors'book
rec2
df1.head()
def clean(x):
    x="".join(x)
    return str.lower(x)
    
df1['authors']=df1['authors'].apply(clean)
df1.head()
df1['title'][1]
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df1['authors'])
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

cosine_sim2[0][0]
df1.set_index('title')
l=list(enumerate(cosine_sim2[indices['Harry Potter Collection (Harry Potter  #1-6)']]))
print(l)
indices = pd.Series(df1.index, index=df1['title']).drop_duplicates()

def get_recommendations(title, cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:31]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df1['title'].iloc[movie_indices]
get_recommendations('Simply Beautiful Beading: 53 Quick and Easy Projects',cosine_sim2)

get_recommendations('Harry Potter Collection (Harry Potter  #1-6)',cosine_sim2)

get_recommendations('The Salmon of Doubt (Dirk Gently  #3)',cosine_sim2)

get_recommendations('Life of Pi',cosine_sim2)

get_recommendations('Self',cosine_sim2)

