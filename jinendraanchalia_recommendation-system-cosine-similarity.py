# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
  #  for filename in filenames:
   #     print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
!pip install rake_nltk
import pandas as pd
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
df = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
df = df[['title','type','listed_in','rating','director','cast','description']]
df.head()
#converting the entire description column into a list
#where every word is an element and renaming the description column into bag_of_words.
df['Key_words'] = ""

for index, row in df.iterrows():
    plot = row['description']
    r = Rake()
    r.extract_keywords_from_text(plot)
    key_words_dict_scores = r.get_word_degrees()
    #a rating is also assigned to every word
    row['Key_words'] = list(key_words_dict_scores.keys())
    
df.drop(columns = ['description'], inplace = True)
df.head()
df.isnull().sum()
df['cast'] = df['cast'].fillna('')
df['director'] = df['director'].fillna('')
df['rating'] = df['rating'].fillna('')

df['cast'] = df['cast'].map(lambda x: x.split(',')[:3])
df['director'] = df['director'].map(lambda x: x.split(',')[:3])
df['listed_in'] = df['listed_in'].map(lambda x: x.lower().split(','))
df.head()
#removing all spaces between words and names
#convering all the words into lower case
for index, row in df.iterrows():
    row['cast'] = [x.lower().replace(' ','') for x in row['cast']]
for index, row in df.iterrows():
    row['director'] = [x.lower().replace(' ','') for x in row['director']]
for index, row in df.iterrows():
    row['listed_in'] = [x.lower().replace(' ','') for x in row['listed_in']]
#all the columns, recommendation is based on, should be all lists
df['rating'] = df['rating'].map(lambda x: x.lower())
df['rating'] = df['rating'].map(lambda x: x.replace('-',''))
df['type'] = df['type'].map(lambda x: x.lower())
df['type'] = df['type'].map(lambda x: x.replace(' ',''))
df['type'] = df['type'].map(lambda x: x.lower().split(','))
df['rating'] = df['rating'].map(lambda x: x.lower().split(','))
df.set_index('title', inplace = True)
df.head()
#creating a bag_of_words
#combining all the lists into a single list
df['bag_of_words'] = ''
columns = df.columns
for index, row in df.iterrows():
    words = ''
    for col in columns:
        words = words + ' '.join(row[col])+ ' '
    row['bag_of_words'] = words
    
df.drop(columns = [col for col in df.columns if col!= 'bag_of_words'], inplace = True)
df.head()
count = CountVectorizer()
count_matrix = count.fit_transform(df['bag_of_words'])

indices = pd.Series(df.index)
indices[:5]
# generating the cosine similarity matrix
# comparing bag_of_words of every title with every other title creating nxn matrix where n is total number of rows(titles)
cosine_sim = cosine_similarity(count_matrix, count_matrix)
cosine_sim
def recommendations(title, cosine_sim = cosine_sim):
    recommended_movies = []
    idx = indices[indices == title].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    top_10_indexes = list(score_series.iloc[1:11].index)
    for i in top_10_indexes:
        recommended_movies.append(list(df.index)[i])
        
    return recommended_movies
# Time for some recommendations
recommendations('Naruto Shippuden : Blood Prison')
recommendations('PK')