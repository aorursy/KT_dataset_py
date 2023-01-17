!pip install rake-nltk
#import the necessary libraries

import numpy as np

import pandas as pd

from rake_nltk import Rake

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import CountVectorizer
#Fetch the csv file from the web and make the dataframe and display the first 5 rows of dataframe

pd.set_option('display.max_columns', 100)

df = pd.read_csv('https://query.data.world/s/uikepcpffyo2nhig52xxeevdialfl7')

df.head()
#display the shape of dataframe

print("The dataframe has {} rows and {} columns".format(df.shape[0],df.shape[1]))
#display the all column names

df.columns
#so we will use only title,director,genre,actor,plot column only to build our reommendation system.

df = df[['Title','Genre','Director','Actors','Plot']]
df.head()
#display the shape of dataframe

df.shape
#before building any recommendation system we have to preprocess the text. Let's start

df['Genre'] = df['Genre'].map(lambda x:x.lower().split(','))
df['Director'] = df['Director'].map(lambda x: x.lower().split(' '))
df['Actors'] = df['Actors'].map(lambda x:x.lower().split(','))
for index,row in df.iterrows():

    row['Actors'] = [x.replace(' ','') for x in row['Actors']]

    row['Director'] = ''.join(row['Director'])
df['Keywords'] = ''

for index,row in df.iterrows():

    plot = row['Plot']

    r = Rake()

    r.extract_keywords_from_text(plot)

    key_words_dict_scores = r.get_word_degrees()

    row['Keywords'] = list(key_words_dict_scores.keys())
df.drop(columns=['Plot'],inplace=True)
df.set_index('Title',inplace=True)
df['bag_of_words'] = ''

for index,row in df.iterrows():

    words = ''

    for col in df.columns:

        if col != 'Director':

            words = words + ' '.join(row[col])+ ' '

        else:

            words = words +row[col]+' '

    row['bag_of_words'] = words

    

df.drop(columns=[x for x in df.columns if x!='bag_of_words'],inplace=True)
df['bag_of_words'][0]
count = CountVectorizer()

count_matrix = count.fit_transform(df['bag_of_words'])
indices = pd.Series(df.index)

indices[:3]
count_matrix.shape
count_matrix.toarray()
print(count_matrix)
count.vocabulary_
cosine_sim= cosine_similarity(count_matrix,count_matrix)

cosine_sim
cosine_sim.shape
def recommendations(title,cosine_sim = cosine_sim):

    recommendation_movies = []

    # gettin the index of the movie that matches the title

    idx = indices[indices == title].index[0]

    # creating a Series with the similarity scores in descending order

    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar movies

    top_10_indexes = list(score_series.iloc[1:11].index)

    print(top_10_indexes)

    # populating the list with the titles of the best 10 matching movies

    for i in top_10_indexes:

        recommendation_movies.append(list(df.index)[i])

    return recommendation_movies
recommendations('The Godfather')