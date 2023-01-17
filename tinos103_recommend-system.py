import numpy as np

import pandas as pd

!pip install rake-nltk

from rake_nltk import Rake

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import CountVectorizer



df = pd.read_csv('../input/netflix-shows/netflix_titles.csv')

'''

df["date_added"] = pd.to_datetime(df['date_added'])

df['year_added'] = df['date_added'].dt.year

df['month_added'] = df['date_added'].dt.month



df['season_count'] = df.apply(lambda x : x['duration'].split(" ")[0] if "Season" in x['duration'] else "", axis = 1)

df['duration'] = df.apply(lambda x : x['duration'].split(" ")[0] if "Season" not in x['duration'] else "", axis = 1)





#c_data = np.loadtxt('../input/consinesimilaritymatrix/cosine.csv', delimiter=',')'''



new_df = df[['title','director','cast','listed_in','description']]



# REMOVE NaN VALUES AND EMPTY STRINGS:

new_df.dropna(inplace=True)



blanks = []  # start with an empty list



col=['title','director','cast','listed_in','description']

for i,col in new_df.iterrows():  # iterate over the DataFrame

    if type(col)==str:            # avoid NaN values

        if col.isspace():         # test 'review' for whitespace

            blanks.append(i)     # add matching index numbers to the list



new_df.drop(blanks, inplace=True)



# initializing the new column

new_df['Key_words'] = ""



for index, row in new_df.iterrows():

    description = row['description']

    

    # instantiating Rake, by default it uses english stopwords from NLTK

    # and discards all puntuation characters as well

    r = Rake()



    # extracting the words by passing the text

    r.extract_keywords_from_text(description)



    # getting the dictionary whith key words as keys and their scores as values

    key_words_dict_scores = r.get_word_degrees()

    

    # assigning the key words to the new column for the corresponding movie

    row['Key_words'] = list(key_words_dict_scores.keys())



# dropping the Plot column

new_df.drop(columns = ['description'], inplace = True)



# discarding the commas between the actors' full names and getting only the first three names

new_df['cast'] = new_df['cast'].map(lambda x: x.split(',')[:3])



# putting the genres in a list of words

new_df['listed_in'] = new_df['listed_in'].map(lambda x: x.lower().split(','))



new_df['director'] = new_df['director'].map(lambda x: x.split(' '))



# merging together first and last name for each actor and director, so it's considered as one word 

# and there is no mix up between people sharing a first name

for index, row in new_df.iterrows():

    row['cast'] = [x.lower().replace(' ','') for x in row['cast']]

    row['director'] = ''.join(row['director']).lower()

    

new_df.set_index('title', inplace = True)



new_df['bag_of_words'] = ''

columns = new_df.columns

for index, row in new_df.iterrows():

    words = ''

    for col in columns:

        if col != 'director':

            words = words + ' '.join(row[col])+ ' '

        else:

            words = words + row[col]+ ' '

    row['bag_of_words'] = words

    

new_df.drop(columns = [col for col in new_df.columns if col!= 'bag_of_words'], inplace = True)



# instantiating and generating the count matrix

count = CountVectorizer()

count_matrix = count.fit_transform(new_df['bag_of_words'])



# creating a Series for the movie titles so they are associated to an ordered numerical

# list I will use later to match the indexes

indices = pd.Series(new_df.index)



cosine_sim = cosine_similarity(count_matrix, count_matrix)





# function that takes in movie title as input and returns the top 10 recommended movies

def recommendations(Title, cosine_sim = cosine_sim):

    

    recommended_movies = []

    

    # gettin the index of the movie that matches the title

    idx = indices[indices == Title].index[0]



    # creating a Series with the similarity scores in descending order

    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)



    # getting the indexes of the 10 most similar movies

    top_10_indexes = list(score_series.iloc[1:11].index)

    

    # populating the list with the titles of the best 10 matching movies

    for i in top_10_indexes:

        recommended_movies.append(list(new_df.index)[i])

        

    return recommended_movies
recommendations("Logan's Run")