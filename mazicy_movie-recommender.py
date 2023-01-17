# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import re

%matplotlib inline



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_movies = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv')

df_movies.head(2)
def clean_dict_str(s, items_to_include):

    '''

    Required Libraries:

        re

    

    - Formats dictionary string into actual dict format.

    

    s: string to be formatted

        it must have dictionary-format strings within

    items_to_include: elements of the dictionary to include

        if len==1 then it returns a list of values

        if len>1 then it returns a list of dictionaries

    '''

    

    dict_list = []

    

# skips the empty dictionaries

    if len(s)>2:

    

    # extract dictionary patterns from the string

        re_pattern = '\{.*?\}'

        lst_temp = re.findall(re_pattern, s)

        

    # list of dictionary format strings

        for dict_str in lst_temp:

            dict_str = dict_str.split(', "')

            dict_temp = {}



        # looping each element of the dictionaries

            for element in dict_str:

                element = element.strip('{').strip('}').replace('"','')

                

            # only extract the chosen items

                if element.split(':')[0].strip() in items_to_include: 

                    # make sure they are unique

                    if element.split(':')[1].strip() in dict_temp.values():

                        dict_temp[element.split(':')[0].strip()] = element.split(':')[1].strip()

                    if len(items_to_include)==1:

                         # make sure they are unique

                        if element.split(':')[1].strip() not in dict_list:

                            dict_list.append(element.split(':')[1].strip())

            if len(items_to_include)>1:

                dict_list.append(dict_temp)

                    

    return dict_list
def str_search_dict_list(dict_list, s):

    if len(dict_list)==0:

        return False

    value = False

    if type(dict_list[0])==dict:

        for dictionary in dict_list:

            if value == False: 

                value = s in list(dictionary.values())

    else: 

        if value == False:

            value = s in dict_list

    return value
df_movies = df_movies.copy()

df_movies['keywords_cleaned'] = df_movies.keywords.apply(lambda x: clean_dict_str(x,['name']))

df_movies['genres_cleaned'] = df_movies.genres.apply(lambda x: clean_dict_str(x,['name']))

df_movies['lang_cleaned'] = df_movies.spoken_languages.apply(lambda x: clean_dict_str(x,['iso_639_1']))

df_movies['overview_len'] = df_movies.overview.fillna(value='').apply(len)
df_movies.head()
df_crd = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv')

df_crd.head()
df_crd.info()
df_crd['cast_cleaned'] = df_crd.cast.apply(lambda x: clean_dict_str(x,['name']))

df_crd['crew_cleaned'] = df_crd.crew.apply(lambda x: clean_dict_str(x,['name']))
df_crd.head()
df_movies.reset_index(inplace=True)

df_movies.set_index('id', inplace=True)

df_crd.reset_index(inplace=True)

df_crd.set_index('movie_id', inplace=True)



BoW = pd.concat([df_movies,df_crd], axis=1, join='inner')



drop_col = ['index','cast','crew','production_companies', 'production_countries',

            'homepage', 'keywords','genres','spoken_languages']

BoW.drop(drop_col, axis=1, inplace=True)



BoW.reset_index(inplace=True)

BoW.set_index('original_title', inplace=True)
BoW.head()
bag_of_words_columns=['genres_cleaned','keywords_cleaned','cast_cleaned']





BoW['bag_of_words'] = ''





for index, row in BoW.iterrows():

    words = ''

    for col in bag_of_words_columns:

        if col == 'cast_cleaned':

            words = words + ' '.join(row[col][:9])+ ' '

        else:

            words = words + ' '.join(row[col])+ ' '

    BoW.loc[index,'bag_of_words'] = words

    

# BoW.drop(columns = [col for col in BoW.columns if col!= 'bag_of_words'], inplace=True)
BoW.head()
count = CountVectorizer()

count_matrix = count.fit_transform(BoW['bag_of_words'])

indices = pd.Series(BoW.index)

cosine_sim = cosine_similarity(count_matrix, count_matrix)

cosine_sim
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

        recommended_movies.append(list(BoW.index)[i])

        

    return recommended_movies
BoW.overview_len
def find_movie(string, dataframe, top_n = 10):

    '''

    Searches for movie titles similar to the given term string

    DataFrame's indices must be the movie titles

    '''

    top_n += 1 

    

    # add the string to the list and prepare

    

    title_pool = list(dataframe.index)

    title_pool.insert(0, string)



    temp_indices = pd.Series(title_pool)



    # create similarity indices

    vector = CountVectorizer().fit_transform(title_pool)

    cosine_matrix = cosine_similarity(vector, vector)

    similar_movies_indices = pd.Series(cosine_matrix[0]).sort_values(ascending=False)[1:top_n]

    

    # only values with > 0 similarity

    similar_movies_indices = similar_movies_indices[similar_movies_indices>0]

    

    

    

    similar_movies_list = list(temp_indices.loc[similar_movies_indices.index].values)

    

    for movie in dataframe[dataframe.index.map(lambda x: string in x.lower())].index:

        similar_movies_list.append(movie)

    

    similar_movies_list = sorted(list(set(similar_movies_list)))

    return similar_movies_list



def find_movie_and_recommend(string, dataframe=BoW):

    find_movie_list = find_movie(string, dataframe)

    i = 0

    for movie in find_movie_list:

        print('{}: {}'.format(i+1,movie))

        i=i+1

    print('')

    

    if len(find_movie_list)==0:

        print ('There is no search result with that term.')

    else:

        while True:

            try: 

                correct_movie_pos = int(input('Please choose one number from the above list: '))

                find_movie_list[correct_movie_pos-1]

            except ValueError: 

                print('* Please choose one of the above integer choices')

                continue

            else:

                break

        print("\nYou chose: '{}'\n".format(find_movie_list[correct_movie_pos-1]))



        recommend_list = recommendations(find_movie_list[correct_movie_pos-1])



        print("Here is the recommended list of similar movies to '{}'.\n".format(find_movie_list[correct_movie_pos-1]))

        for movie in recommend_list:

            print ('-',movie)
find_movie_and_recommend('superman')