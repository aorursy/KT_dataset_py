import surprise
import tensorflow as tf
from surprise import KNNWithMeans

from surprise import Dataset

from surprise.model_selection import GridSearchCV

from surprise.model_selection import cross_validate, train_test_split

import zipfile

from surprise import Reader, Dataset, SVD

from surprise import accuracy

import random

from random import randint

import re

from itertools import groupby

import pandas as pd

import copy

import operator
def read_data(extended = False):   

    if(extended):

        with open('./new_data.data') as f:

            # Data is read line by line, easier to transform into dataframe

            all_movies = f.readlines()



        # Prepare the data to be used in Surprise

        reader = Reader(line_format='user item rating timestamp', sep='\t')

        data = Dataset.load_from_file('./new_data.data', reader=reader)

        

    else:

        with open('../input/movielens-100k-dataset/ml-100k/u.data') as f:

            all_movies = f.readlines()



        # Prepare the data to be used in Surprise

        reader = Reader(line_format='user item rating timestamp', sep='\t')

        data = Dataset.load_from_file('../input/movielens-100k-dataset/ml-100k/u.data', reader=reader)

    

    return all_movies, data
all_lines, data = read_data()

# all_lines
def create_dataframe(data):

    data = [ x.replace('\t', ',').replace('\n', '') for x in data ]



    df = pd.DataFrame([sub.split(",") for sub in data])

    df.rename(columns={0:'userID', 1:'movieID', 2:'rating', 3: 'timestamp'}, 

                         inplace=True)

    df = df.drop(columns=['timestamp'])

    return df
df_final = create_dataframe(all_lines).astype('int')
def strip_content(data):

    r_unwanted = re.compile("[\n\t\r]")

    return r_unwanted.sub(",", data)
algo = SVD(n_epochs=gs.best_params["rmse"]['n_epochs'], lr_all=gs.best_params["rmse"]['lr_all'], reg_all=gs.best_params["rmse"]['reg_all'])
algo = SVD(n_epochs=10, lr_all=0.005, reg_all=0.1)
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=2, verbose=True)
def RMSE_predict_train_test(data):

    # sample random trainset and testset

    # test set is made of 25% of the ratings.

    trainset, testset = train_test_split(data, test_size=.25)



    # We'll use the famous SVD algorithm.

    algo = SVD(n_epochs=10, lr_all=0.005, reg_all=0.4)



    # Train the algorithm on the trainset, and predict ratings for the testset

    algo.fit(trainset)

    predictions = algo.test(testset)



    # Then compute RMSE

    return accuracy.rmse(predictions)
import timeit



def training_time(data):

    start = timeit.default_timer()



    trainset = data.build_full_trainset()



    algo.fit(trainset)



    stop = timeit.default_timer()



    print('Time: ', stop - start)  
training_time(data)
def predict_scores_dict(data):

    # Build Training set. Needed to fit to create model.

    print("build trainingset")

    trainset = data.build_full_trainset()

    

    print("training algo")

    algo.fit(trainset)

    

    # Get all the user and item IDs

    user_ids = trainset.all_users()

    item_ids = trainset.all_items()

    

    # Create empty list to store predictions

    ratings = {}

    ratings_list = []

    

    print("start prediction")

    start = timeit.default_timer()

    # For loop, estimate rating of each user for every movie.

    for user_id in user_ids:

        for item_id in item_ids:

            

            prediction = algo.predict(str(user_id), str(item_id)).est

            ratings['userID'] = int(user_id)

            ratings['movieID'] = int(item_id)

            ratings['rating'] = prediction

            

            ratings_list.append(ratings)

            

            ratings = {}

    stop = timeit.default_timer()

    

    print("Prediction time: ", int(stop-start), " seconds")

            

    return ratings_list
predicted_ratings = predict_scores_dict(data)
predicted_ratings_df = pd.DataFrame(predicted_ratings)
df_final.loc[df_final['userID'] == 94].where(df_final['movieID'] == 96).dropna()['rating']
def create_data(data, df_final):

    trainset = data.build_full_trainset()



    user_ids = trainset.all_users()

    item_ids = trainset.all_items()



    

    data_list = []

    # Create new movies (168200 in total)

    for movie in range(item_ids[-1]+1, item_ids[-1]*2):

        # For every movie, there will be 100 users rating the movie

        user_generated = [randint(0, user_ids[-1]) for p in range(0, 100)]

        for user in user_generated:

            try:

                rating = df_final_index.at[(user, movie), 'rating']

            except:

                rating = str(random.randint(1,5))

            new_data = str(user)+'\t'+str(movie)+'\t'+str(int(rating))+'\t'+'NaN\n'

            data_list.append(new_data)

    return data_list
data_newest = create_data(data, df_final)
# def create_data(data, df_final):

#     trainset = data.build_full_trainset()



#     user_ids = trainset.all_users()

#     item_ids = trainset.all_items()



#     data_list = []

#     # Create new movies (168200 in total)

#     for movie in range(item_ids[-1]+1, item_ids[-1]*2):

#         # For every movie, there will be 100 users rating the movie

#         user_generated = [randint(0, user_ids[-1]) for p in range(0, 100)]

#         for user in user_generated:

#             # Create a random generated score for the movies

#             new_data = str(user)+'\t'+str(movie)+'\t'+str(random.randint(1,5))+'\t'+'NaN\n'



#             data_list.append(new_data)

#     return data_list
# test = create_data(data, df_final)
# new_data = create_data(data, df_final)
new_data = []



new_data.extend(all_lines)

new_data.extend(data_newest)
len(data_newest)/len(all_lines)
def save_file(data):

    with open("new_data1.data","w") as f:

        f.writelines(data)
# save_file(new_data)
trainset.all_users()
def drop_dups(userID, df_final, predicted_scores_df):        

    df_user0_predicted = predicted_scores_df[predicted_scores_df['userID'] == userID]

    # df_user0_predicted.sort_values('rating', ascending=False)

    df_user0_predicted.reset_index(drop=True, inplace=True)

    df_final = df_final.astype('int')

    df_user0_original = df_final[df_final['userID'] == userID]

    df_user0_original.reset_index(drop=True, inplace=True)



    dfs_dictionary = {'DF1':df_user0_predicted,'DF2':df_user0_original}

    df3=pd.concat(dfs_dictionary)

    df3=df3.drop_duplicates(subset=['userID', 'movieID'],keep=False)

    

    return df3
def replace_predicted_with_original_for_all_users(df_final, predicted_scores_df, user_ids = None):

    trainset = data.build_full_trainset()

    # Check if the user id is specified or not

    if(user_ids is None):

        # Get all user ids from the trainset

        user_ids = trainset.all_users()

    else:

        # Use specified userID

        user_ids = [user_ids]

        

    df_append = pd.DataFrame()

    df_final = df_final.astype('int')

    

    for userID in user_ids:

        df_user0_predicted = predicted_scores_df[predicted_scores_df['userID'] == userID]

        df_user0_predicted.reset_index(drop=True, inplace=True)

        

        df_user0_original = df_final[df_final['userID'] == userID]

        df_user0_original.reset_index(drop=True, inplace=True)



        dfs_dictionary = {'DF1':df_user0_predicted,'DF2':df_user0_original}

        

        

        df3=pd.concat(dfs_dictionary)

        

        df3=df3.drop_duplicates(subset=['userID', 'movieID'],keep=False)

        df_append = df_append.append(df3)

        

        df_append = df_append.append(df_user0_original)

        df_append.reset_index(drop=True, inplace=True)

    return df_append
def best_movies_for_user(userID, amount_of_movies, df_final, predicted_scores_df):

    ascending_ratings = drop_dups(userID, df_final, predicted_scores_df).sort_values('rating', ascending=False)

    return ascending_ratings[0:amount_of_movies]
def create_dict_movie_genres():

    with open('./ml-100k/u.genre') as f:

        all_genres = f.readlines()

    movie_genres = {}



    for i in range(len(all_genres)-1):

        split_genres = all_genres[i].split('|')

        movie_genres[int(split_genres[1].split("\n")[0])] = split_genres[0]

        

    return movie_genres
def genre_per_movie():

    with open('./ml-100k/u.item') as f:

        movie_details = f.readlines()



    movie_details_dict = {}



    for i in range(len(movie_details)):

        genre_list = ([pos for pos, char in enumerate(movie_details[i][-39:-1]) if char == '1'])

        genre_list = np.array(genre_list)

        genre_list = genre_list//2

        genre_list = genre_list.tolist()

        

        movie_details_dict[movie_details[i].split('|')[1]] = genre_list



    return movie_details_dict
def number_to_movie():

    with open('./ml-100k/u.item') as f:

        movie_details = f.readlines()

        

    movie_details_dict = {}

    

    for i in range(len(movie_details)):

        movie_details_dict[int(movie_details[i].split('|')[0])] = movie_details[i].split('|')[1]

    

    movie_details_dict[0] = 'unknown'

    return movie_details_dict
def insert_into_dataframe(predicted_scores_dict):

    complete_prediction = []

    complete_prediction = pd.DataFrame(complete_prediction)

    for i in range(0, len(predicted_scores_dict), 10000000):

        print("appending",i, i+10000000)

        complete_prediction = complete_prediction.append(predicted_scores_dict[i:i+10000000])

        

    return complete_prediction
def prepare_data():

    all_lines, data = read_data()

    

    print("create df_final")

    df_final = create_dataframe(all_lines).astype('int')

    

    print("predict scores")

    predicted_scores_dict =  predict_scores_dict(data)

    

    df_predicted_scores = insert_into_dataframe(predicted_scores_dict)

    

    return df_final, df_predicted_scores
def convert_to_titles(best_movies, number_to_movie):

    i=0

    for movieID in best_movies['movieID']:

        best_movies['movieID'][i] = number_to_movie[int(movieID)]

        i+=1

        

    return best_movies
def best_genre_for_user(userID, amount_of_movies, genre_per_movie, number_to_movie):

    best_movies = best_movies_for_user(userID,amount_of_movies,df_final, df_predicted_scores)    



    list_of_genres = []

    for movieID in best_movies['movieID']:

        list_of_genres.extend(genre_per_movie[number_to_movie[int(movieID)]])



    sorted_genres = {value: len(list(freq)) for value, freq in groupby(sorted(list_of_genres))}

    

    best_genre = max(sorted_genres, key=sorted_genres.get)



    return int_to_genre[best_genre]
def create_genre_cols(best_movies, movie_genres):

    for i in range(len(movie_genres)):

        best_movies[movie_genres[i]] = int(0)

    return best_movies
def to_onehot(data):

    for int_genre in movie_to_genre[number_to_mov[data['movieID']]]:

        data[int_to_genre[int_genre]] = int(1)

    return data
def display_specific_genre(data, genre, amount):

    best_movies_genre = best_movies_for_genre(data,genre, amount)

    best_movies_genre = convert_to_titles(best_movies_genre, number_to_mov) 

    return best_movies_genre[['userID', 'movieID', 'rating', genre]]
def best_movies_for_genre(data, genre, amount):

    return data[data[genre]==1].sort_values('rating', ascending = False)[0:amount]
def convert_to_titles(best_movies, number_to_movie):

    i=0

    for movieID in best_movies['movieID']:

        best_movies['movieID'].iloc[i] = number_to_mov[int(movieID)]

        i+=1

        

    return best_movies
def get_avg_scores_per_movie_for_genre(data, genre):

    data_genre = data[data[genre] == 1]

    create_dict = {}

    for movieID in data_genre['movieID']:

        if(movieID in create_dict):

            continue

        else:

            sum_of_rating = sum(data_genre[data_genre['movieID']==movieID]['rating'])

            amount_of_ratings = len(data_genre[data_genre['movieID']==movieID]['rating'])

            avg_rating = sum_of_rating/amount_of_ratings

            create_dict[(movieID)] = avg_rating

    return create_dict
def get_best_movies_for_genre(data, genre, amount):

    avg_scores = get_avg_scores_per_movie_for_genre(data, genre)

    df_avg_scores  = pd.DataFrame.from_records([avg_scores]).transpose()

    df_avg_scores.columns = ['rating']

    df_avg_scores['movieID'] = df_avg_scores.index

    df_avg_scores = df_avg_scores.sort_values('rating',ascending=False).iloc[0:amount]

    df_avg_scores = convert_to_titles(df_avg_scores, amount)

    return df_avg_scores
def get_best_movies_for_genre2(data, genre, amount):

    avg_scores = get_avg_scores_per_movie_for_genre(data, 'Action')

    avg_scores = sorted(avg_scores.items(), key=lambda kv: kv[1])

    avg_scores.reverse()

    avg_scores[0:amount]

    avg_scores = collections.OrderedDict(avg_scores)

    df_avg_scores  = pd.DataFrame.from_records([avg_scores]).transpose()

    df_avg_scores.columns = ['rating']

    df_avg_scores['movieID'] = df_avg_scores.index

    df_avg_scores = df_avg_scores.sort_values('rating',ascending=False).iloc[0:amount]

    df_avg_scores = convert_to_titles(df_avg_scores, amount)

    return df_avg_scores
def best_genres(data):

    dict_rating = {}

    for idx in range(len(int_to_genre)):

        genre = int_to_genre[idx]

        

        sum_of_rating = sum(data[data[genre] == 1]['rating'])

        nr_of_vals = len(data[data[genre] == 1]['rating'])

        

        avg_rating = sum_of_rating / nr_of_vals



        dict_rating[genre] = avg_rating

        

    sorted_ratings = sorted(dict_rating.items(), key=lambda kv: kv[1])

    sorted_ratings.reverse()

    sorted_ratings = collections.OrderedDict(sorted_ratings)

    df_sorted_ratings  = pd.DataFrame.from_records([sorted_ratings]).transpose()

    df_sorted_ratings.columns = ['rating']

    return df_sorted_ratings
def best_genres2(data):

    genre_scores = pd.DataFrame()

    new_df = pd.DataFrame()

    for idx in int_to_genre:

        genre_scores[int_to_genre[idx]] = data[int_to_genre[idx]] * data['rating']

        genre_scores[genre_scores[int_to_genre[idx]]!=0][int_to_genre[idx]].mean()



        new_df[int_to_genre[idx]] = [genre_scores[genre_scores[int_to_genre[idx]]!=0][int_to_genre[idx]].mean()]

    new_df = new_df.transpose()

    new_df.columns=['rating']

    new_df = new_df.sort_values('rating', ascending=False)

    

    return new_df
# Convert int value to movie genres

int_to_genre = create_dict_movie_genres()

# Convert movie title to list of genres

movie_to_genre = genre_per_movie()

# Convert int value (original movie representation) to movie name

number_to_mov = number_to_movie()
# Create DataFrames with predicted scores and 

df_final, df_predicted_scores = prepare_data()
# df_predicted_scores
best_movies = best_movies_for_user(130,10,df_final, df_predicted_scores)

best_movies = best_movies.astype('str')

best_movies = convert_to_titles(best_movies, number_to_mov)
best_movies
best_genre_for_uer(148, 10, movie_to_genre, number_to_mov)
best_movies_extended = create_genre_cols(df_predicted_scores, int_to_genre)
df_final_extended = create_genre_cols(df_final, int_to_genre)
# df_final_extended
data = df_final_extended.apply(to_onehot, axis = 1)
data
# def genre_per_movie():

#     with open('./ml-100k/u.item') as f:

#         movie_details = f.readlines()



#     movie_details_dict = {}



#     for i in range(len(movie_details)):

#         genre_list = ([pos for pos, char in enumerate(movie_details[i][-39:-1]) if char == '1'])

#         genre_list = np.array(genre_list)

#         genre_list = genre_list//2

#         genre_list = genre_list.tolist()

        

#         movie_details_dict[movie_details[i].split('|')[1]] = genre_list



#     return movie_details_dict
get_best_movies_for_genre(data,'Adventure',15)
get_best_movies = get_best_movies_for_genre(data, 'Action', 1000)
df_best_genres = best_genres(data)
df_best_genres2 = best_genres2(data)
df_best_genres.equals(df_best_genres2)
df_best_genres