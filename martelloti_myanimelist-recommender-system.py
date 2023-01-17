from collections import defaultdict

import pandas as pd

import numpy as np

import scipy

from scipy.sparse.linalg import svds

import matplotlib.pyplot as plt

import surprise as sp

import time
#Importing the CSVs to Dataframe format

UsersDF = pd.read_csv('../input/users_cleaned.csv')

AnimesDF = pd.read_csv('../input/anime_cleaned.csv')

ScoresDF = pd.read_csv('../input/animelists_cleaned.csv')



AnimesDF.head()
UsersDF.head()
ScoresDF.head()
#Since ScoresDF is a huge DF (2GB of data) I`ll only take the columns that are important for the recommendation system

ScoresDF = ScoresDF[['username', 'anime_id', 'my_score', 'my_status']]
ScoresDF['my_score'].describe().apply(lambda x: format(x, '.2f')).reset_index()
#Analysing all the possible values for the score, this will be used as a parameter later on

lower_rating = ScoresDF['my_score'].min()

upper_rating = ScoresDF['my_score'].max()

print('Range of ratings vary between: {0} to {1}'.format(lower_rating, upper_rating))
#Only filtering animes in which people actually watched, are watching or are on hold as they are the most revelant for the rec sys

#RelevantScoresDF = ScoresDF[(ScoresDF['my_status'] == 1) | (ScoresDF['my_status'] == 2) | (ScoresDF['my_status'] == 3)]
#Counting how many relevant scores each user have done, resetting the index (so the series could become a DF again) and changing the column names

UsersAndScores = ScoresDF['username'].value_counts().reset_index().rename(columns={"username": "animes_rated", "index": "username"})
UsersSampled = UsersDF.sample(frac = .01, random_state = 2)

UsersSampled.head()
UsersAndScoresSampled = pd.merge(UsersAndScores, UsersSampled, left_on = 'username', right_on = 'username', how = 'inner')
#Grouping users whom had the same amount of animes rated

UserRatedsAggregated = UsersAndScoresSampled['animes_rated'].value_counts().reset_index().rename(columns={"animes_rated": "group_size", "index": "animes_rated"}).sort_values(by=['animes_rated'])
#Counting how many relevant scores each anime has, resetting the index (so the series could become a DF again) and changing the column names

RatedsPerAnime = ScoresDF['anime_id'].value_counts().reset_index().rename(columns={"anime_id": "number_of_users", "index": "anime_id"})

RatedsPerAnime.head()
#Grouping users whom had the same amount of animes rated

AnimeRatedsAggregated = RatedsPerAnime['number_of_users'].value_counts().reset_index().rename(columns={"number_of_users": "group_size", "index": "number_of_users"}).sort_values(by=['number_of_users'])

AnimeRatedsAggregated.head(n = 30)
#Creating the plots so we can gather information about the distribution of ratings in the sample

plt.suptitle("Distribution of users", fontsize=13, fontweight=0, color='black', style='italic', y=1.02)

plt.plot('animes_rated', 'group_size', data = UserRatedsAggregated, color = 'blue')

plt.xlabel('Number of animes rated')

plt.ylabel('Number of people in that group')

plt.xlim(left = 0, right = 2000)

plt.show()
#Creating the plots so we can gather information about the distribution of ratings in the sample

plt.suptitle("Distribution of animes", fontsize=13, fontweight=0, color='black', style='italic', y=1.02)

plt.plot('number_of_users', 'group_size', data = AnimeRatedsAggregated, color = 'olive')

plt.xlabel('Number of users rated')

plt.ylabel('Number of animes in that group')

plt.xlim(left = 0, right = 2000)

plt.show()
#Creating a dataframe of users  and animes with more than 10 interactions

UserRatedsCutten = UsersAndScoresSampled[UsersAndScoresSampled['animes_rated'] >= 10]

AnimeRatedsCutten = RatedsPerAnime[RatedsPerAnime['number_of_users'] >= 10]

#Joining (merging) our new dataframes with the interactions one (this will already deal with the sample problem,

#as it is an inner join). The "HotStart" name comes from a pun about solving the "Cold Start" issue

ScoresDFHotStart = pd.merge(ScoresDF, UserRatedsCutten, left_on = 'username', right_on = 'username', how = 'inner')

ScoresDFHotStart = pd.merge(ScoresDFHotStart, AnimeRatedsCutten, left_on = 'anime_id', right_on = 'anime_id', how = 'inner')
#Grouping the different scores and resetting the index (so the series could become a DF again) 

AnimeRates = ScoresDF['my_score'].value_counts().reset_index().sort_values('index')

plt.plot('index', 'my_score', data = AnimeRates, color = 'red')

plt.xticks(np.arange(11))

plt.ticklabel_format(axis = 'y', style = 'plain')

plt.xlabel('Score')

plt.ylabel('Frequency of that score')

plt.show()
#Just for the record, lets see the difference in numbers between our initial DF and the sampled and cleaned one



print('The initial dataframe has {0} registers and the sampled one has {1} rows.'.format(ScoresDF['username'].count(), ScoresDFHotStart['username'].count()))
def precision_recall_at_k(predictions, k=10, threshold= 7):

    '''Return precision and recall at k metrics for each user.'''



    # First map the predictions to each user.

    # Predictions: Traz uma lista de 5 campos dentro de uma tupla com as seguintes infos: User_ID, Item_ID, True_ID, Est_ID, Details

    user_est_true = defaultdict(list)

    for uid, _, true_r, est, _ in predictions:

        user_est_true[uid].append((est, true_r))

    # Creates a dict with the key being a user and the value bringing the estimated rating and the true rating.



    precisions = dict()

    recalls = dict()

    for uid, user_ratings in user_est_true.items():



        # Sort user ratings by estimated value

        user_ratings.sort(key=lambda x: x[0], reverse=True)



        # Number of relevant items

        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)



        # Number of recommended items in top k

        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])



        # Number of relevant and recommended items in top k

        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))

                              for (est, true_r) in user_ratings[:k])



        # Precision@K: Proportion of recommended items that are relevant

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1



        # Recall@K: Proportion of relevant items that are recommended

        recalls[uid] = n_rel_and_rec_k /  n_rel if n_rel != 0 else 1



    return precisions, recalls
random_state = 42

reader = sp.Reader(rating_scale=(0, 10))

data = sp.Dataset.load_from_df(ScoresDFHotStart[['username', 'anime_id', 'my_score']], reader)

trainset, testset = sp.model_selection.train_test_split(data, test_size=.25, random_state = random_state)

analysis = defaultdict(list)



test_dict = {'SVD' : sp.SVD(random_state=random_state), 'SlopeOne' : sp.SlopeOne(), 'NMF' : sp.NMF(random_state=random_state), 'NormalPredictor' : sp.NormalPredictor(), 'KNNBaseline' : sp.KNNBaseline(random_state=random_state), 'KNNBasic' : sp.KNNBasic(random_state=random_state), 'KNNWithMeans' : sp.KNNWithMeans(random_state=random_state), 'KNNWithZScore' : sp.KNNWithZScore(random_state=random_state), 'BaselineOnly' : sp.BaselineOnly(), 'CoClustering': sp.CoClustering(random_state=random_state)}



for key, value in test_dict.items():

    start = time.time()    

    value.fit(trainset)

    predictions = value.test(testset)



    rmse = sp.accuracy.rmse(predictions)

    precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=7)

    precision_avg = sum(prec for prec in precisions.values()) / len(precisions)



    analysis[value] = (key, rmse, precision_avg, time.time() - start)



print(analysis)
analysis_df = pd.DataFrame.from_dict(analysis, orient = 'index', columns = ['Algorithm', 'RMSE', 'Precision@10', 'Time to run (in seconds)']).reset_index()



#analysis_df['Algorithm'] = ['SVD', 'SlopeOne', 'NMF', 'NormalPredictor', 'KNNBaseline', 'KNNBasic', 'KNNWithMeans', 'KNNWithZScore', 'BaselineOnly', 'CoClustering']

analysis_df = analysis_df[['Algorithm', 'RMSE', 'Precision@10', 'Time to run (in seconds)']]

analysis_df = analysis_df.sort_values(by=['Precision@10'], ascending = False)

analysis_df['RMSE^-1'] = analysis_df['RMSE'] ** -1

analysis_df.head(n = 15)
ax = analysis_df.set_index('RMSE^-1')['Precision@10'].plot(style='o', c = 'DarkBlue', figsize = (15, 20))

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

def label_point(x, y, val, ax):

    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)

    for i, point in a.iterrows():

        ax.text(point['x'], point['y'], str(point['val']))



label_point(analysis_df['RMSE^-1'], analysis_df['Precision@10'], analysis_df['Algorithm'], ax)
als_param_grid = {'bsl_options': {'method': ['als'],

                              'reg_i': [5, 10, 15],

                              'reg_u': [10, 15, 20],

                              'n_epochs': [5, 10, 15, 20]

                              }

              }



sgd_param_grid = {'bsl_options': {'method': ['sgd'],

                              'reg': [0.01, 0.02, 0.03],

                              'n_epochs': [5, 10, 15, 20],

                              'learning_rate' : [0.001, 0.005, 0.01]

                              }

              }



als_gs = sp.model_selection.GridSearchCV(sp.BaselineOnly, als_param_grid, measures=['rmse'], cv = 3, joblib_verbose = 0)



sgd_gs = sp.model_selection.GridSearchCV(sp.BaselineOnly, sgd_param_grid, measures=['rmse'], cv = 3, joblib_verbose = 0)
als_gs.fit(data)



# best RMSE score

print(als_gs.best_score['rmse'])



# combination of parameters that gave the best RMSE score

print(als_gs.best_params['rmse'])
sgd_gs.fit(data)



# best RMSE score

print(sgd_gs.best_score['rmse'])



# combination of parameters that gave the best RMSE score

print(sgd_gs.best_params['rmse'])
trainset = data.build_full_trainset()

algo = sp.BaselineOnly()

algo.fit(trainset)

testset = trainset.build_anti_testset()

predictions = algo.test(testset)

    

last_predictions = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])

last_predictions.drop('rui', inplace = True, axis = 1)
def bringing_first_n_values(df, uid, n=10):

    df = df[df['uid'] == uid].nlargest(n, 'est')[['uid', 'iid', 'est']]

    df = pd.merge(df, AnimesDF, left_on = 'iid', right_on = 'anime_id', how = 'left')

    return df[['uid', 'est', 'title', 'genre']]
bringing_first_n_values(last_predictions, 'Tomoki-sama')
sim_options = {'name': 'pearson_baseline', 'user_based': False}

algo_items = sp.KNNBaseline(sim_options=sim_options)

algo_items.fit(trainset)
def get_item_recommendations(anime_title, anime_id=100000, k=10):

    if anime_id == 100000:     

        anime_id = AnimesDF[AnimesDF['title'] == anime_title]['anime_id'].iloc[0]

        

    iid = algo_items.trainset.to_inner_iid(anime_id)

    neighbors = algo_items.get_neighbors(iid, k=k)

    raw_neighbors = (algo.trainset.to_raw_iid(inner_id) for inner_id in neighbors)

    df = pd.DataFrame(raw_neighbors, columns = ['Anime_ID'])

    df = pd.merge(df, AnimesDF, left_on = 'Anime_ID', right_on = 'anime_id', how = 'left')

    return df[['Anime_ID', 'title', 'genre']]
get_item_recommendations('Dragon Ball Z', k=30)