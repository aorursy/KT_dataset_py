import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from surprise import Reader, Dataset, SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF

from surprise.model_selection import cross_validate, KFold
data = Dataset.load_from_file('../input/ml-100k/u.data', reader= Reader())



ratings = pd.read_csv('../input/ml-100k/u.data', sep='\t', header=None)

ratings.columns = ['user_id', 'item_id', 'rating', 'timestamp']
items_cols = ['movie_id' , 'movie_title' , 'release_date' , 'video_release_date' , 'IMDb_URL' , 'unknown|' , 'Action|' , 'Adventure|', 'Animation|', "Children's|", 'Comedy|', 'Crime|', 'Documentary|', 'Drama|',\

              'Fantasy|', 'Film-Noir|', 'Horror|', 'Musical|', 'Mystery|', 'Romance|', 'Sci-Fi|', 'Thriller|', \

              'War|', 'Western|']

movies = pd.read_csv('../input/ml-100k/u.item', sep='|', encoding='latin-1', names=items_cols, parse_dates=True, index_col='movie_id')
ratings.head()
ratings.shape
movies.head()
uniq = ratings.user_id.unique()

len(uniq)
ratings = ratings.rating

ratings.value_counts().sort_index().plot.bar()
ratings.describe()
model_random = NormalPredictor()
model_random_results = cross_validate(model_random, data, measures=['RMSE'], cv=5, verbose=True)
model_user_based = KNNBasic(sim_options={'user_based': True})
model_user_based_results = cross_validate(model_user_based, data, measures=['RMSE'], cv=5, verbose=True)
model_item_based = KNNBasic(sim_options={'user_based': False})
model_item_based_results = cross_validate(model_item_based, data, measures=['RMSE'], cv=5, verbose=True)
model_mat_fac = SVD()
model_mat_fac_results = cross_validate(model_mat_fac, data, measures=['RMSE'], cv=5, verbose=True)
means = [round(model_random_results['test_rmse'].mean(),4),round(model_user_based_results['test_rmse'].mean(),4), round(model_item_based_results['test_rmse'].mean(),4), round(model_mat_fac_results['test_rmse'].mean(),4)]

table = pd.Series(means, ['Random','User-based', 'Item-based', 'Matrix factorization'])

print("\t RMSE Means for each model\n")

print(table)
def get_top_n(predictions, n=5):

    # First map the predictions to each user.

    top_n = dict()

    for uid, iid, true_r, est, _ in predictions:

        current = top_n.get(uid, [])

        current.append((iid, movies.loc[int(iid),'movie_title'], round(est,2)))

        top_n[uid] = current



    # Then sort the predictions for each user and retrieve the k highest ones.

    for uid, user_ratings in top_n.items():

        user_ratings.sort(key=lambda x: x[2], reverse=True)

        top_n[uid] = user_ratings[:n]



    return top_n
trainset = data.build_full_trainset()

testset = trainset.build_anti_testset()
models = [model_random, model_user_based, model_item_based, model_mat_fac]
listt = []

for model in models:

    model.fit(trainset)

    predictions = model.test(testset)

    top_n = get_top_n(predictions, n=5)

    user = list(top_n.keys())[0]

    print('User:',user)

    print('Model:',model)

    table = pd.DataFrame(top_n[user], columns=['Movie ID','Movie Name', 'rating'])

    print(table)

    print('\n')

    