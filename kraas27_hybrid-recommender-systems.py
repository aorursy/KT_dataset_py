import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
links = pd.read_csv('../input/links.csv')

movies = pd.read_csv('../input/movies.csv')

ratings = pd.read_csv('../input/ratings.csv')

tags = pd.read_csv('../input/tags.csv')
movies_and_ratings = movies.merge(ratings, on='movieId')

movies_and_ratings.head(3)
mean_rating = movies_and_ratings.groupby('userId')['rating'].mean()

movies_and_ratings['mean_rating'] = movies_and_ratings['userId'].apply(lambda x: mean_rating[x])
movies_and_ratings['good_rating'] = movies_and_ratings.apply(lambda x: x['rating'] if x['mean_rating'] <= x['rating'] else np.NaN, axis=1)

movies_and_ratings = movies_and_ratings[ pd.isnull( movies_and_ratings['good_rating'] ) == 0 ]

movies_and_ratings = movies_and_ratings.drop(['mean_rating', 'good_rating'], axis=1).reset_index(drop=True)

movies_and_ratings.head()
good_feedback = movies_and_ratings
good_feedback = movies_and_ratings.sort_values(['userId' ,'timestamp'], ascending=[True, False])
good_feedback_dict = {}



all_users = good_feedback['userId'].unique()



for user in all_users:

    top_movies = []

    for top in range(5):

        try:

            top_movies.append(good_feedback[good_feedback['userId']==user]['movieId'].values[top])

            good_feedback_dict[user] = top_movies

        except:

            continue
movies_and_ratings['userId'] = movies_and_ratings['userId'].astype("category").cat.codes

movies_and_ratings['movieId'] = movies_and_ratings['movieId'].astype("category").cat.codes
shape_0 = len(movies_and_ratings['movieId'].unique())

shape_1 = len(movies_and_ratings['userId'].unique())
users_act = movies_and_ratings.loc[:, ['userId','movieId']].reset_index(drop=True)

users_act['act'] = 1

users_act.head(3)
activity = list(users_act['act'])

cols = users_act['movieId'].astype(int)

rows = users_act['userId'].astype(int)
len(rows), len(activity), len(cols)
from scipy import sparse

data_sparse = sparse.csr_matrix((activity, (rows, cols)), shape=(shape_1, shape_0))
from implicit.als import AlternatingLeastSquares

algo_0 = AlternatingLeastSquares(factors=50)

algo_0.fit(data_sparse)
userid = 1



user_items = data_sparse.T.tocsr()

recommendations = algo_0.recommend(userid, user_items, N=15)
recommendations_list = []

for i in recommendations:

    recommendations_list.append(i[0])
movies.iloc[recommendations_list]
movies_ = movies.copy()

movies_['description'] = movies_.apply(lambda x: x['genres'].replace('|', ' '), axis=1)
movies_ = movies_.drop('genres', axis=1)

movies_.head(3)
movies_list = []

description_list = []



for mov, desc in movies_[['title', 'description']].values:

    movies_list.append(mov)

    description_list.append(desc)
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

coutn_v = CountVectorizer()

X_train = coutn_v.fit_transform(description_list)

X_train.toarray(), X_train.toarray().shape
tfidf = TfidfTransformer()

X_train_col = tfidf.fit_transform(X_train)

X_train_col.toarray(), X_train_col.toarray().shape
for i in range(X_train_col.shape[1]):

    col_name = 'd{}'.format(i)

    movies_[col_name] = pd.Series(X_train_col.toarray()[:, i])
movies_ = movies_.drop('description', axis=1)

movies_.head(3)
train_data = movies_.iloc[:, 2:]

test_data = movies_[movies_['title'] == 'Jumanji (1995)'].iloc[:, 2:]
from sklearn.neighbors import NearestNeighbors

neighbor = NearestNeighbors(n_neighbors=10, n_jobs=-1, metric='euclidean')

neighbor.fit(train_data)
predict = neighbor.kneighbors(test_data, return_distance=True)

movies.iloc[predict[1][0]]
from surprise import KNNBasic, Dataset, Reader, accuracy, SVD

from surprise.model_selection import train_test_split, GridSearchCV
movies_ratings = movies.join(ratings.set_index('movieId'), on='movieId').reset_index(drop=True)

movies_ratings.dropna(inplace=True)

movies_ratings.head(3)
dataset = pd.DataFrame({

    'uid': movies_ratings.userId,

    'iid': movies_ratings.title,

    'rating': movies_ratings.rating

})
dataset.head(3) 
ratings.rating.min()
ratings.rating.max()
reader = Reader(rating_scale=(0.5, 5.0))

data = Dataset.load_from_df(dataset, reader)
trainset, testset = train_test_split(data, test_size=.15, random_state=42)
params = {'k':np.arange(10, 101, 10),

          'sim_options': {'name': ['pearson_baseline'], 'user_based': [True]}

         }

grid_algo = GridSearchCV(KNNBasic, params, measures=['rmse', 'mae'], cv=5, n_jobs=-1)

grid_algo.fit(data)
grid_algo.best_params
algo_1 = KNNBasic(k=40, sim_options={'name': 'pearson_baseline', 'user_based': True})

algo_1.fit(trainset)
test_pred = algo_1.test(testset)

accuracy.rmse(test_pred, verbose=True)
algo_2 = SVD(n_factors=20, n_epochs=20)

algo_2.fit(trainset)
test_pred = algo_2.test(testset)

accuracy.rmse(test_pred, verbose=True)
def get_movies(user):

    '''первая часть каскада собранная из предсказания ALS и ближайших соседей к трем понравившимся 

    фильмам'''

    list_for_user = []

    recommendations = algo_0.recommend(user, user_items, N=15)

    for i in recommendations:

        list_for_user.append(i[0])

        

    films = np.random.choice(good_feedback_dict[user], 3)

    for film in films: 

        data_for_pred = movies_[movies_['movieId'] == film].iloc[:, 2:]

        predict = neighbor.kneighbors(data_for_pred, return_distance=True)

        for i in predict[1][0]:

            if i is not list_for_user:

                list_for_user.append(i)



    '''Вторая часть каскада. Находим с помошью двух обученных алгоритмов и усрядняем их оценку'''

    

    user_movies = movies_and_ratings[movies_and_ratings.userId == user].title.unique()

    

    scores = []

    titles = []

    for iid in movies_and_ratings.loc[movies_and_ratings['movieId'].isin(list_for_user)].title.unique():

        if iid is not user_movies:

            scores.append((algo_1.predict(user, iid).est + 

                          algo_2.predict(user, iid).est)/2)

            titles.append(iid)

        

        

    best_indexes = np.argsort(scores)[-10:]

    for i in reversed(best_indexes):

        print(titles[i], scores[i])
get_movies(27)