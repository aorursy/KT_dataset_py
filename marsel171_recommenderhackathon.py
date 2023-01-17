# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

from collections import Counter

import math



import pandas as pd

import numpy as np



from sklearn.utils import shuffle



import matplotlib.pyplot as plt

import seaborn 

seaborn.set()



from IPython.core.display import display, HTML

from collections import defaultdict

from sklearn.decomposition import TruncatedSVD

from scipy.sparse import csr_matrix



from sklearn.feature_extraction.text import CountVectorizer



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
ratings = pd.read_csv('../input/train.csv') #обучающие данные

movies = pd.read_csv('../input/movies.csv') #файл с мета-информацией о фильмах

tags = pd.read_csv('../input/tags.csv') #файл с мета-информацией о соответствующих тегах
##подключение тэгов

#un_tags = pd.unique(tags.tag)

#for i in range(len(un_tags)):

#    tagsi = tags[tags.tag == un_tags[i]]

#    for j in range(tagsi.shape[0]):

#        c1 = np.where(ratings.userId - tagsi.userId.values[j] == 0)

#        c2 = np.where(ratings.movieId - tagsi.movieId.values[j] == 0)

#        a = np.intersect1d(c1, c2)

#        if a:

#            print(i,j)
print(ratings.shape)

ratings.tail()
print(movies.shape)

movies.tail()
movies.loc[movies['genres'] == '(no genres listed)']
vec = CountVectorizer()

xv = vec.fit_transform(movies.genres)    

movies_vec = pd.DataFrame(xv.toarray(),columns = vec.get_feature_names())

print(movies_vec.shape)

movies_vec.tail()
movies_vec_drop = movies_vec.drop(columns=['no','genres', 'listed'])

print(movies_vec_drop.shape)

movies_vec_drop.tail()
movies_all = pd.concat([movies,movies_vec_drop],axis=1)

movies_all = movies_all.drop('genres',axis=1);

print(movies_all.shape)

movies_all.tail()
ratings_part2 = movies_all.set_index('movieId').loc[ratings.movieId].reset_index().drop(['movieId','title'],axis=1);

print(ratings_part2.shape)

ratings_part2
ratings_new = pd.concat([ratings,ratings_part2],axis=1)

print(ratings_new.shape)

ratings_new
ratings_new2 = ratings_new.copy()

for i in range(len(ratings_part2.columns)):

    ratings_new2[ratings_part2.columns[i]] = ratings_new[ratings_part2.columns[i]].mul(ratings_new.rating-3)
userId_un = pd.unique(ratings_new.userId)

matrix_userId_un = np.zeros([len(userId_un),len(ratings_part2.columns)])

k=0

for i in userId_un:

   matrix_userId_un[k,:] = np.sum(ratings_new2[ratings_new.userId==i][ratings_part2.columns],axis=0)

   k+=1
rows=[];

cols=[];

vals=[];

for i in range(matrix_userId_un.shape[0]):

    for j in range(matrix_userId_un.shape[1]):

        if matrix_userId_un[i,j]>0:

            rows.append(i)

            cols.append(j+movies.shape[0])

            #cols.append(j)

            vals.append(matrix_userId_un[i,j])
ratings = ratings_new2.copy()
with open('../input/test_user_id.list', 'r') as file:

    test_user_id = file.read()

test_user_id = [int(user_id) for user_id in test_user_id.split(',')]
sorted_timestamps = sorted(ratings['timestamp'])

total_actions = len(sorted_timestamps)

border_timestamp = sorted_timestamps[int(total_actions*0.75)]

train = ratings[ratings.timestamp <= border_timestamp]

validation = ratings[ratings.timestamp > border_timestamp]

train.shape, validation.shape
#train = ratings[:int(ratings.shape[0] * 0.75)]

#validation = ratings[int(ratings.shape[0] * 0.75):]

#train.shape, validation.shape
K = 30

max_n = 35



x = [i for i in range(1, max_n)]

y = [(i <= K) * 1/math.log2(i + 1) for i in range(1, max_n)]



plt.figure(figsize=(10, 6))

plt.title("Относительная важность ошибки на i-й позиции метрики NDCG@{}".format(K))

plt.xlabel("Номер позиции")

plt.ylabel("Относительная важность ошибки")

plt.text(5, 0.1, """после {}й позиции происходит резкий скачок в ноль,

так как метрика NDCG@{} считает все позиции > {} неважными""".format(K, K, K), bbox=dict(facecolor='white', alpha=0.5))



plt.plot(x, y);

plt.show();
def dcg_at_k(r, k, method=0):

    """

    Args:

        r: Relevance scores (list or numpy) in rank order

            (first element is the first item)

        k: Number of results to consider

        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]

                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]

    Returns:

        Discounted cumulative gain

    """

    r = np.asfarray(r)[:k]

    if r.size:

        if method == 0:

            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))

        elif method == 1:

            return np.sum(r / np.log2(np.arange(2, r.size + 2)))

        else:

            raise ValueError('method must be 0 or 1.')

    return 0.







def ndcg_at_k(r, k, method=0):

    """

    Args:

        r: Relevance scores (list or numpy) in rank order

            (first element is the first item)

        k: Number of results to consider

        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]

                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]

    Returns:

        Normalized discounted cumulative gain

    """

    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)

    if not dcg_max:

        return 0.

    return dcg_at_k(r, k, method) / dcg_max
class TopRecommender(object):

    def fit(self, train_data):

        counts = Counter(train_data['movieId'])

        self.predictions = counts.most_common()

        

    def predict(self, user_id, n_recommendations=10):

        return [movie_id for movie_id, frequency in self.predictions[:n_recommendations]]
class SVDRecommender(object):

    

    def create_viewed_matrix(self,data):

        self.users = defaultdict(lambda: len(self.users))

        self.movies = defaultdict(lambda: len(self.movies))

        rows = data.userId.apply(lambda userId: self.users[userId])

        cols = data.movieId.apply(lambda movieId: self.movies[movieId])

        vals = [1.0]* len(cols)

        print(len(rows))

        self.viewed_matrix = csr_matrix((vals, (rows, cols)))

        

    def fit(self, data, n_components = 30):    

        

        self.top_recommender = TopRecommender()

        self.top_recommender.fit(data)

        

        #Имитация неявной обрктной связи

        data_implicit = data

        

        #Создаем словарии соотвествия

        #UserID -> Номер строки в матрице

        #MovieId -> Номер колонки в матрице

        self.users = defaultdict(lambda: len(self.users))

        self.movies = defaultdict(lambda: len(self.movies))

       

        #Создаем матрицу взаимодействий пользователь -> фильм

        rows = data.userId.apply(lambda userId: self.users[userId])

        cols = data.movieId.apply(lambda movieId: self.movies[movieId])

        #vals = [1.0]* len(cols)

        vals = data.rating-3

        self.interactions_matrix = csr_matrix((vals, (rows, cols)))

        

        #Обучаем модель SVD

        self.model = TruncatedSVD(n_components = n_components, algorithm='arpack') #algorithm='randomized'

        self.model.fit(self.interactions_matrix)

        

        #Обратный словарь колонка -> ID фильма. Понадобится для предсказаний 

        self.movies_reverse = {}

        for movie_id in self.movies:

            movie_idx = self.movies[movie_id]

            self.movies_reverse[movie_idx] = movie_id

            

    def fit_huge(self, data, rows_add, cols_add, vals_add, n_components = 30,lys=5):    

        

        self.top_recommender = TopRecommender()

        self.top_recommender.fit(data)

        

        #Имитация неявной обрктной связи

        data_implicit = data

        

        #Создаем словарии соотвествия

        #UserID -> Номер строки в матрице

        #MovieId -> Номер колонки в матрице

        self.users = defaultdict(lambda: len(self.users))

        self.movies = defaultdict(lambda: len(self.movies))

       

        #Создаем матрицу взаимодействий пользователь -> фильм

        rows = data.userId.apply(lambda userId: self.users[userId])

        cols = data.movieId.apply(lambda movieId: self.movies[movieId])

        #vals = [1.0]* len(cols)

        #vals = data.rating-3

        

        def wtime(times,maxw,maxtime,stdt):

            return maxw*np.exp(-(times-maxtime)**2/(2*stdt*stdt))

        maxtime=max(data.timestamp)

        stdt=lys*(365*3600*24)#YEARS

        vals = (data.rating-3)* wtime(data.timestamp,1,maxtime,stdt)

        

        rows = np.array(rows)

        rows_add = np.array(rows_add)

        cols = np.array(cols)

        cols_add = np.array(cols_add)

        vals = np.array(vals)

        vals_add = np.array(vals_add)

        rows_new = np.concatenate([rows,rows_add])

        cols_new = np.concatenate([cols,cols_add])

        vals_new = np.concatenate([vals,vals_add])

        

        self.interactions_matrix = csr_matrix((vals_new, (rows_new, cols_new)))

        

        #Обучаем модель SVD

        self.model = TruncatedSVD(n_components = n_components, algorithm='arpack') #algorithm='randomized'

        self.model.fit(self.interactions_matrix)

        

        #Обратный словарь колонка -> ID фильма. Понадобится для предсказаний 

        self.movies_reverse = {}

        for movie_id in self.movies:

            movie_idx = self.movies[movie_id]

            self.movies_reverse[movie_idx] = movie_id           

        

    def predict(self, user_id, n_recommendations=10):  

        if user_id not in self.users:

            return self.top_recommender.predict(user_id,n_recommendations)

        

        #Получить представление пользователя в сниженной размерности     

        user_interactions = self.interactions_matrix.getrow(self.users[user_id])    

        user_low_dimensions = self.model.transform(user_interactions)  

        return self.predict_low_dimension(user_low_dimensions, user_interactions, n_recommendations)

    

    def predict_low_dimension(self, user_low_dimensions, user_interactions, max_n=10):

        #Получить приближенное представление пользователя

        user_predictions = self.model.inverse_transform(user_low_dimensions)[0]

        recommendations = []

        

        #Пробегаем по колонкам в порядке убывания предсказанного значения

        for movie_idx in reversed(np.argsort(user_predictions)):

            #Добавляем фильм к рекомендациям только если пользователь его еще не смотрел

            if user_interactions[0, movie_idx] == 0.0:

                movie = self.movies_reverse[movie_idx]

                score = user_predictions[movie_idx]

                #recommendations.append((movie, score))

                recommendations.append(movie)

            if (len(recommendations) >= max_n):

                return recommendations
#recommender_train = SVDRecommender()

#recommender_train.create_viewed_matrix(train)

#viewed_matrix_dense = recommender_train.viewed_matrix.todense()

#usrs = recommender_train.users

#movs = recommender_train.movies

#print(len(usrs))

#print(len(movs))
#recommender_train = TopRecommender()

recommender_train = SVDRecommender()

recommender_train.fit_huge(train, rows, cols, vals,n_components = 30,lys=0.1)
#from lightgbm import LGBMRanker

#lgbranker = LGBMRanker()

#csrmatrix = recommender_train.return_inter_matrix(train)

#lgbranker.fit(csrmatrix,train.userId)
verbose = True

num_to_print = 10

total_ndcg = 0



for user_id, group in validation.groupby('userId'):

    ground_truth_films = [int(data.movieId) for row, data in group.iterrows()]

    recommendations = recommender_train.predict(user_id, n_recommendations=10)

    relevance_scores = []

    for rec in recommendations:

        if rec in ground_truth_films:

            relevance_scores.append(len(ground_truth_films) - ground_truth_films.index(rec))

        else:

            relevance_scores.append(0)

    total_ndcg += ndcg_at_k(relevance_scores, k=10)

    

    if verbose and np.random.random() > 0.999:

        user_films_train = train[train.userId == user_id].movieId.values

        print('Идентификатор пользователя: ', user_id)

        print(

            'Фильмы в обучающей выборке для этого пользователя:',

            [movies[movies.movieId == movie_id].title.values[0] for movie_id in user_films_train[:num_to_print]],

            '\n'

        )

        print(

            'Просмотренные на самом деле фильмы: ', 

            [movies[movies.movieId == movie_id].title.values[0] for movie_id in ground_truth_films[:num_to_print]],

            '\n'

        )

        print(

            'Рекомендации топ-рекомендера: ', 

            [movies[movies.movieId == rec_id].title.values[0] for rec_id in recommendations],

            '\n'

        )

        print('Значение NDCG@10 = ', ndcg_at_k(relevance_scores, k=10), '\n\n')
total_ndcg / validation.shape[0]
#recommender = TopRecommender()

recommender = SVDRecommender()

recommender.fit_huge(ratings, rows, cols, vals,n_components = 30,lys=0.1)
recommender.predict(user_id=test_user_id[0], n_recommendations=10)
with open('submity.csv', 'w') as f:

    f.write('userId,movieId\n')

    for user_id in test_user_id:

        recommendations = recommender.predict(user_id=user_id, n_recommendations=10)

        for rec in recommendations:

            f.write(str(user_id) + ',' + str(int(rec)) + '\n')

    print('Отлично! Время загрузить файл submit.csv на kaggle!')