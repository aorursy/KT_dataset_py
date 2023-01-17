# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from collections import Counter

from collections import defaultdict

from IPython.core.display import display, HTML

from sklearn.decomposition import TruncatedSVD

from scipy.sparse import csr_matrix
ratings = pd.read_csv('../input/train.csv')

movies = pd.read_csv('../input/movies.csv')
with open('../input/test_user_id.list', 'r') as file:

    test_user_id = file.read()

test_user_id = [int(user_id) for user_id in test_user_id.split(',')]

test_user_id[:10]
sorted_timestamps = sorted(ratings['timestamp'])



total_actions = len(sorted_timestamps)

border_timestamp = sorted_timestamps[int(total_actions * 0.75)]



ratings_train = ratings[ratings.timestamp <= border_timestamp]

ratings_test = ratings[ratings.timestamp > border_timestamp]



ratings_train.shape, ratings_test.shape
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
class PrettyPrinter(object):

    def __init__(self, train, test, movies):

        self.movies = movies.set_index('movieId')

        self.train_data = train.set_index('userId')

        self.test_data = test.set_index('userId')

    

    def print_history(self, user_id, n=10):

        display(HTML("<h4>history for user {}<h4>".format(user_id)))

        if(user_id in self.train_data.index):

            user_data = self.train_data.loc[user_id].sort_values('timestamp', ascending=False)

            titles = user_data.movieId.apply(lambda movieId: self.movies.loc[movieId]['title'])

            genres = user_data.movieId.apply(lambda movieId: self.movies.loc[movieId]['genres']) 

            user_data['title'] = titles

            user_data['genres'] = genres

            new_index = list(range(1, len(user_data) + 1))

            user_data = user_data[['title', 'genres', 'rating']]

            user_data.index = new_index

            display(user_data.head(n))

        

    def print_reality(self, user_id, n=10):

        display(HTML("<h4>reality for user {}<h4>".format(user_id)))

        if(user_id in self.test_data.index):

            user_data = self.test_data.loc[user_id].sort_values('timestamp', ascending=True)

            titles = user_data.movieId.apply(lambda movieId: self.movies.loc[movieId]['title'])

            genres = user_data.movieId.apply(lambda movieId: self.movies.loc[movieId]['genres']) 

            user_data['title'] = titles

            user_data['genres'] = genres

            new_index = list(range(1, len(user_data) + 1))

            user_data = user_data[['title', 'genres', 'rating']]

            user_data.index = new_index

            display(user_data.head(n))

        

    def print_user_rec(self, user_id, recommender):

        display(HTML("<h4>recommendations for user {}<h4>".format(user_id)))

        recommendations = recommender.predict(user_id)

        self.print_recommendations(recommendations)

    

    def print_recommendations(self, recommendations):

        pos = 0

        index = []

        recom = []

        positions = []

        for rec in recommendations:

            pos += 1

            positions.append(pos)

            recom.append({'title': self.movies.loc[rec]['title'], 

                          'genres': self.movies.loc[rec]['genres']})

        df = pd.DataFrame(recom, columns=['title', 'genres'])

        display(df)

        

    def print_ndcg(self, user_id, recommender):

        recommendations = recommender.predict(user_id)

        ground_truth_films = list(self.test_data.loc[user_id].sort_values('timestamp', ascending=True)['movieId'].values)

        l = len(ground_truth_films)

        relevance_scores = []

        for rec in recommendations:

            if rec in ground_truth_films:

                relevance_scores.append(l - ground_truth_films.index(rec))

            else:

                relevance_scores.append(0)

        display(HTML("<h3>NDCG@10 = ({}) for user {}<h3>".format(ndcg_at_k(relevance_scores, k=10), user_id)))

        

    def print_user_info(self, user_id, recommender):

        self.print_history(user_id)

        display(HTML("</br>"))

        self.print_user_rec(user_id, recommender)

        display(HTML("</br>"))

        self.print_reality(user_id)

        display(HTML("</br>"))

        self.print_ndcg(user_id, recommender)

        

pretty_printer = PrettyPrinter(ratings_train, ratings_test, movies)
model = TopRecommender()

model.fit(ratings_train)
pretty_printer.print_user_info(79366, model)
with open('tr_submit.csv', 'w') as f:

    f.write('userId,movieId\n')

    for user_id in test_user_id:

        recommendations = model.predict(user_id)

        for rec in recommendations:

            f.write(str(user_id) + ',' + str(int(rec)) + '\n')

    print('Отлично! Время загрузить файл tr_submit.csv на kaggle!')
class SVDRecommender(object):

    def fit(self, data, n_components=99):

        

        #Обучим рекомендер для отсутствующих юзеров

        self.top_recommender = TopRecommender()

        self.top_recommender.fit(data)

        

        #Создаем словарии соотвествия

        #UserID -> Номер строки в матрице

        #MovieId -> Номер колонки в матрице

        self.users = defaultdict(lambda: len(self.users))

        self.movies = defaultdict(lambda: len(self.movies))

       

        #Создаем матрицу взаимодействий пользователь -> фильм

        rows = data.userId.apply(lambda userId: self.users[userId])

        cols = data.movieId.apply(lambda movieId: self.movies[movieId])

        vals = [1.0]* len(cols)

        self.interactions_matrix = csr_matrix((vals, (rows, cols)))

                

        #Обучаем модель SVD

        self.model = TruncatedSVD(n_components=n_components, algorithm='arpack')

        self.model.fit(self.interactions_matrix)

        

        #Обратный словарь колонка -> ID фильма. Понадобится для предсказаний 

        self.movies_reverse = {}

        for movie_id in self.movies:

            movie_idx = self.movies[movie_id]

            self.movies_reverse[movie_idx] = movie_id

        

    def predict(self, user_id, n_recommendations=10):        

        #для отсутствующих юзеров

        if user_id not in self.users:

            return self.top_recommender.predict(user_id, n_recommendations)



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

                recommendations.append(movie)

                

            if (len(recommendations) >= max_n):

                return recommendations
model = SVDRecommender()

model.fit(ratings_train)
pretty_printer.print_user_info(79366, model)
with open('svd_submit.csv', 'w') as f:

    f.write('userId,movieId\n')

    for user_id in test_user_id:

        recommendations = model.predict(user_id)

        for rec in recommendations:

            f.write(str(user_id) + ',' + str(int(rec)) + '\n')

    print('Отлично! Время загрузить файл svd_submit.csv на kaggle!')