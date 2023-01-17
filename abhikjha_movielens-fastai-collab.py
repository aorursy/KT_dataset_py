# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy as sp

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns



# Standard plotly imports

#import plotly.plotly as py

import plotly.graph_objs as go

import plotly.tools as tls

from plotly.offline import iplot, init_notebook_mode

#import cufflinks

#import cufflinks as cf

import plotly.figure_factory as ff



# Using plotly + cufflinks in offline mode

init_notebook_mode(connected=True)

#cufflinks.go_offline(connected=True)



# Preprocessing, modelling and evaluating

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix, roc_auc_score

from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold

from xgboost import XGBClassifier

import xgboost as xgb



## Hyperopt modules

from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING

from functools import partial



import os

import gc



import matplotlib.pyplot as plt



pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

plt.style.use('fivethirtyeight')



plt.figure(figsize=(25,25))



import pandas_profiling as pp



%reload_ext autoreload

%autoreload 2

%matplotlib inline



!pip install fastai==1.0.57

import fastai



from fastai import *

from fastai.tabular import *

from fastai.collab import *



# from torchvision.models import *

# import pretrainedmodels



# from utils import *

import sys



from fastai.callbacks.hooks import *



from fastai.callbacks.tracker import EarlyStoppingCallback

from fastai.callbacks.tracker import SaveModelCallback



import pandas as pd 

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

from sklearn.manifold import TSNE



from sklearn.experimental import enable_hist_gradient_boosting 

from sklearn.ensemble import HistGradientBoostingClassifier



import gc

from datetime import datetime 

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score



from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier

from catboost import CatBoostClassifier

from sklearn import svm

import lightgbm as lgb

from lightgbm import LGBMClassifier

import xgboost as xgb

from xgboost.sklearn import XGBClassifier  

from xgboost.sklearn import XGBRegressor



from scipy.special import erfinv

import matplotlib.pyplot as plt

import torch

from torch.utils.data import *

from torch.optim import *

from fastai.tabular import *

import torch.utils.data as Data

from fastai.basics import *

from fastai.callbacks.hooks import *

from tqdm import tqdm_notebook as tqdm



from hyperopt import hp, tpe

from hyperopt.fmin import fmin

from hyperopt import STATUS_OK



from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.metrics import make_scorer

from surprise import Reader, Dataset, SVD

from surprise.model_selection import cross_validate, train_test_split

from sklearn.decomposition import PCA



import warnings

warnings.filterwarnings('ignore')



sns.set_style('whitegrid')
df_mov = pd.read_csv('../input/movielens-100k/ml-latest-small/movies.csv')

df_tag = pd.read_csv('../input/movielens-100k/ml-latest-small/tags.csv', sep=',', parse_dates=['timestamp'])

df_rating = pd.read_csv('../input/movielens-100k/ml-latest-small/ratings.csv', sep=',', parse_dates=['timestamp'])



df_mov.shape, df_tag.shape, df_rating.shape
df_mov.head(2)
df_tag.head(2)
df_rating.head(2)
df_rating.rating.value_counts(normalize=True)
df_mov.isnull().any()
df_rating.isnull().any()
movies_with_ratings = df_mov.merge(df_rating, on='movieId', how='inner')

movies_with_ratings.shape
gc.collect()
movies_with_ratings.head(10)
import collections



collections.Counter(" ".join(movies_with_ratings['genres']).split("|")).most_common(10)
collections.Counter(" ".join(movies_with_ratings['genres']).split("|")).most_common()[-10:]
movie_stats = movies_with_ratings.groupby('title').agg({'rating': [np.size, np.mean]})

movie_stats.head()
atleast_100 = movie_stats['rating']['size'] >= 100

movie_stats[atleast_100].sort_values([('rating', 'mean')], ascending=False)[:15]
movie_stats[atleast_100].sort_values([('rating', 'mean')], ascending=True)[:15]
movies_with_ratings.shape
pivot = movies_with_ratings.pivot(index='userId', columns='movieId', values='rating')

pivot.replace(np.nan, 0, inplace=True, regex=True)

pivot.head(6)
print(pivot.shape)

from sklearn.preprocessing import StandardScaler 

X_std = StandardScaler().fit_transform(pivot)

del pivot
gc.collect()
cov_mat = np.cov(X_std.T)

evals, evecs = np.linalg.eig(cov_mat)

del X_std
k = 10

movieId = 1 # Grab an id from movies.dat

top_n = 5



def top_cosine_similarity(data, movieId, top_n=10):

    index = movieId - 1 # Movie id starts from 1

    movie_row = data[index, :]

    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))

    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)

    sort_indexes = np.argsort(-similarity)

    return sort_indexes[:top_n]



# Helper function to print top N similar movies

def print_similar_movies(movie_data, movieId, top_indexes):

    print('Recommendations for {0}: \n'.format(

    movie_data[movies_with_ratings.movieId == movieId].title.values[0]))

    for id in top_indexes + 1:

        print(movie_data[movies_with_ratings.movieId == id].title.values[0])



sliced = evecs[:, :k] # representative data

indexes = top_cosine_similarity(sliced, movieId, top_n)

print_similar_movies(movies_with_ratings, movieId, indexes)
df_mov.head(10)
df_rating.head(10)
R_df = df_rating.pivot(index = 'userId', columns='movieId', values='rating').fillna(0)

R_df.head()
R = R_df.as_matrix()

user_ratings_mean = np.mean(R, axis = 1)

R_demeaned = R - user_ratings_mean.reshape(-1, 1)
from scipy.sparse.linalg import svds

U, sigma, Vt = svds(R_demeaned, k = 50)
sigma = np.diag(sigma)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)
def recommend_movies(predictions_df, userId, movies_df, original_ratings_df, num_recommendations=5):

    

    # Get and sort the user's predictions

    user_row_number = userId - 1 # UserID starts at 1, not 0

    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)

    

    # Get the user's data and merge in the movie information.

    user_data = original_ratings_df[original_ratings_df.userId == (userId)]

    user_full = (user_data.merge(df_mov, how = 'left', left_on = 'movieId', right_on = 'movieId').

                     sort_values(['rating'], ascending=False)

                 )



    print ('User {0} has already rated {1} movies.'.format(userId, user_full.shape[0]))

    print ('Recommending the highest {0} predicted ratings movies not already rated.'.format(num_recommendations))

    

    # Recommend the highest predicted rating movies that the user hasn't seen yet.

    recommendations = (df_mov[~df_mov['movieId'].isin(user_full['movieId'])].

         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',

               left_on = 'movieId',

               right_on = 'movieId').

         rename(columns = {user_row_number: 'Predictions'}).

         sort_values('Predictions', ascending = False).

                       iloc[:num_recommendations, :-1]

                      )



    return user_full, recommendations



already_rated, predictions = recommend_movies(preds_df, 8, df_mov, df_rating, 5)
already_rated.dropna().head(5)
predictions
mov_with_rating = movies_with_ratings[['userId', 'title', 'rating']]

reader = Reader(rating_scale=(1,5))

data = Dataset.load_from_df(mov_with_rating, reader)
train, test = train_test_split(data, test_size=0.25)
model = SVD(n_factors=200)

model.fit(train)
item_to_row_idx: Dict[Any, int] = model.trainset._raw2inner_id_items

toy_story_row_idx : int = item_to_row_idx['Toy Story (1995)']
model.qi[toy_story_row_idx]
movies_with_ratings.head(2)
a_user = 196

a_product = 'Toy Story (1995)'



model.predict(a_user, a_product)
predictions = model.test(test)



from surprise import accuracy



accuracy.rmse(predictions)
Mapping_file = dict(zip(movies_with_ratings.title.tolist(), movies_with_ratings.movieId.tolist()))
def pred_user_rating(ui):

    if ui in movies_with_ratings.userId.unique():

        ui_list = movies_with_ratings[movies_with_ratings.userId == ui].movieId.tolist()

        d = {k: v for k,v in Mapping_file.items() if not v in ui_list}        

        predictedL = []

        for i, j in d.items():     

            predicted = model.predict(ui, j)

            predictedL.append((i, predicted[3])) 

        pdf = pd.DataFrame(predictedL, columns = ['movies', 'ratings'])

        pdf.sort_values('ratings', ascending=False, inplace=True)  

        pdf.set_index('movies', inplace=True)    

        return pdf.head(10)        

    else:

        print("User Id does not exist in the list!")

        return None
user_id = 87

pred_user_rating(user_id)
from collections import defaultdict



from surprise import SVD

from surprise import Dataset





def get_top_n(predictions, n=10):

    '''Return the top-N recommendation for each user from a set of predictions.



    Args:

        predictions(list of Prediction objects): The list of predictions, as

            returned by the test method of an algorithm.

        n(int): The number of recommendation to output for each user. Default

            is 10.



    Returns:

    A dict where keys are user (raw) ids and values are lists of tuples:

        [(raw item id, rating estimation), ...] of size n.

    '''



    # First map the predictions to each user.

    top_n = defaultdict(list)

    for uid, iid, true_r, est, _ in predictions:

        top_n[uid].append((iid, est))



    # Then sort the predictions for each user and retrieve the k highest ones.

    for uid, user_ratings in top_n.items():

        user_ratings.sort(key=lambda x: x[1], reverse=True)

        top_n[uid] = user_ratings[:n]



    return top_n





# First train an SVD algorithm on the movielens dataset.

data = data

trainset = data.build_full_trainset()

algo = SVD()

algo.fit(trainset)



# Than predict ratings for all pairs (u, i) that are NOT in the training set.

testset = trainset.build_anti_testset()

predictions = algo.test(testset)



top_n = get_top_n(predictions, n=3)



# Print the recommended items for each user

for uid, user_ratings in top_n.items():

    print(uid, [iid for (iid, _) in user_ratings])
movies_with_ratings.drop(['genres'], axis=1, inplace=True)
movies_with_ratings.head()
data = CollabDataBunch.from_df(movies_with_ratings, seed=42, valid_pct=0.1, item_name='title', user_name='userId', 

                              rating_name='rating')
data.show_batch()
y_range = [0,5.5]

learn = collab_learner(data, n_factors=40, y_range=y_range, use_nn=True, layers=[256, 128])

learn.loss = torch.nn.SmoothL1Loss

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(5, 1e-3, wd=[0.1])
learn.recorder.plot_losses()
movies_with_ratings.iloc[250]
learn.predict(movies_with_ratings.iloc[250])
learn.get_preds(ds_type=DatasetType.Valid)
def pred_user_rating(ui):

    if ui in movies_with_ratings.userId.unique():

        ui_list = movies_with_ratings[movies_with_ratings.userId == ui].movieId.tolist()

        d = {k: v for k,v in Mapping_file.items() if not v in ui_list}

        

        predictedL = []

        for i, j in d.items():     

            predicted = learn.predict(movies_with_ratings.iloc[ui])

            predictedL.append((i, predicted[0])) 

        pdf = pd.DataFrame(predictedL, columns = ['movies', 'ratings'])

        #pdf.sort_values('ratings', ascending=False, inplace=True)  

        pdf.set_index('movies', inplace=True)    

        return pdf.head(10)        

    else:

        print("User Id does not exist in the list!")

        return None
predicted = learn.predict(movies_with_ratings.iloc[8])

predicted[0]
user_id = 9

pred_user_rating(user_id)
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate, Dropout

from keras.models import Model
mov_with_rating = movies_with_ratings[['userId', 'movieId', 'rating']]
from sklearn.model_selection import train_test_split

train, test = train_test_split(mov_with_rating, test_size=0.2, random_state=42)
train.shape, test.shape
train.head()
n_users = len(train.userId.unique())
n_movies = len(train.movieId.unique())
n_users, n_movies
# Create Movie Embedding

movie_input = Input(shape=[1], name="Movie-Input")

movie_embedding = Embedding(200000, 5, name='Movie-Embedding')(movie_input)

movie_vec = Flatten(name='Movie-Vector')(movie_embedding)
# Create User Embedding

user_input = Input(shape=[1], name="User-Input")

user_embedding = Embedding(200000, 5, name='User-Embedding')(user_input)

user_vec = Flatten(name='User-Vector')(user_embedding)
# Concatenate two vectors



conc = Concatenate()([user_vec, movie_vec])
# fully connected layers



from keras.models import Sequential

model = Sequential()



fc1 = Dense(128, activation='relu')(conc)

fc1 = Dropout(0.3)(fc1)

fc2 = Dense(32, activation='relu')(fc1)

fc2 = Dropout(0.3)(fc2)

out = Dense(1)(fc2)



# Create Model and Compile it



model = Model([user_input, movie_input], out)

model.compile('adam', 'mean_squared_error')
from keras.models import load_model



if os.path.exists('regression_model.h5'):

    model = load_model('regression_model.h5')

else:

    history = model.fit([train.userId, train.movieId], train.rating, epochs=15, verbose=1)

    model.save('regression_model.h5')

    plt.plot(history.history['loss'])

    plt.xlabel("Epochs")

    plt.ylabel("Training Error")
model.evaluate([test.userId, test.movieId], test.rating)
predictions = model.predict([test.userId.head(10), test.movieId.head(10)])



[print(predictions[i], test.rating.iloc[i]) for i in range(0,10)]
# Extract embeddings

mov_em = model.get_layer('Movie-Embedding')

mov_em_weights = mov_em.get_weights()[0]
from sklearn.decomposition import PCA

import seaborn as sns



pca = PCA(n_components=2)

pca_result = pca.fit_transform(mov_em_weights)

sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1])
mov_em_weights.shape
movie_data = np.array(list(set(movies_with_ratings.movieId)))

movie_data[:5]
user_data = np.array([1 for i in range(len(movie_data))])

user_data[:5]
predictions = model.predict([user_data, movie_data])

predictions = np.array([a[0] for a in predictions])

predictions[:5]
recommended_movie_ids = (-predictions).argsort()[:5]



recommended_movie_ids
predictions[recommended_movie_ids]
movies_with_ratings[movies_with_ratings.movieId == 5877]
movies_with_ratings[movies_with_ratings['movieId'].isin(recommended_movie_ids)]