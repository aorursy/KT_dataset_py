# Visualisation

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

import missingno as msno





# Configure visualisations

%matplotlib inline

mpl.style.use( 'ggplot' )

plt.style.use('fivethirtyeight')

sns.set(context="notebook", palette="dark", style = 'whitegrid' , color_codes=True)



from IPython.core.display import HTML

HTML("""

<style>

.output_png {

    display: table-cell;

    text-align: center;

    vertical-align: middle;

}

</style>

""");





# Make Visualizations better

params = { 

    'axes.labelsize': "large",

    'xtick.labelsize': 'x-large',

    'legend.fontsize': 20,

    'figure.dpi': 150,

    'figure.figsize': [25, 7]

}

plt.rcParams.update(params)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# IMPORTING THE DATA

# Data set columns

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']

users = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.user', sep='|', names=u_cols,  encoding='latin-1')



# Removing duplicates

users = users.drop_duplicates(keep='first')



# Converting columns to specific data type

user_int_columns = ['user_id', 'age']

users[user_int_columns] = users[user_int_columns].applymap(np.int64)



total_users = int(users.shape[0])
users.head()
users.dtypes
i_cols = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action',

          'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',

          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']



movies = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')



# Removing duplicates

movies = movies.drop_duplicates(keep='first')



# Dropping the unnecessary columns

movies.drop(['Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary',

                          'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',

                          'Thriller', 'War', 'Western', 'video release date', 'IMDb URL'], axis=1, inplace=True)



total_movies = int(movies.shape[0])
movies.head()
movies.dtypes
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.data', sep='\t', names=r_cols, encoding='latin-1')



# Dropping the timestamp column

ratings.drop(['unix_timestamp'], axis=1, inplace=True)



# Removing duplicates

ratings = ratings.drop_duplicates(keep='first')



# Converting to appropriate data types

ratings_int_columns = ['user_id', 'movie_id', 'rating']

ratings[ratings_int_columns] = ratings[ratings_int_columns].applymap(np.int64)



total_ratings = int(ratings.shape[0])
ratings.head()
ratings.dtypes
frame = ratings.set_index('movie_id').join(movies.set_index('movie id'))

ratings_per_movie = frame['rating'].groupby(frame['movie title']).sum()



print('MOST RATED MOVIES')

print(ratings_per_movie.sort_values(ascending=False))



from plotly.offline import init_notebook_mode, plot, iplot

import plotly.graph_objs as go

init_notebook_mode(connected=True)



data = frame['rating'].value_counts().sort_index(ascending=False)

trace = go.Bar(x = data.index,

               text = ['{:.1f} %'.format(val) for val in (data.values / frame.shape[0] * 100)],

               textposition = 'auto',

               textfont = dict(color = '#000000'),

               y = data.values,

               )

# Create layout

layout = dict(title = 'Distribution Of {} movie-ratings'.format(frame.shape[0]),

              xaxis = dict(title = 'Rating'),

              yaxis = dict(title = 'Count'))

# Create plot

fig = go.Figure(data=[trace], layout=layout)

iplot(fig)
data_matrix = np.zeros((total_users, total_movies))

for line in ratings.itertuples():

    data_matrix[line[1]-1, line[2]-1] = line[3]

    

print("USER_MOVIE RATING MATRIX")

print(data_matrix)
from sklearn.metrics.pairwise import pairwise_distances 



user_similarity = pairwise_distances(data_matrix, metric='cosine')

movie_similarity = pairwise_distances(data_matrix.T, metric='cosine')



print("USER-USER SIMILARITY MATRIX")

print(user_similarity)

print('\n')

print("MOVIE-MOVIE SIMILARITY MATRIX")

print(movie_similarity)
def predict(ratings, similarity, typ='user'):

    if typ == 'user':

        mean_user_rating = ratings.mean(axis=1)

        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])

        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T

    elif typ == 'movie':

        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])

    return pred



user_user_prediction = predict(data_matrix, user_similarity, typ='user')

movie_movie_prediction = predict(data_matrix, movie_similarity, typ='movie')



print('USER-USER PREDICTION MATRIX')

print(user_user_prediction)

print()

print('MOVIE-MOVIE PREDICTION MATRIX')

print(movie_movie_prediction)
from surprise import Reader, Dataset, KNNBasic, SVD, NMF, NormalPredictor, BaselineOnly, KNNWithMeans, KNNWithZScore, KNNBaseline, SVDpp, SlopeOne, CoClustering

from surprise.model_selection import GridSearchCV, cross_validate



reader = Reader(rating_scale=(0.0, 5.0))

data = Dataset.load_from_df( ratings[['user_id', 'movie_id', 'rating']], reader = reader )

sim_options = {'name': 'msd',

               'user_based': False  # compute  similarities between items

               }

algo = KNNBasic(k=30,sim_options=sim_options)

cross_validate(algo=algo, data=data, measures=['RMSE'], cv=5, verbose=True)
n_neighbours = [10, 20, 30]

param_grid = {'n_neighbours' : n_neighbours,'k':n_neighbours,'user_based':[True,False]}



gs = GridSearchCV(KNNBasic, measures=['RMSE'], param_grid=param_grid)

gs.fit(data)



print('\n\n\n')

# Best RMSE score

print('Best Score :', gs.best_score['rmse'])



# Combination of parameters that gave the best RMSE score

print('Best Parameters :', gs.best_params['rmse'])
algo = SVD()

cross_validate(algo=algo, data=data, measures=['RMSE'], cv=5, verbose=True)
param_grid = {'n_factors' : [50, 60, 70], 'lr_all' : [0.5, 0.05, 0.01], 'reg_all' : [0.06, 0.04, 0.02]}



gs = GridSearchCV(algo_class=SVD, measures=['RMSE'], param_grid=param_grid)

gs.fit(data)



print('\n\n\n')

# Best RMSE score

print('Best Score :', gs.best_score['rmse'])



# Combination of parameters that gave the best RMSE score

print('Best Parameters :', gs.best_params['rmse'])
cross_validate(algo=NMF(), data=data, measures=['RMSE'], cv=5, verbose=True)
param_grid = {'n_factors' : [50, 60, 70],'verbose':[True]}



gs = GridSearchCV(algo_class=NMF, measures=['RMSE'], param_grid=param_grid)

gs.fit(data)



print('\n\n\n')

# Best RMSE score

print('Best Score :', gs.best_score['rmse'])



# Combination of parameters that gave the best RMSE score

print('Best Parameters :', gs.best_params['rmse'])