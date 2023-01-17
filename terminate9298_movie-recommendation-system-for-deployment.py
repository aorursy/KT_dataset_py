import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import ast

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import linear_kernel

from sklearn.metrics import pairwise_distances

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split



from scipy.spatial.distance import cosine, correlation

from surprise import Reader, Dataset, SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF

from surprise.model_selection import cross_validate, KFold ,GridSearchCV , RandomizedSearchCV



from keras.models import Sequential

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.layers import  Input, dot, concatenate

from keras.models import Model

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM



import gc

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

pd.set_option('display.max_rows',50)

pd.set_option('display.max_columns', 50)
def reduce_mem_usage(df):

    # iterate through all the columns of a dataframe and modify the data type

    #   to reduce memory usage.        

    

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



    for col in df.columns:

        col_type = df[col].dtype



        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df
credits = pd.read_csv("../input/tmdb-movie-metadata/tmdb_5000_credits.csv")

movies = pd.read_csv("../input/tmdb-movie-metadata/tmdb_5000_movies.csv")
display(credits.head(5))

display(movies.head(5))
credits.columns = ['id','tittle','cast','crew']

movies= movies.merge(credits,on='id')

plots = movies['overview']

tfidf = TfidfVectorizer(stop_words = 'english' , max_df = 4 , min_df= 1)

plots = plots.fillna('')

tfidf_matrix = tfidf.fit_transform(plots)

cos_similar = linear_kernel(tfidf_matrix , tfidf_matrix)

cos_similar.shape
indices = pd.Series(movies.index , index = movies['title']).drop_duplicates()
def get_movies(title):

    idx = indices[title]

    similar = list(enumerate(cos_similar[idx]))

    similar = sorted(similar , key = lambda x: x[1] , reverse = True)

    similar = similar[:11]

    indic = []

    for i in similar:

        indic.append(i[0])

    return movies['title'].iloc[indic]

get_movies('Spider-Man 3')
get_movies('Toy Story')
readme= open('../input/movielens-100k-dataset/ml-100k/README','r') 

print(os.listdir('../input/movielens-100k-dataset/ml-100k'))

print(readme.read()) 
info = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.info' , sep=" ", header = None)

info.columns = ['Counts' , 'Type']



occupation = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.occupation' , header = None)

occupation.columns = ['Occupations']



items = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.item' , header = None , sep = "|" , encoding='latin-1')

items.columns = ['movie id' , 'movie title' , 'release date' , 'video release date' ,

              'IMDb URL' , 'unknown' , 'Action' , 'Adventure' , 'Animation' ,

              'Childrens' , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,

              'Film_Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci_Fi' ,

              'Thriller' , 'War' , 'Western']



data = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.data', header= None , sep = '\t')

user = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.user', header= None , sep = '|')

genre = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.genre', header= None , sep = '|' )



genre.columns = ['Genre' , 'genre_id']

data.columns = ['user id' , 'movie id' , 'rating' , 'timestamp']

user.columns = ['user id' , 'age' , 'gender' , 'occupation' , 'zip code']

display(info)

display(user.shape)

display(items.shape)

display(data.shape)
# Merging the columns with data table to better visualise

data = data.merge(user , on='user id')

data = data.merge(items , on='movie id')
# Data Cleaning for Model Based Recommandation System

def convert_time(x):

    return datetime.utcfromtimestamp(x).strftime('%d-%m-%Y')

def date_diff(date):

    d1 = date['release date'].split('-')[2]

    d2 = date['rating time'].split('-')[2]

    return abs(int(d2) - int(d1))



# data.drop(columns = ['movie title' , 'video release date' , 'IMDb URL'] , inplace = True)

data.dropna(subset = ['release date'] , inplace = True)



user_details = data.groupby('user id').size().reset_index()

user_details.columns = ['user id' , 'number of user ratings']

data = data.merge(user_details , on='user id')



movie_details = data.groupby('movie id').size().reset_index()

movie_details.columns = ['movie id' , 'number of movie ratings']

data = data.merge(movie_details , on='movie id')



user_details = data.groupby('user id')['rating'].agg('mean').reset_index()

user_details.columns = ['user id' , 'average of user ratings']

data = data.merge(user_details , on='user id')



movie_details = data.groupby('movie id')['rating'].agg('mean').reset_index()

movie_details.columns = ['movie id' , 'average of movie ratings']

data = data.merge(movie_details , on='movie id')





user_details = data.groupby('user id')['rating'].agg('std').reset_index()

user_details.columns = ['user id' , 'std of user ratings']

data = data.merge(user_details , on='user id')



movie_details = data.groupby('movie id')['rating'].agg('std').reset_index()

movie_details.columns = ['movie id' , 'std of movie ratings']

data = data.merge(movie_details , on='movie id')



data['age_group'] = data['age']//10

data['rating time'] = data.timestamp.apply(convert_time)

data['time difference'] = data[['release date' , 'rating time']].apply(date_diff, axis =1)



data['total rating'] = (data['number of user ratings']*data['average of user ratings'] + data['number of movie ratings']*data['average of movie ratings'])/(data['number of movie ratings']+data['number of user ratings'])

data['rating_new'] = data['rating'] - data['total rating']



del movie_details

del user_details
pivot_table_user = pd.pivot_table(data=data,values='rating_new',index='user id',columns='movie id')

pivot_table_user = pivot_table_user.fillna(0)

pivot_table_movie = pd.pivot_table(data=data,values='rating',index='user id',columns='movie id')

pivot_table_movie = pivot_table_movie.fillna(0)
user_based_similarity = 1 - pairwise_distances( pivot_table_user.values, metric="cosine" )

movie_based_similarity = 1 - pairwise_distances( pivot_table_movie.T.values, metric="cosine" )
user_based_similarity = pd.DataFrame(user_based_similarity)

user_based_similarity.columns = user_based_similarity.columns+1

user_based_similarity.index = user_based_similarity.index+1



movie_based_similarity = pd.DataFrame(movie_based_similarity)

movie_based_similarity.columns = movie_based_similarity.columns+1

movie_based_similarity.index = movie_based_similarity.index+1
# Testing movie based Recommendation



def rec_movie(movie_id):

    temp_table = pd.DataFrame(columns = items.columns)

    movies = movie_based_similarity[movie_id].sort_values(ascending = False).index.tolist()[:11]

    for mov in movies:

#         display(items[items['movie id'] == mov])

        temp_table = temp_table.append(items[items['movie id'] == mov], ignore_index=True)

    return temp_table

def rec_user(user_id):

    temp_table = pd.DataFrame(columns = user.columns)

    us = user_based_similarity[user_id].sort_values(ascending = False).index.tolist()[:101]

    for u in us:

#         display(items[items['movie id'] == mov])

        temp_table = temp_table.append(user[user['user id'] == u], ignore_index=True)

    return temp_table
display(rec_movie(176))

display(rec_movie(11))
def user_rating(x):

    similar_user = rec_user(x)

    similar_user.drop(columns= ['age' , 'gender' , 'occupation' , 'zip code'] , inplace = True)

    similar_user = similar_user.merge(pivot_table_movie , on= 'user id')

    similar_user = similar_user.set_index('user id')

    similar_user.replace(0, np.nan, inplace=True)

    u_ratings = similar_user[similar_user.index==x]

    similar_user.drop(similar_user.index[0] , inplace = True)

    return u_ratings.append(similar_user.mean(axis = 0 , skipna = True), ignore_index = True)   
display(user_rating(771))

display(user_rating(900))
reader = Reader(rating_scale=(1, 5))

sup_data = Dataset.load_from_df(data[['user id', 'movie title', 'rating']], reader)
algo = NormalPredictor()

cross_validate(algo, sup_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
algo = SVD()

cross_validate(algo, sup_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
algo = KNNBasic(k=20)

cross_validate(algo, sup_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
algo = KNNBasic(sim_options={'user_based': False} , k=20) # https://surprise.readthedocs.io/en/stable/prediction_algorithms.html#similarity-measure-configuration

cross_validate(algo, sup_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
algo = NMF()

cross_validate(algo, sup_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
sup_train = sup_data.build_full_trainset()

algo = SVD(n_factors = 200 , lr_all = 0.005 , reg_all = 0.02 , n_epochs = 40 , init_std_dev = 0.05)

algo.fit(sup_train)
def prediction_algo(uid = None , iid = None):

    predictions = []

    if uid is None:

        for ui in sup_train.all_users():

            predictions.append(algo.predict(ui, iid, verbose = False))

        return predictions

    

    if iid is None:

        for ii in sup_train.all_items():

            ii = sup_train.to_raw_iid(ii)

            predictions.append(algo.predict(uid, ii, verbose = False))

        return predictions

    return predictins.append(algo.predict(uid,iid,verbose = False))
predictions = prediction_algo(uid = 112)

predictions.sort(key=lambda x: x.est, reverse=True)

print('#### Best Recommanded Movies are ####')

for pred in predictions[:21]:

#     print('Movie -> {} with Score-> {}'.format(sup_train.to_raw_iid(pred.iid) , pred.est))

    print('Movie -> {} with Score-> {}'.format(pred.iid , pred.est))
meta_data = pd.read_csv('/kaggle/input/the-movies-dataset/movies_metadata.csv')

keywords = pd.read_csv('/kaggle/input/the-movies-dataset/keywords.csv')

credits = pd.read_csv('/kaggle/input/the-movies-dataset/credits.csv')



meta_data = meta_data[meta_data.id!='1997-08-20']

meta_data = meta_data[meta_data.id!='2012-09-29']

meta_data = meta_data[meta_data.id!='2014-01-01']

meta_data = meta_data.astype({'id':'int64'})



meta_data = meta_data.merge(keywords , on = 'id')

meta_data = meta_data.merge(credits , on = 'id')
def null_values(df):

    for col in df.columns:

        if df[col].isnull().sum() != 0:

            print('Total values missing in {} are {}'.format(col , df[col].isnull().sum()))

null_values(meta_data)
meta_data[meta_data['production_companies'].isnull()]

meta_data.dropna(subset=['production_companies'] , inplace = True)
def btc_function(data):

    if type(data) == str:

        return ast.literal_eval(data)['name'].replace(" ","")

    return data

# https://www.kaggle.com/hadasik/movies-analysis-visualization-newbie

def get_values(data_str):

    if isinstance(data_str, float):

        pass

    else:

        values = []

        data_str = ast.literal_eval(data_str)

        if isinstance(data_str, list):

            for k_v in data_str:

                values.append(k_v['name'].replace(" ",""))

            return str(values)[1:-1]

        else:

            return None

meta_data['btc_name'] = meta_data.belongs_to_collection.apply(btc_function)

meta_data[['genres', 'production_companies', 'production_countries', 'spoken_languages' ,'keywords','cast', 'crew']] = meta_data[['genres', 'production_companies', 'production_countries', 'spoken_languages' ,'keywords' ,'cast' , 'crew']].applymap(get_values)

meta_data['is_homepage'] = meta_data['homepage'].isnull()
meta_data['status'] = meta_data['status'].fillna('')

meta_data['original_language'] = meta_data['original_language'].fillna('')

meta_data['btc_name'] = meta_data['btc_name'].fillna('')
meta_data.drop_duplicates(inplace = True)

meta_data.drop(index = [2584 , 201 , 963 , 5769 , 5931 , 5175, 5587 , 845, 9661 ,11448 , 4145 , 4394 , 11254 , 10511 , 13335 , 13334 , 13329 , 16345 , 16348 , 16349 , 9658 , 9662 , 4391 , 4395 , 846 , 849 , 850 , 5927 , 5932 , 24363 , 33395 , 14101] , inplace = True)
def vector_values(df , columns , min_df_value):

    c_vector = CountVectorizer(min_df = min_df_value)

    df_1 = pd.DataFrame(index = df.index)

    for col in columns:

        print(col)

        df_1 = df_1.join(pd.DataFrame(c_vector.fit_transform(df[col]).toarray(),columns =c_vector.get_feature_names(),index= df.index).add_prefix(col+'_'))

    return df_1

meta_data_addon_1 = vector_values(meta_data , columns = ['status','original_language','genres', 'production_companies' ,'production_countries' , 'spoken_languages' , 'keywords' , 'cast' ,'crew'] ,min_df_value = 20)

meta_data_addon_2 = vector_values(meta_data , columns = ['btc_name'] , min_df_value = 2)
col = ['belongs_to_collection', 'genres' , 'homepage' , 'id' , 'imdb_id' , 'overview' ,'poster_path' , 'status' , 'original_language' , 

'production_companies', 'production_countries', 'spoken_languages', 'keywords',  'cast',  'crew', 'tagline','adult'  ]

meta_data.drop(columns = col , inplace = True)

col = [ 'video', 'is_homepage']

for c in col:

    meta_data[c] = meta_data[c].astype(bool)

    meta_data[c] = meta_data[c].astype(int)
def get_year(date):

    return str(date).split('-')[0]

meta_data['popularity'] = meta_data['popularity'].astype(float)

meta_data['budget'] = meta_data['budget'].astype(float)

meta_data['vote_average_group'] = pd.qcut(meta_data['vote_average'], q=10, precision=2,duplicates = 'drop')

meta_data['popularity_group'] = pd.qcut(meta_data['popularity'], q=10, precision=2,duplicates = 'drop')

meta_data['vote_average_group'] =pd.qcut(meta_data['vote_average'], q=10, precision=2,duplicates = 'drop')

meta_data['runtime_group'] = pd.qcut(meta_data['runtime'], q=10, precision=2,duplicates = 'drop')

meta_data['budget_group'] = pd.qcut(meta_data['budget'], q=10, precision=2,duplicates = 'drop')

meta_data['revenue_group'] = pd.qcut(meta_data['revenue'], q=10, precision=2,duplicates = 'drop')

meta_data['vote_count_group'] = pd.qcut(meta_data['vote_count'], q=10, precision=2,duplicates = 'drop')

meta_data['release_year'] = meta_data['release_date'].apply(get_year)

meta_data['release_year'] = meta_data['release_year'].fillna('')

meta_data['release_year'] = meta_data['release_year'].astype(float)

meta_data['release_year_group'] = pd.qcut(meta_data['release_year'], q=10, precision=2,duplicates = 'drop')

meta_data['title_new'] = meta_data.apply(lambda x: str(x['title'])+' ('+str(x['release_date'])+')' , axis =1)
meta_data_addon_3 = pd.get_dummies(meta_data[['vote_average_group' , 'popularity_group' , 'runtime_group' , 'budget_group' , 'revenue_group' , 'vote_count_group' , 'release_year_group']])

meta_data_train = pd.concat([meta_data_addon_1,meta_data_addon_2,meta_data_addon_3 , meta_data[['video' , 'is_homepage']]] , axis = 1)
meta_data_train.index = meta_data['title_new']
del meta_data_addon_1,meta_data_addon_2,meta_data_addon_3

gc.collect()
def get_similar_movies(movie_title , num_rec = 10):

    try:

        sample_1 = 1 - pairwise_distances([meta_data_train.loc[movie_title].values] , meta_data_train.values , metric = 'cosine')

        sample_1 = pd.DataFrame(sample_1.T , index = meta_data_train.index )

        return sample_1.sort_values(by = 0 , ascending  = False).head(num_rec).index

    except ValueError as e:

        print(e)

#         sample_1 = 1 - pairwise_distances(meta_data_train.loc[movie_title].values, meta_data_train.values , metric = 'cosine')

#         sample_1 = pd.DataFrame(sample_1.T , index = meta_data_train.index )

#         return sample_1.sort_values(by = 0 , ascending  = False).head(20).index.names
print(get_similar_movies('Undisputed III : Redemption (2010-05-22)'))

print(get_similar_movies('Finding Nemo (2003-05-30)'))

print(get_similar_movies('Mindhunters (2004-05-07)'))

print(get_similar_movies('Thor (2011-04-21)'))

print(get_similar_movies('Kong: Skull Island (2017-03-08)'))
def multi_rec(seen_movies , num_rec = 10):

    rec_movies = []

    for mov in seen_movies:

        rec_movies.append(get_similar_movies(mov , 5).values)

    return rec_movies

multi_rec(['Star Wars: The Clone Wars (2008-08-05)' , 'Marvel One-Shot: Item 47 (2012-09-13)'])
data = data.sample(frac = 1)

data_train_x = np.array(data[['user id' , 'movie id']].values)

data_train_y = np.array(data['rating'].values)

x_train, x_test, y_train, y_test = train_test_split(data_train_x, data_train_y, test_size = 0.2, random_state = 98)

n_factors = 50

n_users = data['user id'].max()

n_movies = data['movie id'].max()
user_input = Input(shape=(1,), name='User_Input')

user_embeddings = Embedding(input_dim = n_users+1, output_dim=n_factors, input_length=1,name='User_Embedding')(user_input)

user_vector = Flatten(name='User_Vector') (user_embeddings)



movie_input = Input(shape = (1,) , name = 'Movie_input')

movie_embeddings = Embedding(input_dim = n_movies+1 , output_dim = n_factors , input_length = 1 , name = 'Movie_Embedding')(movie_input)

movie_vector = Flatten(name = 'Movie_Vector')(movie_embeddings)



merged_vectors = concatenate([user_vector, movie_vector], name='Concatenation')

dense_layer_1 = Dense(100 , activation = 'relu')(merged_vectors)

dense_layer_3 = Dropout(.5)(dense_layer_1)

dense_layer_2 = Dense(1)(dense_layer_3)

model = Model([user_input, movie_input], dense_layer_2)
model.compile(loss='mean_squared_error', optimizer='adam' ,metrics = ['accuracy'] )

model.summary()
SVG(model_to_dot( model,  show_shapes=True, show_layer_names=True).create(prog='dot', format='svg'))
history = model.fit(x = [x_train[:,0] , x_train[:,1]] , y =y_train , batch_size = 128 , epochs = 30 , validation_data = ([x_test[:,0] , x_test[:,1]] , y_test))
loss , val_loss , accuracy , val_accuracy = history.history['loss'],history.history['val_loss'],history.history['accuracy'],history.history['val_accuracy']
plt.figure(figsize = (12,10))

plt.plot( loss, 'r--')

plt.plot(val_loss, 'b-')

plt.plot( accuracy, 'g--')

plt.plot(val_accuracy,'-')

plt.legend(['Training Loss', 'Validation Loss' , 'Training Accuracy' , 'Validation Accuracy'])

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.show()
score = model.evaluate([x_test[:,0], x_test[:,1]], y_test)

print(np.sqrt(score))