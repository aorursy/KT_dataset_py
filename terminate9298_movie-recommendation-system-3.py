import numpy as np 

import pandas as pd 

import os

import ast

import random

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import pairwise_distances

from tqdm import tqdm

import time



# import tensorflow.contrib.eager as tfe

from keras.models import Sequential

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.models import Model

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM , CuDNNGRU , GRU

from keras.layers import  Input, dot, concatenate

from keras.optimizers import Adam

from keras.utils import to_categorical

from keras.utils.vis_utils import plot_model

from keras.models import load_model

from sklearn.decomposition import PCA



import gc

import matplotlib.pyplot as plt

from datetime import datetime

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

pd.set_option('display.max_rows',50)

pd.set_option('display.max_columns', 50)
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

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



def vector_values(df , columns , min_df_value , max_df_value = 1.0):

    c_vector = CountVectorizer(min_df = min_df_value  , max_df = max_df_value)

    df_1 = pd.DataFrame(index = df.index)

    for col in columns:

        print(col)

        df_1 = df_1.join(pd.DataFrame(c_vector.fit_transform(df[col]).toarray(),columns =c_vector.get_feature_names(),index= df.index).add_prefix(col+'_'))

    return df_1



def get_year(date):

    return str(date).split('-')[0]



%%time

ratings = pd.read_csv('/kaggle/input/the-movies-dataset/ratings.csv')

ratings.rating = ratings.rating*2

ratings.rating = ratings.rating.astype(int)

ratings = reduce_mem_usage(ratings)
ratings = ratings.sample(frac = 1.0)
ratings.sample(15).T
print('Number of Unique Movies are {}'.format(len(ratings.movieId.unique())))

print('Number of Unique Users are {}'.format(len(ratings.userId.unique())))

print('Number of Max Movie Id is {}'.format(ratings.movieId.max()))

print('Number of Max User Id is {}'.format(ratings.userId.max()))
# Remove the movies below Threshold Values

user_value_count = ratings.userId.value_counts()

movie_value_count = ratings.movieId.value_counts()
print('Average of the users counts is {}'.format(user_value_count.mean()))

print('Average of the movies counts is {}'.format(movie_value_count.mean()))

print('Max of the User counts is {}'.format(user_value_count.max()))

print('Max of the Movie counts is {}'.format(movie_value_count.max()))
print('Description of User Value Count \n',user_value_count.describe())

print('\nDescription of Movie Value Count \n',movie_value_count.describe())
meta_data = pd.read_csv('/kaggle/input/the-movies-dataset/movies_metadata.csv')



meta_data = meta_data[meta_data.id!='1997-08-20']

meta_data = meta_data[meta_data.id!='2012-09-29']

meta_data = meta_data[meta_data.id!='2014-01-01']

meta_data = meta_data.astype({'id':'int64'})

meta_data = meta_data[['id' , 'release_date' , 'title']]

meta_data.columns = ['movieId' , 'release_date' , 'title']
print('Before Size of Ratings is ' , ratings.shape)

ratings = ratings.merge(meta_data , on = 'movieId')

print('After Size of Ratings is ',ratings.shape)
movie_details = ratings.groupby('movieId')['movieId'].agg('count')

movie_details =  pd.DataFrame(movie_details)

movie_details.columns = ['movie_counts']

movie_details.reset_index(inplace = True)

ratings = ratings.merge(movie_details, on = 'movieId')

del movie_details
user_details = ratings.groupby('userId')['userId'].agg('count')

user_details = pd.DataFrame(user_details)

user_details.columns = ['user_counts']

user_details.reset_index(inplace = True)

ratings = ratings.merge(user_details , on = 'userId')

del user_details
ratings.groupby(pd.qcut(ratings.user_counts, 5))['user_counts'].count()
ratings.groupby(pd.qcut(ratings.movie_counts, 5))['movie_counts'].count()
print('Size of Ratings is {} by {}'.format(ratings.shape[0] , ratings.shape[1]))
user_thresold = 10
# ratings = ratings[ratings['movie_counts'] > movie_thresold]

ratings = ratings[ratings['user_counts'] > user_thresold]

print('Size of Ratings is {} by {}'.format(ratings.shape[0] , ratings.shape[1]))
ratings = reduce_mem_usage(ratings)
movies_dataframe = ratings.movieId.unique()

random.shuffle(movies_dataframe)

movies_dataframe = pd.DataFrame(movies_dataframe)

movies_dataframe.reset_index(inplace = True)

movies_dataframe.columns = ['movie_index','movieId']

# movies_dataframe['movie_index'] = movies_dataframe['movie_index'] + 1

ratings = ratings.merge(movies_dataframe , on='movieId')
users_dataframe = ratings.userId.unique()

random.shuffle(users_dataframe)

users_dataframe = pd.DataFrame(users_dataframe)

users_dataframe.reset_index(inplace = True)

users_dataframe.columns = ['user_index','userId']

# users_dataframe['user_index'] = users_dataframe['user_index'] + 1

ratings = ratings.merge(users_dataframe , on='userId')
ratings.sample(5)
n_users = ratings['user_index'].max()

n_movies = ratings['movie_index'].max()

n_factors = 100
print(n_users)

print(n_movies)
ratings.sort_values(by=['user_index' , 'timestamp'] , inplace = True)
gby = ratings.drop(columns = ['timestamp','title','movieId','userId','release_date','rating','movie_counts','user_counts']).groupby('user_index')[['movie_index']]



grouped_data = []

def fi(x):

    grouped_data.append(np.array(x.values.reshape(-1)))

gby.apply(fi)
del gby ,users_dataframe , movies_dataframe , meta_data , user_value_count , movie_value_count

gc.collect()
def prepare_data(step = 1):

    x_data = []

    y_data = []

    if step == 1:

        for g_data in grouped_data:

            for i in range(len(g_data) - step):

                d = i + step

                x_data.append(g_data[i,])

                y_data.append(g_data[d,])

    else:

        for g_data in grouped_data:

            for i in range(len(g_data) - step):

                d = i + step

                x_data.append(g_data[i:d,])

                y_data.append(g_data[d,])

            

    return x_data , y_data

x_data , y_data = prepare_data(step =1)



y_data = np.array(y_data)

x_data = np.array(x_data)
def data_shuffle(x_data , y_data):

    randomize = np.arange(len(x_data))

    np.random.shuffle(randomize)

    x_data = x_data[randomize]

    y_data = y_data[randomize]

    return x_data , y_data
def session_data(data_x , data_y ,  max_size):

    data_x = to_categorical(data_x , num_classes = max_size , dtype = 'bool')

    data_y = to_categorical(data_y , num_classes = max_size , dtype = 'bool')

    data_x = np.array(data_x.reshape([data_x.shape[0] , 1 , data_x.shape[1]]))

    data_y = np.array(data_y)

    return  data_x , data_y
def train_model(model_to_fit , x_data , y_data , max_size , batch_size = 64 , sample_size = 640000 ,epochs = 3 , epochs_per_step = 3):

#     model_to_fit = model

    history = []

    i = 0

    present_time = time.time()

    for epoch in range(0,epochs):

        print('Starting to Train \nEpochs value - {} '.format(epoch))

        x_data , y_data = data_shuffle(x_data , y_data)

        for i_size in range( 0 , len(x_data) - sample_size , sample_size):

            x_train , y_train = session_data(x_data[i_size : i_size + sample_size] , y_data[i_size : i_size + sample_size] , max_size)

            model_to_fit.fit(x = x_train , y = y_train , verbose = 1 ,epochs = epochs_per_step , batch_size = batch_size )

            i = i+1

            if i > 5:

                gc.collect()

                i=0

                print('RAM Resetted..')

                if (time.time()-present_time)>25000:

                    model_to_fit.save('model_e_final.h5')

                    return model_to_fit

        print('Coming to predict..')

        model_to_fit.save('model_e_{}.h5'.format(epoch))

#         x_val, y_val = session_data(x_val[-(len(x_val)//batch_size)*batch_size : ], y_val[-(len(x_val)//batch_size)*batch_size : ], max_size)

#         accuracy = model_to_fit.evaluate(x_val, y_val , batch_size = batch_size)

#         print('Accuracy on Validation set is ' , accuracy)

#         model_to_fit.save('model_e_{}_a_{:.2f}.h5'.format(epoch , accuracy))

#     return model_to_fit
movie_input = Input(batch_shape=(64 , 1 , n_movies+1), name='Movie_Input')

movie_model = LSTM(100, stateful=True)(movie_input)

movie_model = Dropout(0.25)(movie_model)

movie_model = Dense(n_movies+1)(movie_model)

model = Model(movie_input, movie_model)



opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon = 1e-24 , decay=0.0, amsgrad=False)

model.compile(loss='categorical_crossentropy', optimizer=opt)
model.summary()
x_data_sample , y_data_sample = x_data[:107219*2] , y_data[:107219*2]
train_model(model , x_data, y_data , max_size = x_data.max()+1  ,sample_size = 32000)
sample_1 = np.zeros([1,1,7556])

sample_1[0][0][86] = 1.0
movie_input_new = Input(batch_shape=(1 , 1 , n_movies+1), name='Movie_Input')

movie_model_new = LSTM(100, stateful=True)(movie_input_new)

movie_model_new = Dropout(0.25)(movie_model_new)

movie_model_new = Dense(n_movies+1)(movie_model_new)

model_new = Model(movie_input_new, movie_model_new)



opt_new = Adam(lr=0.001, beta_1=0.9, beta_2=0.999,  decay=0.0, amsgrad=False)

model_new.compile(loss='categorical_crossentropy', optimizer=opt_new)
old_weights = model.get_weights()

model_new.set_weights(old_weights)
predict_out = model_new.predict(sample_1 , batch_size = 1)

predict_out = np.array(predict_out)

print(predict_out)
predict_out.argmax()
ratings[ratings.movie_index == 5788]
ratings.sample(10)