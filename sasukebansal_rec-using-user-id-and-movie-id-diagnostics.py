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



# Importing the keras modules for Deep and Wide Neural Network

from keras.models import Sequential, Model, load_model

from keras.layers import Dense

from keras.layers import Input, concatenate, Embedding, Reshape

from keras.layers import Flatten, merge, Lambda, Dropout

from keras.layers.normalization import BatchNormalization

from keras.regularizers import l2, l1_l2

from keras.optimizers import SGD, Adam, RMSprop

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder

import keras.backend as K

import keras



pd.options.display.max_columns = None

pd.options.display.max_rows = 10



import warnings

warnings.filterwarnings('ignore')



from sklearn.metrics import mean_absolute_error
df_ratings= pd.read_csv("/kaggle/input/the-movies-dataset/ratings_small.csv")[['userId', 'movieId', 'rating']]

df_movies= pd.read_csv("/kaggle/input/the-movies-dataset/movies_metadata.csv").rename(columns={'id':'movieId'})



df_movies.drop_duplicates(subset= 'movieId', inplace= True)

# Changing the data type of movieId and removing the incorrect movieIds

df_movies= df_movies[~df_movies.movieId.str.contains('-')]

df_movies['movieId']= df_movies['movieId'].astype(int)



# Merging movies data with ratings data

print('Shape before merging', df_ratings.shape[0])

df_final= df_ratings.merge(df_movies, on=['movieId'])

print('Shape after merging', df_final.shape[0])



del df_ratings, df_movies
# Users should have rated atleast 10 movies

frequent_users= df_final.userId.value_counts()[df_final.userId.value_counts()>10].index.values



df_final= df_final[df_final.userId.isin(frequent_users)]

print('Shape after selecting high users', df_final.shape[0])
df_title= df_final[['movieId', 'title', 'genres']].drop_duplicates()
req_columns= ['userId', 'rating', 'adult', 'budget', 'genres', 'movieId', 'original_language','popularity', 

                     'production_countries', 'revenue', 'runtime', 'spoken_languages', 'status', 'video', 'vote_average']

df_final= df_final[req_columns]



# Functions to get country from json structure

import ast

def modify_prod_country(x):

    try:

        country= [i['iso_3166_1'] for i in ast.literal_eval(x)][0]

    except:

        return 'MS' # Missing

    return country



def modify_language(x):

    try:

        lang= [i['iso_639_1'] for i in ast.literal_eval(x)][0]

    except:

        return 'MS' # Missing

    return lang



df_final['production_countries']= df_final['production_countries'].apply(modify_prod_country)

df_final['spoken_languages']= df_final['spoken_languages'].apply(modify_language)



df_final['video']= np.where(df_final['video']==True, 1, 0)

df_final['adult']= np.where(df_final['adult']==True, 1, 0)





# One hot encoding of Genre module

def modify_genre(x):

    genre= [i['name'] for i in ast.literal_eval(x)]

    return genre



df_final['genre_modified']= df_final['genres'].apply(modify_genre)

df_title['genres']= df_title['genres'].apply(modify_genre)



all_genre=[]

for i in df_final['genre_modified']:

    for j in i:

        all_genre.append(j)

        

new_genre_cols= list(set(all_genre))



for col in new_genre_cols:

    df_final[col]=0

    

for i in new_genre_cols:

    df_final.loc[df_final['genre_modified'].apply(lambda x: True if i in x else False), i]=1

    

df_final.drop(['genres', 'genre_modified'], axis=1, inplace= True)
df_final.head(2)
# Removing null values from dataset

df_final.dropna(inplace= True, axis=0)

df_final.head(2)
# Scaling and label encoding

cat_cols= ['original_language', 'production_countries', 'spoken_languages', 'status']

num_cols= ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'rating']



from sklearn.preprocessing import MinMaxScaler

scaler= MinMaxScaler()

df_final[num_cols]= scaler.fit_transform(df_final[num_cols])



for col in cat_cols:

    le= LabelEncoder()

    le.fit(list(df_final[col]))

    df_final[col]= le.transform(list(df_final[col]))

    

df_final.head(1)
df_title= df_title[df_title.movieId.isin(df_final.movieId)]
# Label encoding of movies and users

le_movies= LabelEncoder()

le_movies= le_movies.fit(df_final.movieId)

df_final['movieId']= le_movies.transform(df_final['movieId'])

df_title['movieId']= le_movies.transform(df_title['movieId'])



le_users= LabelEncoder()

le_users= le_users.fit(df_final.userId)

df_final['userId']= le_users.transform(df_final['userId'])
df_final_predictions= df_final.copy()

df_final_predictions.head(2)
print(df_final_predictions.shape)

print(df_final_predictions.drop_duplicates(subset= ['userId', 'movieId']).shape)
# Creating validation data (one movie for each user)

val_data= df_final_predictions.drop_duplicates(subset= 'userId')

val_data['dummy']= 1



# Removing the validation data from training data

df_final_predictions= df_final_predictions.merge(val_data[['userId', 'movieId', 'dummy']], how= 'left')

df_final_predictions= df_final_predictions[df_final_predictions.dummy.isnull()]

df_final_predictions.drop('dummy', axis=1, inplace= True)



df_final_predictions.head(2)
cat_cols= ['userId','movieId', 'original_language', 'production_countries', 'spoken_languages', 'status', 'video']

target_cols= ['rating']

num_cols= [col for col in df_final.columns if col not in cat_cols + target_cols]
# Numeric Columns to be used for Wide Network

X_train_wide= df_final_predictions[num_cols]

X_val_wide= val_data[num_cols]
# Categorical columns to be used for Deep Network

X_train_deep= df_final_predictions[cat_cols]

X_val_deep= val_data[cat_cols]
# Rating as target variable

y= df_final_predictions[target_cols]

y_val= val_data[target_cols]
K.clear_session()



#Wide Network

w = Input(shape=(len(num_cols),), dtype="float32", name="num_inputs")

wd = Dense(64, activation="relu")(w)

wd = BatchNormalization()(wd)

wd = Dense(32, activation="relu")(wd)
# Creating a input layer for each categorical variable along with its embedding layer

embed_tensors = []



for input_col in cat_cols:

    vocab_size= X_train_deep[input_col].nunique()

    input_cat= Input(shape=(1,), name=input_col)

    embed_chain = Embedding(vocab_size, 100, input_length= 1)(input_cat)

    embed_tensors.append((input_cat, embed_chain))
inp_layers = [et[0] for et in embed_tensors]

inp_embed = [et[1] for et in embed_tensors]
leaky_relu= keras.layers.LeakyReLU(alpha=0.3)

d = concatenate(inp_embed)

dp = Flatten()(d)



dp = BatchNormalization()(dp)

dp = Dense(1024, activation=leaky_relu)(dp)

dp = BatchNormalization()(dp)



dp = Dense(512, activation=leaky_relu)(dp)

dp = BatchNormalization()(dp)

dp = Dropout(0.3, seed=111)(dp)



dp = Dense(512, activation=leaky_relu)(dp)

dp = BatchNormalization()(dp)



dp = Dense(256, activation=leaky_relu)(dp)

dp = Dropout(0.3, seed=111)(dp)



dp = Dense(128, activation=leaky_relu)(dp)

dp = Dense(16, activation=leaky_relu, name="deep")(dp)
#Concatenating 

wd_inp = concatenate([wd, dp])



#wd_inp = BatchNormalization()(wd_inp)

wd_inp = Dense(128, activation=leaky_relu)(wd_inp)

wd_inp = Dropout(0.3, seed=111)(wd_inp)



wd_inp = Dense(50, activation=leaky_relu)(wd_inp)

wd_inp = Dropout(0.3, seed=111)(wd_inp)



wd_inp = Dense(10, activation=leaky_relu)(wd_inp)

wd_inp= Dense(1, activation= leaky_relu)(wd_inp)
wide_deep = Model(inputs = [w]+inp_layers, outputs = wd_inp)
adam= Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

rms= RMSprop(learning_rate=0.0001, rho=0.9)

wide_deep.compile(loss='mean_squared_error', optimizer=rms, metrics=['mse'])
complete_training= [X_train_wide]+[X_train_deep[[col]] for col in X_train_deep.columns]

complete_val= [X_val_wide]+[X_val_deep[[col]] for col in X_val_deep.columns]
history= wide_deep.fit(complete_training, y, epochs = 20, batch_size=512, verbose=1, validation_split=0.15,

                      shuffle= True)
val_data['predicted_rating']= wide_deep.predict(complete_val)



num_cols= ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'rating']

val_data[num_cols]= scaler.inverse_transform(val_data[num_cols])

                   

num_cols= ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'predicted_rating']

val_data[num_cols]= scaler.inverse_transform(val_data[num_cols])
val_data[['rating', 'predicted_rating']].reset_index().drop('index', axis=1)
mean_absolute_error(val_data.rating, val_data.predicted_rating)
# Making recommendations for a particular user

user_id= 85

df_user= df_final_predictions[df_final_predictions['userId']!=user_id]

df_user['userId']= user_id

df_user.drop('rating', axis=1, inplace=True)

df_user= df_user.drop_duplicates()
cat_cols= ['userId','movieId', 'original_language', 'production_countries', 'spoken_languages', 'status', 'video']

target_cols= ['rating']

num_cols= [col for col in df_final.columns if col not in cat_cols + target_cols]



X_train_wide_pred= df_user[num_cols]

X_train_deep_pred= df_user[cat_cols]



complete_predictions= [X_train_wide_pred]+[X_train_deep_pred[[col]] for col in X_train_deep_pred.columns]

df_user['predicted_rating']= wide_deep.predict(complete_predictions)
df_user= df_user.sort_values(by='predicted_rating', ascending= False).head(10)[['movieId', 'predicted_rating']]
df_user= df_user.reset_index()

df_user.drop('index', inplace= True, axis= 1)
df_user= df_user.merge(df_title)

df_user
from keras.models import Model

layer_name = 'embedding_2'

intermediate_layer_model = Model(inputs=wide_deep.input,

                                 outputs=wide_deep.get_layer(layer_name).output)

intermediate_output = intermediate_layer_model.predict(complete_training)

intermediate_output= intermediate_output.reshape(-1, intermediate_output.shape[2])

embeddings= pd.DataFrame(intermediate_output)
embeddings['movieId']= complete_training[2]
print(embeddings.shape)

embeddings= embeddings.drop_duplicates()

print(embeddings.shape)

df_embeddings= embeddings.merge(df_title)
df_movie_titles= df_embeddings[['title', 'genres']]

df_embeddings.drop(['movieId', 'title', 'genres'], axis=1, inplace= True)

df_embeddings.to_csv("Embeddings Matrix.csv", index= False)

df_movie_titles.to_csv("Movies Matrix.csv", index= False)