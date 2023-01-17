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
from keras.models import Sequential, Model, load_model

from keras.layers import Dense

from keras.layers import Input, concatenate, Embedding, Reshape

from keras.layers import Flatten, merge, Lambda, Dropout

from keras.layers.normalization import BatchNormalization

from keras.regularizers import l2, l1_l2

from keras.optimizers import SGD, Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder

import keras.backend as K
import warnings

warnings.filterwarnings('ignore')
df_ratings= pd.read_csv("/kaggle/input/the-movies-dataset/ratings.csv")

df_movies= pd.read_csv("/kaggle/input/the-movies-dataset/movies_metadata.csv")
df_ratings['movieId'].nunique()
# Average Movies per user

df_ratings.shape[0]/df_ratings['userId'].nunique()
movies_req_columns= ['adult', 'budget', 'genres', 'id', 'original_language',

                    'popularity', 'production_countries', 'revenue', 'runtime',

                    'spoken_languages', 'status', 'video', 'vote_average']



df_movies= df_movies[movies_req_columns]



df_movies['video']= np.where(df_movies['video']==True, 1, 0)

df_movies['adult']= np.where(df_movies['adult']==True, 1, 0)



import ast

def modify_prod_country(x):

    try:

        country= [i['iso_3166_1'] for i in ast.literal_eval(x)][0]

    except:

        return 'MS'

    return country



df_movies['production_countries']= df_movies['production_countries'].apply(modify_prod_country)



import ast

def modify_language(x):

    try:

        lang= [i['iso_639_1'] for i in ast.literal_eval(x)][0]

    except:

        return 'MS'

    return lang



df_movies['spoken_languages']= df_movies['spoken_languages'].apply(modify_language)
import ast

def modify_genre(x):

    genre= [i['name'] for i in ast.literal_eval(x)]

    return genre



df_movies['genre_modified']= df_movies['genres'].apply(modify_genre)



all_genre=[]

for i in df_movies['genre_modified']:

    for j in i:

        all_genre.append(j)

        

new_genre_cols= list(set(all_genre))



for col in new_genre_cols:

    df_movies[col]=0

    

for i in new_genre_cols:

    df_movies.loc[df_movies['genre_modified'].apply(lambda x: True if i in x else False), i]=1

    

df_movies.drop(['genres', 'genre_modified'], axis=1, inplace= True)

df_movies= df_movies.rename(columns={'id':'movieId'})
df_movies.head(1)
df_movies.dropna(inplace= True, axis=0)
df_ratings.drop(['timestamp'], axis=1, inplace= True)
df_ratings['movieId']=  df_ratings['movieId'].astype(str)
df_final= df_ratings.merge(df_movies, on=['movieId'])
df_final.shape
del df_ratings, df_movies
df_final.drop('movieId', axis=1, inplace=True)
char_cols= ['userId', 'original_language', 'production_countries', 'spoken_languages', 'status',

           'video']

target_cols= ['rating']

num_cols= [col for col in df_final.columns if col not in char_cols+target_cols]
X_train_wide= df_final[num_cols]

df_final.drop(num_cols, axis= 1, inplace= True)
X_train_deep= df_final[char_cols]

df_final.drop(char_cols, axis= 1, inplace= True)
y= df_final[target_cols]
del df_final
X_train_wide.shape
from sklearn.preprocessing import MinMaxScaler

scaler= MinMaxScaler()

X_train_wide= pd.DataFrame(scaler.fit_transform(X_train_wide), columns= X_train_wide.columns)
K.clear_session()

#Wide Network

w = Input(shape=(len(num_cols),), dtype="float32", name="num_inputs")

#wd = Dropout(0.2, seed=111)(w)

wd = Dense(128, activation="relu")(w)

wd = BatchNormalization()(wd)

wd = Dense(64, activation="relu")(wd)
for col in char_cols:

    le= LabelEncoder()

    le.fit(list(X_train_deep[col]))

    X_train_deep[col]= le.transform(list(X_train_deep[col]))
embed_tensors = []



for input_col in char_cols:

    vocab_size= len(set(list(X_train_deep[input_col])))

    input_cat= Input(shape=(1,), name=input_col)

    embed_chain = Embedding(vocab_size, 20, input_length= 1, embeddings_regularizer= l2(0.001))(input_cat)

    embed_tensors.append((input_cat, embed_chain))
inp_layers = [et[0] for et in embed_tensors]

inp_embed = [et[1] for et in embed_tensors]
d = concatenate(inp_embed)

dp = Flatten()(d)



dp = BatchNormalization()(dp)

dp = Dense(1024, activation="relu")(dp)

dp = Dropout(0.2, seed=111)(dp)

dp = Dense(1024, activation="relu")(dp)

dp = Dense(256, activation="relu")(dp)

dp = Dropout(0.2, seed=111)(dp)

dp = Dense(128, activation="relu")(dp)



#Adding dropout



dp = Dense(16, activation="relu", name="deep")(dp)
#Concatenating 

wd_inp = concatenate([wd, dp])



#wd_inp = BatchNormalization()(wd_inp)

wd_inp = Dense(128, activation="relu")(wd_inp)

wd_inp = Dropout(0.2, seed=111)(wd_inp)



wd_inp = Dense(50, activation="relu")(wd_inp)

wd_inp = Dropout(0.2, seed=111)(wd_inp)



wd_inp = Dense(10, activation="relu")(wd_inp)



wd_inp= Dense(1, activation= "relu")(wd_inp)
wide_deep = Model(inputs = [w]+inp_layers, outputs = wd_inp)
wide_deep.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mse'])
complete_training= [X_train_wide]+[X_train_deep[[col]] for col in X_train_deep.columns]
y.to_csv("Target.csv", index= False)

X_train_wide.to_csv("X_train_wide.csv", index= False)

X_train_deep.to_csv("X_train_deep.csv", index= False)
# history= wide_deep.fit(complete_training, y, epochs = 100, batch_size=1024, verbose=1, validation_split=0.20)
# import matplotlib.pyplot as plt



# plt.plot(history.history['mse'])

# plt.plot(history.history['val_mse'])

# plt.title('model mseuracy')

# plt.ylabel('mseuracy')

# plt.xlabel('epoch')

# plt.legend(['train', 'validation'], loc='upper left')

# plt.show()

# # summarize history for loss

# plt.plot(history.history['loss'])

# plt.plot(history.history['val_loss'])

# plt.title('model loss')

# plt.ylabel('loss')

# plt.xlabel('epoch')

# plt.legend(['train', 'validation'], loc='upper left')

# plt.show()