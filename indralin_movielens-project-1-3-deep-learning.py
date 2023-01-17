import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os

import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = [18, 8]
reviews = pd.read_csv('/kaggle/input/movielens-1m/ml-1m/ratings.dat', names=['userId', 'movieId', 'rating', 'time'], delimiter='::', engine='python')
users = pd.read_csv('/kaggle/input/movielens-1m/ml-1m/users.dat', names=['userId','gender','age','occupation','zip'], delimiter='::', engine='python')
movies = pd.read_csv('/kaggle/input/movielens-1m/ml-1m/movies.dat', names=['movieId', 'Movie_names', 'Genres'], delimiter='::', engine='python')

print('Reviews shape:', reviews.shape)
print('Users shape:', users.shape)
print('Movies shape:', movies.shape)
reviews.drop(['time'], axis=1, inplace=True)
users.drop(['zip'], axis=1, inplace=True)
movies['release_year'] = movies['Movie_names'].str.extract(r'(?:\((\d{4})\))?\s*$', expand=False)
final_df = reviews.merge(movies, on='movieId', how='left').merge(users, on='userId', how='left')

print('Final_df shape:', final_df.shape)
final_df.head()
userid_nunique = final_df['userId'].nunique()
movieid_nunique = final_df['movieId'].nunique()

print('User_id total unique:', userid_nunique)
print('Movieid total unique:', movieid_nunique)
from sklearn.preprocessing import LabelEncoder

user_enc = LabelEncoder()
final_df['userId'] = user_enc.fit_transform(final_df['userId'])

movie_enc = LabelEncoder()
final_df['movieId'] = movie_enc.fit_transform(final_df['movieId'])
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Reshape, Dot, Flatten, concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.utils import model_to_dot
from IPython.display import SVG
# Notes: Reshape(n_dim, ) is same with Flatten, we can use both (choose one)

def RecommenderV1(n_users, n_movies, n_dim):
    
    # User
    user = Input(shape=(1,))
    U = Embedding(n_users, n_dim)(user)
    U = Flatten()(U)
    
    # Movie
    movie = Input(shape=(1,))
    M = Embedding(n_movies, n_dim)(movie)
    M = Flatten()(M)
    
    # Dot U and M
    x = Dot(axes=1)([U, M])
    
    model = Model(inputs=[user, movie], outputs=x)
    
    model.compile(optimizer=Adam(0.0001),
                  loss='mean_squared_error')
    
    return model
model1 = RecommenderV1(userid_nunique, movieid_nunique, 100)
SVG(model_to_dot(model1,  show_shapes=True, show_layer_names=True).create(prog='dot', format='svg'))
model1.summary()
from sklearn.model_selection import train_test_split

X = final_df.drop(['rating'], axis=1)
y = final_df['rating']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=2020)

X_train.shape, X_val.shape, y_train.shape, y_val.shape
checkpoint1 = ModelCheckpoint('model1.h5', monitor='val_loss', verbose=0, save_best_only=True)
history1 = model1.fit(x=[X_train['userId'], X_train['movieId']], y=y_train, batch_size=64, epochs=10, verbose=1, validation_data=([X_val['userId'], X_val['movieId']], y_val), callbacks=[checkpoint1])
# Get training and test loss histories
training_loss1 = history1.history['loss']
test_loss1 = history1.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss1) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss1, 'r--')
plt.plot(epoch_count, test_loss1, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
## Notes: Reshape(n_dim, ) is same with Flatten, we can use both (choose one)

def RecommenderV2(n_users, n_movies, n_dim):
    
    # User
    user = Input(shape=(1,))
    U = Embedding(n_users, n_dim)(user)
    U = Flatten()(U)
    
    # Movie
    movie = Input(shape=(1,))
    M = Embedding(n_movies, n_dim)(movie)
    M = Flatten()(M)
    
    # Ganti disini
    merged_vector = concatenate([U, M])
    dense_1 = Dense(128, activation='relu')(merged_vector)
    dropout = Dropout(0.5)(dense_1)
    final = Dense(1)(dropout)
    
    model = Model(inputs=[user, movie], outputs=final)
    
    model.compile(optimizer=Adam(0.001),
                  loss='mean_squared_error')
    
    return model
model2 = RecommenderV2(userid_nunique, movieid_nunique, 100)
SVG(model_to_dot(model2,  show_shapes=True, show_layer_names=True).create(prog='dot', format='svg'))
model2.summary()
checkpoint2 = ModelCheckpoint('model2.h5', monitor='val_loss', verbose=0, save_best_only=True)
history2 = model2.fit(x=[X_train['userId'], X_train['movieId']], y=y_train, batch_size=64, epochs=20,
                      verbose=1, validation_data=([X_val['userId'], X_val['movieId']], y_val), callbacks=[checkpoint2])
# Get training and test loss histories
training_loss2 = history2.history['loss']
test_loss2 = history2.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss2) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss2, 'r--')
plt.plot(epoch_count, test_loss2, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
# Notes: Reshape(n_dim, ) is same with Flatten, we can use both (choose one)

def RecommenderV3(n_users, n_movies, n_dim):
    
    # User
    user = Input(shape=(1,))
    U = Embedding(n_users, n_dim)(user)
    U = Flatten()(U)
    U = Dense(64, activation='relu')(U)
    
    # Movie
    movie = Input(shape=(1,))
    M = Embedding(n_movies, n_dim)(movie)
    M = Flatten()(M)
    M = Dense(64, activation='relu')(M)
    
    # Dot U and M
    x = concatenate([U, M])
    x = Dropout(0.25)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    final = Dense(1)(x)
    
    model = Model(inputs=[user, movie], outputs=final)
    
    model.compile(optimizer=Adam(0.0001),
                  loss='mean_squared_error')
    
    return model
model3 = RecommenderV3(userid_nunique, movieid_nunique, 100)
SVG(model_to_dot(model3,  show_shapes=True, show_layer_names=True).create(prog='dot', format='svg'))
model3.summary()
checkpoint3 = ModelCheckpoint('model3.h5', monitor='val_loss', verbose=0, save_best_only=True)
history3 = model3.fit(x=[X_train['userId'], X_train['movieId']], y=y_train, batch_size=64, epochs=20,
                      verbose=1, validation_data=([X_val['userId'], X_val['movieId']], y_val), callbacks=[checkpoint3])
# Get training and test loss histories
training_loss3 = history3.history['loss']
test_loss3 = history3.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss3) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss3, 'r--')
plt.plot(epoch_count, test_loss3, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
from tensorflow.keras.models import load_model

model1 = load_model('model1.h5')
model2 = load_model('model2.h5')
model3 = load_model('model3.h5')
def make_pred(user_id, movie_id, model):
    return model.predict([np.array([user_id]), np.array([movie_id])])[0][0]
def get_topN_rec(user_id, model):
    
    user_ratings = final_df[final_df['userId'] == user_id][['userId', 'movieId', 'rating']]
    recommendation = final_df[~final_df['movieId'].isin(user_ratings['movieId'])][['movieId']].drop_duplicates()
    recommendation['rating_predict'] = recommendation.apply(lambda x: make_pred(user_id, x['movieId'], model), axis=1)
    
    final_rec = recommendation.sort_values(by='rating_predict', ascending=False).merge(movies[['movieId', 'Movie_names', 'release_year']],
                                                                       on='movieId',
                                                                       how='inner').head(10)
    
    return final_rec.sort_values('release_year', ascending=False).drop(['movieId', 'release_year'], axis=1)  # sort by recent year
get_topN_rec(23, model1)
get_topN_rec(23, model3)