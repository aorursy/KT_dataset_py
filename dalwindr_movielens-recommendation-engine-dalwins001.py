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
import pandas as pd
import numpy as np
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt
movielens_dir="/kaggle/input/ml-latest-small"
readme= movielens_dir +"/README.txt"
!cat $readme
pd.read_csv(movielens_dir + "/movies.csv").head()
pd.read_csv(movielens_dir + "/tags.csv").head()
pd.read_csv(movielens_dir + "/links.csv").head()
# # Download the actual data from http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
# # Use the ratings.csv file
# movielens_data_file_url = (
#     "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
# )
# movielens_zipped_file = keras.utils.get_file(
#     "ml-latest-small.zip", movielens_data_file_url, extract=False
# )
# keras_datasets_path = Path(movielens_zipped_file).parents[0]
# movielens_dir = keras_datasets_path / "ml-latest-small"

# # Only extract the data the first time the script is run.
# if not movielens_dir.exists():
#     with ZipFile(movielens_zipped_file, "r") as zip:
#         # Extract files
#         print("Extracting all the files now...")
#         zip.extractall(path=keras_datasets_path)
#         print("Done!")

ratings_file = movielens_dir + "/ratings.csv"
df = pd.read_csv(ratings_file)
global obs_cnt
obs_cnt = 0
def observation(comment):
    global obs_cnt
    obs_cnt= obs_cnt+1
    print("\nObservation-",obs_cnt,"->",comment,"\n")
print(df.shape)
df.head()
# Null checking
observation("""No null value found""")
df.isna().sum()


user_ids = df["userId"].unique().tolist()
num_users=len(user_ids)
print("no of users:-",num_users)
observation("no of unique user id's is 610")

movie_ids = df["movieId"].unique().tolist()
num_movies = len(movie_ids)
print("no of movie Id:-",num_movies)
observation("no of unique movies id is 9724")
min_rating = np.min(df["rating"])
max_rating = np.max(df["rating"])
print("min Rating:", min_rating, "\nmax Rating:",max_rating)
observation("no of min and max rating value given to movie are : 0.5 and 5.0")
observation("  userId and MovieID columns are of integer type \n\t\t Need to change the datatype into categorical.")
df.dtypes
# covert the datatype of the 
df.userId = df.userId.astype('category').cat.codes.values
df.movieId = df.movieId.astype('category').cat.codes.values
observation("  userId and MovieID columns datatype changed into categorical.")
df.dtypes
df['userId'].value_counts(ascending=True).head(100).values
df['userId'].value_counts(ascending=False).head(100).values
df['movieId'].value_counts(ascending=True).head(100).values
df['movieId'].value_counts(ascending=False).head(100).values
util_df=pd.pivot_table(data=df,values='rating',index='userId',columns='movieId').fillna(0)
util_df
from sklearn.metrics.pairwise import pairwise_distances 
user_similarity = pairwise_distances(util_df, metric='cosine')
item_similarity = pairwise_distances(util_df.T, metric='cosine')
print(user_similarity)
#user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
#movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
#user2user_encoded[1]
df['userId'] = df['userId'].apply(lambda x: userencoded2user[x])
df['movieId'] = df['movieId'].apply(lambda x: movie_encoded2movie[x])
split = np.random.rand(len(df)) < 0.8
train = df[split]
valid = df[~split]
print(train.shape , valid.shape)
#preprocess.
from keras.preprocessing.image import ImageDataGenerator

#dl libraraies
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense , merge
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical
from keras.utils.vis_utils import model_to_dot
from keras.callbacks import ReduceLROnPlateau


from keras.layers.merge import Dot, multiply, concatenate
from keras.models import Model


# specifically for deeplearning.
from keras.layers import Dropout, Flatten,Activation,Input,Embedding
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
n_latent_factors=64  # hyperparamter to deal with. 
num_users, num_movies # Already calculated above

user_ids[:10], "\n",movie_ids[:10]
user_input=Input(shape=(1,),name='user_input',dtype='int64')
user_embedding=Embedding(num_users,n_latent_factors,name='user_embedding')(user_input)
#user_embedding.shape
user_vec =Flatten(name='FlattenUsers')(user_embedding)
user_vec.shape
movie_input=Input(shape=(1,),name='movie_input',dtype='int64')
movie_embedding=Embedding(num_movies,n_latent_factors,name='movie_embedding')(movie_input)
movie_vec=Flatten(name='FlattenMovies')(movie_embedding)
movie_vec
sim=dot([user_vec,movie_vec],name='Simalarity-Dot-Product',axes=1)
model =keras.models.Model([user_input, movie_input],sim)
model.summary()
# # A summary of the model is shown below-->
model.compile(optimizer=Adam(lr=1e-4),loss='mse')
#[train.userId,train.movieId]
#train.rating
#[valid.userId,valid.movieId]
#valid.rating
History = model.fit([train.userId,train.movieId],train.rating,
                    batch_size=64,epochs =50, 
                    validation_data = ([valid.userId,valid.movieId],valid.rating),verbose = 1
                   )
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5
import matplotlib.pyplot as plt
plt.plot(History.history['loss'] , 'g')
plt.plot(History.history['val_loss'] , 'b')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()
# Creating the Embeddings
n_latent_factors=50
n_movies=len(df['movieId'].unique())
n_users=len(df['userId'].unique())

# model configuration part 1
user_input=Input(shape=(1,),name='user_input',dtype='int64')
user_embedding=Embedding(n_users,n_latent_factors,name='user_embedding')(user_input)
user_vec=Flatten(name='FlattenUsers')(user_embedding)
user_vec=Dropout(0.40)(user_vec)

# model configuration part 2
movie_input=Input(shape=(1,),name='movie_input',dtype='int64')
movie_embedding=Embedding(n_movies,n_latent_factors,name='movie_embedding')(movie_input)
movie_vec=Flatten(name='FlattenMovies')(movie_embedding)
movie_vec=Dropout(0.40)(movie_vec)

# model configuration part 3
sim=dot([user_vec,movie_vec],name='Simalarity-Dot-Product',axes=1)
nn_inp=Dense(96,activation='relu')(sim)
nn_inp=Dropout(0.4)(nn_inp)
# nn_inp=BatchNormalization()(nn_inp)
nn_inp=Dense(1,activation='relu')(nn_inp)

# Ensemle Part 1, Part 2, Part 3
nn_model =keras.models.Model([user_input, movie_input],nn_inp)
nn_model.summary()

nn_model.compile(optimizer=Adam(lr=1e-3),loss='mse')
History = nn_model.fit([train.userId,train.movieId],train.rating, batch_size=128,
                              epochs =40, validation_data = ([valid.userId,valid.movieId],valid.rating),
                              verbose = 1)
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5
import matplotlib.pyplot as plt
plt.plot(History.history['loss'] , 'g')
plt.plot(History.history['val_loss'] , 'b')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()
def get_model_3(max_work, max_user):
    dim_embedddings = 30
    bias = 1
    # inputs - part 1
    w_inputs = Input(shape=(1,), dtype='int32')
    w = Embedding(max_work+1, dim_embedddings, name="work")(w_inputs)
    w_bis = Embedding(max_work + 1, bias, name="workbias")(w_inputs)

    # context - part 2
    u_inputs = Input(shape=(1,), dtype='int32')
    u = Embedding(max_user+1, dim_embedddings, name="user")(u_inputs)
    u_bis = Embedding(max_user + 1, bias, name="userbias")(u_inputs)
    
    # dot product to find similarity - part 3
    o = multiply([w, u])
    #o = dot([w,u],name='Simalarity-Dot-Product',axes=1)
    o = Dropout(0.5)(o)
    o = concatenate([o, u_bis, w_bis])
    o = Flatten()(o)
    o = Dense(10, activation="relu")(o)
    o = Dense(1)(o)

    # Ensembling part 1 , part 2, part 3 to make final Model
    rec_model = Model(inputs=[w_inputs, u_inputs], outputs=o)
    #rec_model.summary()
    
    # compile Model
    rec_model.compile(loss='mae', optimizer='adam', metrics=["mae"])

    return rec_model
model=get_model_3(num_users,num_movies)
model.compile(optimizer=Adam(lr=1e-3),loss='mse')
History = model.fit([train.userId,train.movieId],train.rating, batch_size=128,
                              epochs =40, validation_data = ([valid.userId,valid.movieId],valid.rating),
                              verbose = 1)
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5
import matplotlib.pyplot as plt
plt.plot(History.history['loss'] , 'g')
plt.plot(History.history['val_loss'] , 'b')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()
from sklearn.metrics import mean_absolute_error, mean_squared_error
predictions = model.predict([train.movieId, train.userId])

test_performance = mean_absolute_error(train["rating"], predictions)
test_performance1 = mean_squared_error(train["rating"], predictions)


print(" Test Mae model 1 : %s " % test_performance)
print(" Test Mae model 1 : %s " % test_performance1)
#Pandas sample() is used to generate a sample random row or column from the function caller data frame.
df2 = df.sample(frac=1, random_state=42)
x = df2[["userId", "movieId"]].values
# Normalize the targets between 0 and 1. Makes it easy to train.
y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
# Assuming training on 90% of the data and validating on 10%.
train_indices = int(0.9 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:],
)
x_val.shape, y_val.shape, x_train.shape, y_train.shape
EMBEDDING_SIZE = 50


class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        # Add all the components (including bias)
        x = dot_user_movie + user_bias + movie_bias
        # The sigmoid activation forces the rating to between 0 and 1
        return tf.nn.sigmoid(x)


# model = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)
# model.compile(
#     loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=0.001)
# )


# history = model.fit(
#     x=x_train,
#     y=y_train,
#     batch_size=64,
#     epochs=5,
#     verbose=1,
#     validation_data=(x_val, y_val),
# )
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()
movie_df = pd.read_csv(movielens_dir + "/movies.csv")

# # Let us get a user and see the top recommendations.
user_id = df.userId.sample(1).iloc[0]
movies_watched_by_user = df[df.userId == user_id]
#movies_watched_by_user
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}

movies_not_watched = movie_df[
    ~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)]["movieId"]

# Take the moview from the training input data
movies_not_watched = list(
    set(movies_not_watched).intersection(set(movie2movie_encoded.keys()))
)

movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]


user_encoder = user2user_encoded.get(user_id)

user_movie_array = np.hstack(
    ([[user_id]] * len(movies_not_watched), movies_not_watched)
)

ratings = model.predict(user_movie_array).flatten()
top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_movie_ids = [
    movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
]

print("Showing recommendations for user: {}".format(user_id))
print("====" * 9)
print("Movies with high ratings from user")
print("----" * 8)
top_movies_user = (
    movies_watched_by_user.sort_values(by="rating", ascending=False)
    .head(5)
    .movieId.values
)
movie_df_rows = movie_df[movie_df["movieId"].isin(top_movies_user)]
for row in movie_df_rows.itertuples():
    print(row.title, ":", row.genres)

print("----" * 8)
print("Top 10 movie recommendations")
print("----" * 8)
recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
for row in recommended_movies.itertuples():
    print(row.title, ":", row.genres)