# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Lets import some libraries that we will use
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
# To shift lists
from collections import deque
# Load single data-file
df_raw = pd.read_csv('../input/netflix-prize-data/combined_data_2.txt', header=None, names=['User', 'Rating', 'Date'], usecols=[0, 1, 2])


# Find empty rows to slice dataframe for each movie
tmp_movies = df_raw[df_raw['Rating'].isna()]['User'].reset_index()
movie_indices = [[index, int(movie[:-1])] for index, movie in tmp_movies.values]

# Shift the movie_indices by one to get start and endpoints of all movies
shifted_movie_indices = deque(movie_indices)
shifted_movie_indices.rotate(-1)


# Gather all dataframes
user_data = []

# Iterate over all movies
for [df_id_1, movie_id], [df_id_2, next_movie_id] in zip(movie_indices, shifted_movie_indices):
    
    # Check if it is the last movie in the file
    if df_id_1<df_id_2:
        tmp_df = df_raw.loc[df_id_1+1:df_id_2-1].copy()
    else:
        tmp_df = df_raw.loc[df_id_1+1:].copy()
        
    # Create movie_id column
    tmp_df['Movie_Id'] = movie_id
    
    # Append dataframe to list
    user_data.append(tmp_df)

# Combine all dataframes
df = pd.concat(user_data)
del user_data, df_raw, tmp_movies, tmp_df, shifted_movie_indices, movie_indices, df_id_1, movie_id, df_id_2, next_movie_id
print('Shape User-Ratings:\t{}'.format(df.shape))
df.sample(5)
df_title = pd.read_csv('../input/netflix-prize-data/movie_titles.csv', encoding = "ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name'])
df_title.set_index('Movie_Id', inplace = True)
df_title.head()
data = pd.merge(df, df_title, on='Movie_Id')
data.sample(5)
df.drop(columns=['Date'],inplace=True)
df.head()
df['Movie'].value_counts()
df.groupby('Movie')['Rating'].mean().sort_values(ascending=False).head()


df=df.iloc[0:1000,:]
print(df.shape)
print(df.User.nunique())
print(df.Movie.nunique())
df.isna().sum()
df['User']=df['User'].astype(int)
df.dtypes
columns_titles = ["Movie",'User',"Rating"]
df=df.reindex(columns=columns_titles)
df.head()
from sklearn.model_selection import train_test_split
Xtrain, Xtest = train_test_split(df, test_size=0.2, random_state=1)
print(f"Shape of train data: {Xtrain.shape}")
print(f"Shape of test data: {Xtest.shape}")
#Get the number of unique entities in movies and users columns
nmovies_id = df.Movie.nunique()
nuser_id = df.User.nunique()
import tensorflow.keras as tf
#Movie input network
input_movies = tf.layers.Input(shape=[1])
embed_movies = tf.layers.Embedding(3000000 + 1,30)(input_movies)
movies_out = tf.layers.Flatten()(embed_movies)

#user input network
input_users = tf.layers.Input(shape=[1])
embed_users = tf.layers.Embedding(3000000 + 1,30)(input_users)
users_out = tf.layers.Flatten()(embed_users)

conc_layer = tf.layers.Concatenate()([movies_out, users_out])
x = tf.layers.Dense(4, activation='relu')(conc_layer)
x_out = x = tf.layers.Dense(1, activation='relu')(x)
model = tf.Model([input_movies, input_users], x_out)
opt = tf.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='mean_squared_error',metrics=['accuracy'])
model.summary()
hist = model.fit([Xtrain.Movie, Xtrain.User], Xtrain.Rating, 
                 batch_size=32, 
                 epochs=5, 
                 verbose=1,
                 validation_data=([Xtest.Movie, Xtest.User], Xtest.Rating))
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
plt.plot(train_loss, color='r', label='Train Loss')
plt.plot(val_loss, color='b', label='Validation Loss')
plt.title("Train and Validation Loss Curve")
plt.legend()
plt.show()
train_acc = hist.history['accuracy']
val_acc = hist.history['accuracy']
plt.plot(train_acc, color='r', label='Train accuracy')
plt.plot(val_acc, color='b', label='Validation accuracy')
plt.title("Train and Validation Accuracy Curve")
plt.legend()
plt.show()
