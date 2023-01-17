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
global obs_cnt
obs_cnt = 0
act_cnt = 0 
def observation(comment,_type=0):
    global obs_cnt
    global act_cnt
    
    if _type == 0:
        obs_cnt= obs_cnt+1
        print("\nObservation-",obs_cnt,"->",comment,"\n")
    elif _type == 1:
        act_cnt= act_cnt+1
        print("\nActioned-",act_cnt,"->",comment,"\n")
bp="/kaggle/input/goodbooks-10k/" # base path
booksdf=pd.read_csv(bp+"books.csv")
booksdf.head(5)
observation("2278 records have duplicate rating")
ratingdf=pd.read_csv(bp+"ratings.csv")
print(ratingdf.shape)
ratingdf[['book_id','user_id']][ratingdf[['book_id','user_id']].duplicated()].shape # recodes with duplicate ratings

observation("""Before removing duplicate - rating - shape is (981756, 3) 
              After removing duplicate  - rating shape is (979478, 3)
            """,1
          )
x=ratingdf[['book_id','user_id']].drop_duplicates().index
ratingdf= ratingdf.loc[x,:]
#ratingdf.shape

## unisue users, books
n_users, n_books = len(ratingdf.user_id.unique()), len(ratingdf.book_id.unique())


f'The dataset includes {len(ratingdf)} ratings by {n_users} unique users on {n_books} unique books.'
from sklearn.model_selection import train_test_split

## split the data to train and test dataframes
train, test = train_test_split(ratingdf, test_size=0.1)

f"The training and testing data include {len(train), len(test)} records."
## import keras models, layers and optimizers
from keras.models import Sequential, Model
from keras.layers import Embedding, Flatten, Dense, Dropout, concatenate, multiply, Input
from keras.layers.merge import Dot, multiply, concatenate
from keras.optimizers import Adam
import tensorflow as tf
## specify learning rate (or use the default)
#opt_adam = Adam(lr = 0.002)

## compile model
#model_mf.compile(optimizer = opt_adam, loss = ['mse'], metrics = ['mean_absolute_error'])
#model_mf.compile(optimizer='adam',loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)) 

# History = model_mf.fit([train.user_id, train.book_id],
#                           train.rating,
#                           batch_size = 100,
#                           validation_split = 0.005,
#                           epochs = 4,
#                           verbose = 0)
# creating book embedding path
book_input = Input(shape=[1], name="Book-Input")
book_embedding = Embedding(n_books+1, 5, name="Book-Embedding")(book_input)
book_vec = Flatten(name="Flatten-Books")(book_embedding)

# creating user embedding path
user_input = Input(shape=[1], name="User-Input")
user_embedding = Embedding(n_users+1, 5, name="User-Embedding")(user_input)
user_vec = Flatten(name="Flatten-Users")(user_embedding)

# performing dot product and creating model
prod = Dot(name="Dot-Product", axes=1)([book_vec, user_vec])
model = Model([user_input, book_input], prod)
model.compile('adam', 'mean_squared_error')
## fit model
# from keras import backend as K
History = model.fit([train.user_id, train.book_id],
                          train.rating,
                          batch_size = 100,
                          validation_split = 0.5,
                          epochs = 4,
                          verbose = 1)
#History = model.fit([train.user_id, train.book_id], train.rating, epochs=5, verbose=1)

# creating user embedding path
user_input = Input(shape=[1], name="User-Input")
user_embedding = Embedding(n_users+1, 25, name="User-Embedding")(user_input)
user_vec = Flatten(name="Flatten-Users")(user_embedding)
user_vec=Dropout(0.40)(user_vec)

# creating book embedding path
book_input = Input(shape=[1], name="Book-Input")
book_embedding = Embedding(n_books+1, 25, name="Book-Embedding")(book_input)
book_vec = Flatten(name="Flatten-Books")(book_embedding)
book_vec=Dropout(0.40)(book_vec)


# model configuration part 3
sim=Dot(name="Dot-Product", axes=1)([user_vec,book_vec])
nn_inp=Dense(96,activation='relu')(sim)
nn_inp=Dropout(0.4)(nn_inp)
# nn_inp=BatchNormalization()(nn_inp)
nn_inp=Dense(1,activation='relu')(nn_inp)

# Ensemle Part 1, Part 2, Part 3
nn_model =Model([user_input, book_input],nn_inp)
nn_model.summary()

## fit model
# from keras import backend as K
nn_model.compile(optimizer=Adam(lr=1e-3),loss='mse')
History = nn_model.fit([train.user_id, train.book_id],
                          train.rating,
                          batch_size = 100,
                          validation_split = 0.5,
                          epochs = 4,
                          verbose = 1)
#History = model.fit([train.user_id, train.book_id], train.rating, epochs=5, verbose=1)
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
model=get_model_3(n_users,n_books)
#nn_model.compile(optimizer=Adam(lr=1e-3),loss='mse')
History = model.fit([train.user_id, train.book_id],
                          train.rating,
                          batch_size = 100,
                          validation_split = 0.5,
                          epochs = 4,
                          verbose = 1)
## show loss at each epoch
pd.DataFrame(History.history)
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
## define the number of latent factors (can be different for the users and books)
dim_embedding_user = 50
dim_embedding_book = 30

## book embedding
book_input= Input(shape=[1], name='Book')
book_embedding = Embedding(n_books + 1, dim_embedding_book, name='Book-Embedding')(book_input)
book_vec = Flatten(name='Book-Flatten')(book_embedding)
book_vec = Dropout(0.2)(book_vec)

## user embedding
user_input = Input(shape=[1], name='User')
user_embedding = Embedding(n_users + 1, dim_embedding_user, name ='User-Embedding')(user_input)
user_vec = Flatten(name ='User-Flatten')(user_embedding)
user_vec = Dropout(0.2)(user_vec)

## concatenate flattened values 
concat = concatenate([book_vec, user_vec])
concat_dropout = Dropout(0.2)(concat)

## add dense layer (can try more)
dense_1 = Dense(20, name ='Fully-Connected1', activation='relu')(concat)

## define output (can try sigmoid instead of relu)
result = Dense(1, activation ='relu',name ='Activation')(dense_1)

## define model with 2 inputs and 1 output
model_tabular = Model([user_input, book_input], result)

## show model summary
model_tabular.summary()

## specify learning rate (or use the default by specifying optimizer = 'adam')
opt_adam = Adam(lr = 0.002)

## compile model
model_tabular.compile(optimizer= opt_adam, loss= ['mse'], metrics=['mean_absolute_error'])

## fit model
history_tabular = model_tabular.fit([train['user_id'], train['book_id']],
                                    train['rating'],
                                    batch_size = 256,
                                    validation_split = 0.20,
                                    epochs = 10,
                                    verbose = 1)

# This model performs better that the first one after 4 epochs. Part of the experimentation is to train both 
# for more epochs, 
# tune the hyper parameters or modify the architecture.


from pylab import rcParams
rcParams['figure.figsize'] = 10, 5
import matplotlib.pyplot as plt
plt.plot(history_tabular.history['loss'] , 'g')
plt.plot(history_tabular.history['val_loss'] , 'b')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()
## show loss at each epoch
pd.DataFrame(history_tabular.history)


## import libraries

from sklearn.metrics import mean_absolute_error, mean_squared_error

## define a function to return arrays in the required form 
def get_array(series):
    return np.array([[element] for element in series])

## predict on test data  
predictions = model_tabular.predict([get_array(test['user_id']), get_array(test['book_id'])])

f'mean squared error on test data is {mean_squared_error(test["rating"], predictions)}'
## get weights of the books embedding matrix
book_embedding_weights = model_tabular.get_layer('Book-Embedding').get_weights()[0]
book_embedding_weights.shape
## import PCA
from sklearn.decomposition import PCA
import seaborn as sns

pca = PCA(n_components = 2) ## use 3 components
pca_result1 = pca.fit_transform(book_embedding_weights)
sns.scatterplot(x=pca_result1[:,0], y=pca_result1[:,1])
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tnse_results = tsne.fit_transform(book_embedding_weights)
sns.scatterplot(x=tnse_results[:,0], y=tnse_results[:,1])
## import PCA
from sklearn.decomposition import PCA
import seaborn as sns

pca = PCA(n_components = 3) ## use 3 components
book_embedding_weights_t = np.transpose(book_embedding_weights) ## pass the transpose of the embedding matrix
book_pca = pca.fit(book_embedding_weights_t) ## fit

## display the resulting matrix dimensions
print(book_pca.components_.shape)

# We can look at the percentage of variance explained by each of the selected components. 
## display the variance explained by the 3 components
print(book_pca.explained_variance_ratio_)
## If the variance explained is very low, we might not be able to see a good interpretation. 
##However, for demo purposes, we will just extract the first component/factor that explains the highest 
## percentage of the variance. The array we get can be mapped to the books names as follows.

## create a dictionary out of bookid, book original title
books_dict = booksdf.set_index('book_id')['original_title'].to_dict()
books_dict
from operator import itemgetter

## extract first PCA
pca0 = book_pca.components_[0]

## get the value (pca0, book title)
book_comp0 = [(f, books_dict[i]) for f,i in zip(pca0, list(books_dict.keys()))]
book_comp0
## books corresponding to the highest values of pca0
sorted(book_comp0, key = itemgetter(0), reverse = True)[:10]
## books corresponding to the lowest values of pca0
sorted(book_comp0, key = itemgetter(0))[:10]
# Creating dataset for making recommendations for the first user
book_data = np.array(list(set(ratingdf.book_id)))
user = np.array([1 for i in range(len(book_data))])
predictions = model_tabular.predict([user, book_data])
predictions = np.array([a[0] for a in predictions])
recommended_book_ids = (-predictions).argsort()[:5]
print(recommended_book_ids)
print(predictions[recommended_book_ids])
ratingdf[ratingdf.user_id == 1]
booksdf[booksdf['id'].isin(recommended_book_ids)]

