# installation of keras self attention layer

!pip install keras-self-attention



import os

from collections import Counter

from random import choice

import time



import numpy as np

import pandas as pd

import requests



from IPython.display import Image



import keras

from keras_self_attention import SeqSelfAttention



import tensorflow as tf

from tensorflow.keras.layers import Embedding, LSTM, Dense

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import backend as K

from tensorflow.keras import models



from skopt import gp_minimize

from skopt.space import Real, Categorical, Integer

from skopt.plots import plot_convergence

from skopt.utils import use_named_args
# Tensorflow version checking

print("Tensorflow version " + tf.__version__)
# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)
# define some parameters



## size of the train set

train_percent_split = 0.9



# first hyperparameters of the RNN model

epochs = 1

dropout = 0

embedding_size = 128

hidden_size_lstm = 128

learning_rate = 0.00276

attention_width = 20

add_dense_layer = True

hidden_size_dense = 128



validation_split = 0.1

n_reco = 20

n_min_interactions = 20

max_length_seq = 20

nb_last_item = 20

max_users = 5000



# iterations number for optimization with scikit-optimize lib

n_random_starts = 2

n_calls = 15
# paths definition

path = '/kaggle/input/bookcrossing-dataset/Book reviews/'

book_fp = os.path.join(path, 'BX-Books.csv')

user_fp = os.path.join(path, 'BX-Users.csv')

rating_fp = os.path.join(path, 'BX-Book-Ratings.csv')
# files reading

def read_file(fp):

    data = pd.read_csv(fp,

                       sep=';',

                       encoding='latin-1',

                       low_memory=False,

                       header=0,

                       error_bad_lines=False)

    return data

    

user_df = read_file(user_fp)

item_df = read_file(book_fp)

rating_df = read_file(rating_fp)
print(user_df.shape)

print(user_df.head())
print(item_df.shape)

print(item_df.head())
print(rating_df.shape)

print(rating_df.head())
rating_df = rating_df.sample(frac=1).reset_index(drop=True)
rating_df.isna().sum()
colname_mapping = {

    'ISBN': 'item',

    'User-ID': 'user',

    'Book-Rating': 'rating',

    'Book-Title': 'name',

    'Book-Author': 'author',

    'Image-URL-M': 'image',

    'Publisher': 'publisher',

    'Year-Of-Publication': 'year'

}

rating_df = rating_df.rename(columns=colname_mapping)

item_df = item_df.rename(columns=colname_mapping)

user_df = user_df.rename(columns=colname_mapping)
items = list(set(item_df.item.unique().tolist() + rating_df.item.unique().tolist()))

users = user_df.user.unique() 

print(f'number of unique items: {len(items)}\nnumber of unique users: {len(users)}')
rating_df = rating_df[rating_df.item.isin(items)]

item_df = item_df[item_df.item.isin(items)]

rating_df.shape, item_df.shape
user_to_token = {user: int(token) for token, user in enumerate(users)}

token_to_item = {token: user for user, token in user_to_token.items()}

item_to_token = {item: int(token) for token, item in enumerate(items)}

token_to_item = {token: item for item, token in item_to_token.items()}



rating_df['user_id'] = rating_df['user'].map(user_to_token)

rating_df['item_id'] = rating_df['item'].map(item_to_token).dropna().astype(int)



item_df['item_id'] = item_df['item'].map(item_to_token)

user_df['user_id'] = user_df['user'].map(user_to_token)
#%%timeit

#diff = set(item_df.item_id).difference(set(rating_df.item_id))

#len(diff), diff
item_df.item_id.nunique(), rating_df.item_id.nunique()
item_df.head()
def print_single_item_characteristics(item_id=None, item_df=item_df):

    if not item_id:

        item_id = choice(item_df.item_id)

    if item_id not in set(item_df.item_id):

        print(f'item_id {item_id} not in df')

        return None



    item_df = item_df[item_df['item_id'] == item_id]

    url = item_df.image.values[0]

    response = requests.get(url)



    print(f'item_id: {item_id}; '

          f'name: {item_df.name.values[0]}; '

          f'author: {item_df.author.values[0]} '

          f'publisher: {item_df.publisher.values[0]} '

          f'year: {item_df.year.values[0]}')

    return Image(url)



def print_items_characteristics(item_id_list):

    for item_id in item_id_list:

        display(print_single_item_characteristics(item_id=item_id))
n_rating_by_user = rating_df.user_id.value_counts()
n_rating_by_user.describe()
top3_item_id = list(rating_df.item_id.value_counts().index.values[:3])
top3_item_id
print_items_characteristics(item_id_list=top3_item_id)
user_occurence = Counter(rating_df.user).most_common()

print(user_occurence[:20])
def user_with_n_interaction(data, n):

    print(f'length before filtering: {len(data)}.')

    user_occurence = Counter(data.user)



    user_to_keep = [

        user

        for user, occ in user_occurence.items()

        if occ >= n

    ]



    data_filtered = data[data['user'].isin(user_to_keep)]

    print(f'length after filtering: {len(data_filtered)}.')

    return data_filtered



rating_df = user_with_n_interaction(data=rating_df, n=n_min_interactions)
split_ind = int(len(rating_df) * train_percent_split)

train, test = rating_df[:split_ind], rating_df[split_ind:]

print(f'shape of train: {train.shape}\nshape of test: {test.shape}')
#function to prepare sequence data

def prepare_sequences(data, users, item_to_token, max_length=20, 

                      one_hot_encoding=False):

    

    print('preparing sequences')

    

    #generate sequences - see https://stackoverflow.com/questions/36864699/pandas-pivot-dataframe-with-unequal-columns        

    data = pd.concat([

        pd.DataFrame(

            {

                g:[0] * (max_length+1-len(d['item_id'].tolist()[-max_length-1:])) + d['item_id'].tolist()[-max_length-1:]

            }

        )

        for g,d in data.groupby('user_id')], axis=1)

 

    

    #from pandas dataframe to numpy array

    data = data.transpose().values

        

    #transpose and build the arrays

    x = np.array([i[:-1] for i in data])

    y = np.array([i[1:] for i in data])

            

    #build the one-hot encoding, if we want

    if one_hot_encoding:

        y = np_utils.to_categorical(y, len(item_to_token)+1)

    else:

        y = np.expand_dims(y, -1)

    

    print('sequences prepared')

        

    return (x, y)



#function to extract prediction from keras model at last timestep

def predict_last_timestep(model, data):

    #calculate the model output

    prediction = model.predict(data)

    #keep only the prediction at the final timestep

    return prediction[-1]
x, y = prepare_sequences(data=train, users=users, item_to_token=item_to_token, max_length=max_length_seq)
def keras_model(hidden_size_lstm=hidden_size_lstm, 

                learning_rate=learning_rate, 

                dropout=dropout,

                attention_width=attention_width,

                embedding_size=embedding_size,

                add_dense_layer=add_dense_layer,

                hidden_size_dense=hidden_size_dense,

                embedding_matrix=None,

                item_to_token=item_to_token):

    

    with strategy.scope():

        embedding_layer = Embedding(len(item_to_token)+1,

                                    embedding_size,

                                    weights=embedding_matrix,

                                    mask_zero=True)



        model = Sequential()

        model.add(embedding_layer)

        model.add(LSTM(units=hidden_size_lstm,

                       activation='tanh', dropout=dropout,

                       return_sequences=True))

        model.add(SeqSelfAttention(attention_activation='sigmoid',

                                   attention_width=attention_width,

                                   history_only=True))

        if add_dense_layer:

            model.add(Dense(units=hidden_size_dense, activation='relu'))

        model.add(Dense(units=len(item_to_token)+1, activation='softmax'))

        optimizer = Adam(lr=learning_rate)

    # Compile model

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,

                  metrics=['sparse_categorical_accuracy'])



    return model
#prepare the model

model = keras_model(hidden_size_lstm=hidden_size_lstm, 

                    learning_rate=learning_rate, 

                    dropout=dropout, 

                    embedding_size=embedding_size)  



history = model.fit(x, y,

                    epochs=epochs,

                    validation_split=validation_split,

                    batch_size=64,

                    verbose=1)



print(history.history['val_sparse_categorical_accuracy'][-1])

common_user = list(set(train.user_id).intersection(test.user_id))

common_user[:10]
print(f'There are {len(common_user)} common users between the train and test set')
def reco_from_item_id_interacted(item_id_interacted, model=model, n_reco=10):

    predictions = predict_last_timestep(model=model,

                                        data=[item_id_interacted]).argsort()[0][:n_reco]

    return list(predictions)



def build_user_to_interacted(user_id, train=train):

    return train[train['user_id'] == user_id]['item_id'].dropna().unique().tolist()



#item_id_interacted = build_user_to_interacted(user_id=choice(user_train))

#len(item_id_interacted)
def user_to_last_visited_item_id_dict(train: pd.DataFrame,

                                      user_list: list,

                                      nb_last_item: int=None) -> dict:

    """

    Return a dictionary mapping user to last visited items id.

    input:

            :train: pd.DataFrame, training set

            :user_list: list, head_visitor_id

            :nb_last_item: int, number of last interacted items

                           to use to predict recommendation

    output:

            :: dict, mapping the user (head_visitor_id) to the last

               visited items

    """

    if nb_last_item:

        return train.groupby('user_id')['item_id'].apply(lambda g: g.values

                                                      .tolist()[-nb_last_item:]).to_dict()

    else:

        return train.groupby('user_id')['item_id'].apply(lambda g: g.values

                                                      .tolist()).to_dict()



user_to_last_visited_item_id_dict = user_to_last_visited_item_id_dict(train=train,

                                                                      user_list=train.user_id.unique().tolist(),

                                                                      nb_last_item=nb_last_item)
list(user_to_last_visited_item_id_dict.items())[:5]
def predict_for_one(

    model,

    user_id,

    n_reco,

    user_to_last_visited_item_id_dict=user_to_last_visited_item_id_dict

):

    item_id_interacted = user_to_last_visited_item_id_dict[user_id]

    reco_id = reco_from_item_id_interacted(item_id_interacted,

                                           model=model,

                                           n_reco=n_reco)

    return reco_id
def predict(

    model,

    user_id_list,

    n_reco,

    user_to_last_visited_item_id_dict=user_to_last_visited_item_id_dict,

    max_users=None, # reduce computation time

):

    start = time.time()

    print(f'recommendation computation for {len(user_id_list[:max_users])} users.')

    print(f'n_reco={n_reco}')

    reco_dict = {

        user_id: predict_for_one(

            model=model,

            user_id=user_id,

            n_reco=n_reco,

            user_to_last_visited_item_id_dict=user_to_last_visited_item_id_dict

        )

        for user_id in user_id_list[:max_users]

    }

    print(f'predict for {len(user_id_list[:max_users])} spent {round(time.time()-start, 2)} s.')

    return reco_dict

#max_users=100

#n_reco=50

reco_dict = predict(

    model=model,

    user_id_list=common_user,

    user_to_last_visited_item_id_dict=user_to_last_visited_item_id_dict,

    max_users=max_users,

    n_reco=n_reco)
list(reco_dict.items())[:3]
#Function to calculate, precision, recall and coverage

def statistics_at_k(reco_dict,

                    test_df,

                    train_df,

                    calculate_precision=True,

                    calculate_recall=True, 

                    calculate_coverage=True): 

    '''

    reco_dict: dictionary with the uid as key, list of items recommended as attribute

    test_df: dataframe of user-item interactions

    '''     

    #calculate precision

    if calculate_precision:

        k_relevant = 0

        k_total = 0

        for uid, iid in reco_dict.items():

            iid_test = set(test_df[test_df['user_id'] == uid]['item_id'])

            for j in iid:

                k_total += 1

                if j in iid_test:

                    k_relevant += 1

        if not k_total:

            precision = 0

        else:

            precision = k_relevant/k_total

        print(f'precision={precision}')    

    else:

        precision = None

        

    #calculate precision

    if calculate_recall:

        k_relevant = 0

        k_total = 0

        for uid, iid in reco_dict.items():

            for j in list(test_df[test_df['user_id'] == uid]['item_id']):

                k_total += 1

                if j in set(iid):

                    k_relevant += 1

        

        if not k_total:

            recall = 0

        else:

            recall = k_relevant/k_total

        print(f'recall={recall}')

    else:

        recall = None

        

    #calculate coverage

    if calculate_coverage:

        nb_recommended = len(set(sum(reco_dict.values(), [])))

        nb_total = len(train_df['item_id'].unique())

        coverage = nb_recommended/nb_total

        print(f'coverage={coverage}')

    else:

        coverage = None

    

    return precision, recall, coverage
# statistics with RNN recommendation

statistics_at_k(reco_dict=reco_dict,

                test_df=test,

                train_df=train,

                calculate_precision=True,

                calculate_recall=True, 

                calculate_coverage=True)
topn = list(train.item_id.value_counts().index.values[:n_reco])

topn
reco_topn_dict = {

    user_id: topn

    for user_id in reco_dict.keys() 

}
list(reco_topn_dict.items())[:3]
# statistics with top n recommendation

statistics_at_k(reco_dict=reco_topn_dict,

                test_df=test,

                train_df=train,

                calculate_precision=True,

                calculate_recall=True, 

                calculate_coverage=True)
item_id_interacted = [197383]

item_id_pred = predict_last_timestep(

    model=model,data=[item_id_interacted]

).argsort()[0][:20]
# past interactions

item_df[item_df['item_id'].isin(item_id_interacted)]
# visualization of the pas

print_items_characteristics(item_id_list=item_id_interacted)
# predictions

item_df[item_df['item_id'].isin(item_id_pred)]
item_id_pred
print_items_characteristics(item_id_list=item_id_pred)
del history

del model
# define the dimension to search

dim_epochs = Integer(low=1, high=15, name='epochs')

dim_hidden_size_lstm = Categorical(categories=[32, 64, 128, 256], name='hidden_size_lstm')

dim_learning_rate = Real(low=1e-4, high=5e-1, prior='log-uniform',

                         name='learning_rate')

dim_attention_width = Integer(low=1, high=50, name='attention_width')

dim_dropout = Real(low=0, high=0.9, name='dropout')

dim_embedding_size = Categorical(categories=[64, 128, 256, 512], name='embedding_size')

dim_add_dense_layer = Categorical(categories=[True, False], name='add_dense_layer')

dim_hidden_size_dense = Categorical(categories=[32, 64, 128, 256], name='hidden_size_dense')



dimensions = [dim_epochs,

              dim_hidden_size_lstm,

              dim_learning_rate,

              dim_attention_width,

              dim_dropout,

              dim_embedding_size,

              dim_add_dense_layer,

              dim_hidden_size_dense]



epochs = 3

dropout = 0

embedding_size = 256

hidden_size_lstm = 64

learning_rate = 0.00276
def optimize(dimensions=dimensions, n_calls=15, n_random_starts=3, verbose=1, x0=None):

    print(dimensions)

    @use_named_args(dimensions=dimensions)

    def fitness(**params):

        print(f'params={params}')

        model = keras_model(hidden_size_lstm=params['hidden_size_lstm'],

                            learning_rate=params['learning_rate'],

                            attention_width=params['attention_width'],

                            dropout=params['dropout'],

                            embedding_size=params['embedding_size'],

                            add_dense_layer=params['add_dense_layer'],

                            hidden_size_dense=params['hidden_size_dense'])  



        history = model.fit(x, y,

                            epochs=params['epochs'],

                            validation_split=validation_split,

                            batch_size=64,

                            verbose=verbose)

        sca = history.history['val_sparse_categorical_accuracy'][-1]

        print(f'##sca={sca}## with params={params}')

        del history

        del model

        return -1.0 * sca

    

    res = gp_minimize(func=fitness,

                      dimensions=dimensions,

                      acq_func='EI', # Expected Improvement.

                      n_calls=n_calls,

                      n_random_starts=n_random_starts,

                      x0=x0)

    print(f'best accuracy={-1.0 * res.fun} with {res.x}')

    return res
res = optimize(dimensions=dimensions,

               n_calls=n_calls,

               n_random_starts=n_random_starts,

               x0=[3, 256, 0.001, 40, 0, 512, True, 128])
res
y
x