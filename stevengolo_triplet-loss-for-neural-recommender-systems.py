# Load libraries

import umap



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import plotly.express as px

import tensorflow as tf



from collections import deque



from sklearn.manifold import TSNE

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import QuantileTransformer



from tensorflow.keras import layers

from tensorflow.keras.layers import (Concatenate, Dense, Dot, Dropout,

                                     Embedding, Flatten, Input, Lambda)

from tensorflow.keras.models import Model

from tensorflow.keras.regularizers import l2
# Load and preprocess rating files

df_raw = pd.read_csv('../input/anime-recommendations-database/rating.csv')
print(f"Shape of the ratings data: {df_raw.shape}.") 
df_raw.head(5)
# Load metadata file

metadata = pd.read_csv('../input/anime-recommendations-database/anime.csv')
print(f"Shape of the metadata: {metadata.shape}.")
metadata.head(5)
ratings = df_raw.merge(metadata.loc[:, ['name', 'anime_id', 'type', 'episodes']], left_on='anime_id', right_on='anime_id')
print(f"Shape of the complete data: {ratings.shape}.")
ratings.head(5)
print(f"Number of unique users: {ratings['user_id'].unique().size}.")
print(f"Number of unique animes: {ratings['anime_id'].unique().size}.")
# Histogram of the ratings

x, height = np.unique(ratings['rating'], return_counts=True)



fig, ax = plt.subplots()

ax.bar(x, height, align='center')

ax.set(xticks=np.arange(-1, 11), xlim=[-1.5, 10.5])

plt.show()
# Count the number of ratings for each movie

popularity = ratings.groupby('anime_id').size().reset_index(name='popularity')

metadata = metadata.merge(popularity, left_on='anime_id', right_on='anime_id')
# Get most popular anime id and TV shows

metadata_5000 = metadata.loc[(metadata['popularity'] > 5000) & (metadata['type'] == 'TV')]

# Remove -1 ratings and user id less than 10000

ratings = ratings[(ratings['rating'] > -1) & (ratings['user_id'] < 10000)]
# Create a dataframe for anime_id

metadata_5000 = metadata_5000.assign(new_anime_id=pd.Series(np.arange(metadata_5000.shape[0])).values)

metadata_5000_indexed = metadata_5000.set_index('new_anime_id')
# Merge the dataframe

ratings = ratings.merge(metadata_5000.loc[:, ['anime_id', 'new_anime_id', 'popularity']], left_on='anime_id', right_on='anime_id')
# Create a dataframe for user_id

user = pd.DataFrame({'user_id': np.unique(ratings['user_id'])})

user = user.assign(new_user_id=pd.Series(np.arange(user.shape[0])).values)
# Merge the dataframe

ratings = ratings.merge(user, left_on='user_id', right_on='user_id')
ratings.head(5)
print(f'Shape of the rating dataset: {ratings.shape}.')
MAX_USER_ID = ratings['new_user_id'].max()

MAX_ITEM_ID = ratings['new_anime_id'].max()



N_USERS = MAX_USER_ID + 1

N_ITEMS = MAX_ITEM_ID + 1
print(f'Number of users: {N_USERS} / Number of animes: {N_ITEMS}')
train, test = train_test_split(ratings, test_size=0.2, random_state=42)



user_id_train = np.array(train['new_user_id'])

anime_id_train = np.array(train['new_anime_id'])

ratings_train = np.array(train['rating'])



user_id_test = np.array(test['new_user_id'])

anime_id_test = np.array(test['new_anime_id'])

ratings_test = np.array(test['rating'])
train_pos = train.query('rating > 7')

test_pos = test.query('rating > 7')
def identity_loss(y_true, y_pred):

    """Ignore y_true and return the mean of y_pred.

    This is a hack to work-around the design of the Keras API that is

    not really suited to train networks with a triplet loss by default.

    """

    return tf.reduce_mean(y_pred)





class MarginLoss(layers.Layer):

    """Define the loss for the triple architecture

    

    Parameters

    ----------

    margin: float, default=1.

        Define a margin (alpha)

    """

    def __init__(self, margin=1.):

        super().__init__()

        self.margin = margin

        

    def call(self, inputs):

        pos_pair_similarity = inputs[0]

        neg_pair_similarity = inputs[1]

        

        diff = neg_pair_similarity - pos_pair_similarity

        return tf.maximum(diff + self.margin, 0)
class TripletModel(Model):

    """Define the triplet model architecture

    

    Parameters

    ----------

    embedding_size: integer

        Size the embedding vector

    n_users: integer

        Number of user in the dataset

    n_items: integer

        Number of item in the dataset

    l2_reg: float or None

        Quantity of regularization

    margin: float

        Margin for the loss

        

    Arguments

    ---------

    margin: float

        Margin for the loss

    user_embedding: Embedding

        Embedding layer of user 

    item_embedding: Embedding

        Embedding layer of item

    flatten: Flatten

        Flatten layer

    dot: Dot

        Dot layer

    margin_loss: MarginLoss

        Loss layer

    """

    def __init__(self, n_users, n_items, embedding_size=64, l2_reg=None, margin=1.):

        super().__init__(name='TripletModel')

        

        # Define hyperparameters

        self.margin = margin

        l2_reg = None if l2_reg == 0 else l2(l2_reg)

        

        # Define Embedding layers

        self.user_embedding = Embedding(output_dim=embedding_size,

                                        input_dim=n_users,

                                        input_length=1,

                                        input_shape=(1,),

                                        name='user_embedding',

                                        embeddings_regularizer=l2_reg)

        # The following embedding parameters will be shared to encode

        # both the positive and negative items.

        self.item_embedding = Embedding(output_dim=embedding_size,

                                        input_dim=n_items,

                                        input_length=1,

                                        name='item_embedding',

                                        embeddings_regularizer=l2_reg)

        

        # The two following layers are without parameters, and can

        # therefore be used for oth potisitve and negative items.

        self.flatten = Flatten()

        self.dot = Dot(axes=1, normalize=True)

        

        # Define the loss

        self.margin_loss = MarginLoss(margin)

        

    def call(self, inputs, training=False, y=None, **kwargs):

        """

        Parameters

        ----------

        inputs: list with three elements

            First element corresponds to the users

            Second element corresponds to the positive items

            Third element correponds to the negative items

        """

        user_input = inputs[0]

        item_pos_input = inputs[1]

        item_neg_input = inputs[2]

        

        # Create embeddings

        user_embedding = self.user_embedding(user_input)

        user_embedding = self.flatten(user_embedding)

        

        item_pos_embedding = self.item_embedding(item_pos_input)

        item_pos_embedding = self.flatten(item_pos_embedding)



        item_neg_embedding = self.item_embedding(item_neg_input)

        item_neg_embedding = self.flatten(item_neg_embedding)

        

        # Similarity computation betweeitem_neg_embeddings

        pos_similarity = self.dot([user_embedding, item_pos_embedding])

        neg_similarity = self.dot([user_embedding, item_neg_embedding])



        return self.margin_loss([pos_similarity, neg_similarity])
# Define parameters

EMBEDDING_SIZE = 64

L2_REG = 1e-6



# Define a triplet model

triplet_model = TripletModel(N_USERS, N_ITEMS, EMBEDDING_SIZE, L2_REG)
class MatchModel(Model):

    """Define the triplet model architecture

    

    Parameters

    ----------

    user_layer: Embedding

        User layer from TripletModel

    item_layer: Embedding

        Item layer from TripletModel

        

    Arguments

    ---------

    user_embedding: Embedding

        Embedding layer of user 

    item_embedding: Embedding

        Embedding layer of item

    flatten: Flatten

        Flatten layer

    dot: Dot

        Dot layer

    """

    def __init__(self, user_layer, item_layer):

        super().__init__(name='MathcModel')



        # Reuse the layer from the triplet model

        self.user_embedding = user_layer

        self.item_embedding = item_layer

        

        self.flatten = Flatten()

        self.dot = Dot(axes=1, normalize=True)

        

    def call(self, inputs, **kwargs):

        """

        Parameters

        ----------

        inputs: list with three elements

            First element corresponds to the users

            Second element corresponds to the positive items

        """

        user_input = inputs[0]

        item_pos_input = inputs[1]

        

        # Create embeddings

        user_embedding = self.user_embedding(user_input)

        user_embedding = self.flatten(user_embedding)

        

        item_pos_embedding = self.item_embedding(item_pos_input)

        item_pos_embedding = self.flatten(item_pos_embedding)

                

        # Similarity computation between embeddings

        pos_similarity = self.dot([user_embedding, item_pos_embedding])

        

        return pos_similarity
# Define a match model

match_model = MatchModel(triplet_model.user_embedding, triplet_model.item_embedding)
def average_roc_auc(model, data_train, data_test):

    """Compute the ROC AUC for each user and average over users.

    

    Parameters

    ----------

    model: MatchModel

        A MatchModel to train

    data_train: numpy array

        Train set

    data_test: numpy array

        Test set

        

    Return

    ------

    Average ROC AUC scores across users

    """

    max_user_id = max(data_train['new_user_id'].max(),

                      data_test['new_user_id'].max())

    max_anime_id = max(data_train['new_anime_id'].max(),

                       data_test['new_anime_id'].max())

    

    user_auc_scores = []

    for user_id in range(1, max_user_id + 1):

        pos_item_train = data_train[data_train['new_user_id'] == user_id]

        pos_item_test = data_test[data_test['new_user_id'] == user_id]

        

        # Consider all the items already seen in the training set

        all_item_idx = np.arange(1, max_anime_id + 1)

        items_to_rank = np.setdiff1d(all_item_idx,

                                     pos_item_train['new_anime_id'].values)

        

        # Ground truth: return 1 for each item positively present in

        # the test set and 0 otherwise

        expected = np.in1d(items_to_rank,

                           pos_item_test['new_anime_id'].values)

        

        # At least one positive test value to rank

        if np.sum(expected) >= 1:

            repeated_user_id = np.empty_like(items_to_rank)

            repeated_user_id.fill(user_id)

            

            # Make prediction

            predicted = model.predict([repeated_user_id, items_to_rank], batch_size=4096)

            

            # Compute AUC scores

            user_auc_scores.append(roc_auc_score(expected, predicted))

        

    return sum(user_auc_scores) / len(user_auc_scores)

%%time

print(f'Average ROC AUC on the untrained model: {average_roc_auc(match_model, train_pos, test_pos)}.')
def sample_triplets(pos_data, max_item_id, random_seed=42):

    """Sample negative items ar random

    

    Parameters

    ----------

    pos_data: pd.DataFrame

        Dataframe of positive items

    max_item_id: integer

        Number of items in the complete dataframe

    random_seed: integer, default=42

        Random number generation

    

    Return

    ------

    A list with entries user_ids, pos_items_ids and neg_items_ids

    """

    rng = np.random.RandomState(random_seed)

    

    user_ids = pos_data['new_user_id'].values.astype('int64')

    pos_item_ids = pos_data['new_anime_id'].values.astype('int64')

    neg_item_ids = rng.randint(low=1, 

                               high=max_item_id + 1, 

                               size=len(user_ids), dtype='int64')

    return [user_ids, pos_item_ids, neg_item_ids]
# Define parameters

N_EPOCHS = 10

BATCH_SIZE = 64



# We plug the identity loss and a fake target variable ignored by 

# the model to be able to use the Keras API to train the model.

fake_y = np.ones_like(train_pos["new_user_id"], dtype='int64')

    

triplet_model.compile(loss=identity_loss, optimizer='adam')

    

for i in range(N_EPOCHS):

    # Sample new negative items to build different triplets at each epoch

    triplet_inputs = sample_triplets(train_pos, MAX_ITEM_ID, random_seed=i)

        

    # Fit the model incrementally by doing a single pass over the sampled triplets

    triplet_model.fit(x=triplet_inputs, y=fake_y,

                      shuffle=True, batch_size=BATCH_SIZE, epochs=1)
# Evaluate the convergence of the model. Ideally, we should prepare a

# validation set and compute this at each epoch but this is too slow.

test_auc = average_roc_auc(match_model, train_pos, test_pos)

print(f'Average ROC AUC on the trained model: {test_auc}.')
# Print summary of triplet model

triplet_model.summary()
# Print summary of match model

match_model.summary()
class MLP(layers.Layer):

    """Define the MLP layer for the triplet architecture

    

    Parameters

    ----------

    n_hidden: Integer, default=1

        Number of hidden layer

    hidden_size: list of size `n_hidden`

        Output size of the hidden layer

    p_dropout: float, default=0.

        Probability for the Dropout layer

    l2_reg: float, default=None

        Regularizer

        

    Argument

    --------

    layers: list of Layer

        The different layers used in the MLP

    """

    def __init__(self, n_hidden=1, hidden_size=[64], p_dropout=0., l2_reg=None):

        super().__init__()

        

        self.layers = [Dropout(p_dropout)]

        

        for i in range(n_hidden):

            self.layers.append(Dense(hidden_size[i], 

                                     activation='relu', 

                                     kernel_regularizer=l2_reg))

            self.layers.append(Dropout(p_dropout))

        

        self.layers.append(Dense(1, 

                                 activation='relu', 

                                 kernel_regularizer=l2_reg))

        

    def call(self, x, training=False):

        for layer in self.layers:

            if isinstance(layer, Dropout):

                x = layer(x, training=training)

            else:

                x = layer(x)

        return x

    

    

class DeepTripletModel(Model):

    """Define the triplet model architecture

    

    Parameters

    ----------

    embedding_size_user: integer

        Size of the embedding vector for the user

    embedding_size_item: integer

        Size of the embedding vector for the item

    n_users: integer

        Number of user in the dataset

    n_items: integer

        Number of item in the dataset

    n_hidden: Integer, default=1

        Number of hidden layer

    hidden_size: list of size `n_hidden`

        Output size of the hidden layer

    l2_reg: float or None

        Quantity of regularization

    margin: float

        Margin for the loss

    p_dropout: float, default=0.

        Probability for the Dropout layer

        

    Arguments

    ---------

    margin: float

        Margin for the loss

    user_embedding: Embedding

        Embedding layer of user 

    item_embedding: Embedding

        Embedding layer of item

    flatten: Flatten

        Flatten layer

    concat: Concetenate

        Concatenate layer

    mlp: MLP

        MLP layer

    margin_loss: MarginLoss

        Loss layer

    """

    def __init__(self, n_users, n_items, 

                 embedding_size_user=64, embedding_size_item=64, 

                 n_hidden=1, hidden_size=[64], 

                 l2_reg=None, margin=1., p_dropout=0.):

        super().__init__(name='TripletModel')

        

        # Define hyperparameters

        self.margin = margin

        l2_reg = None if l2_reg == 0 else l2(l2_reg)

        

        # Define Embedding layers

        self.user_embedding = Embedding(output_dim=embedding_size_user,

                                        input_dim=n_users,

                                        input_length=1,

                                        input_shape=(1,),

                                        name='user_embedding',

                                        embeddings_regularizer=l2_reg)

        # The following embedding parameters will be shared to encode

        # both the positive and negative items.

        self.item_embedding = Embedding(output_dim=embedding_size_item,

                                        input_dim=n_items,

                                        input_length=1,

                                        name='item_embedding',

                                        embeddings_regularizer=l2_reg)

        

        # The two following layers are without parameters, and can

        # therefore be used for oth potisitve and negative items.

        self.flatten = Flatten()

        self.concat = Concatenate()

        

        # Define the MLP

        self.mlp = MLP(n_hidden, hidden_size, p_dropout, l2_reg)

        

        # Define the loss

        self.margin_loss = MarginLoss(margin)

        

    def call(self, inputs, training=False, **kwargs):

        """

        Parameters

        ----------

        inputs: list with three elements

            First element corresponds to the users

            Second element corresponds to the positive items

            Third element correponds to the negative items

        """

        user_input = inputs[0]

        item_pos_input = inputs[1]

        item_neg_input = inputs[2]

        

        # Create embeddings

        user_embedding = self.user_embedding(user_input)

        user_embedding = self.flatten(user_embedding)

        

        item_pos_embedding = self.item_embedding(item_pos_input)

        item_pos_embedding = self.flatten(item_pos_embedding)



        item_neg_embedding = self.item_embedding(item_neg_input)

        item_neg_embedding = self.flatten(item_neg_embedding)

        

        # Concatenate embeddings

        pos_embeddings_pair = self.concat([user_embedding, item_pos_embedding])

        neg_embeddings_pair = self.concat([user_embedding, item_neg_embedding])

        

        # Pass trough the MLP

        pos_similarity = self.mlp(pos_embeddings_pair)

        neg_similarity = self.mlp(neg_embeddings_pair)

        

        return self.margin_loss([pos_similarity, neg_similarity])



    

class DeepMatchModel(Model):

    """Define the triplet model architecture

    

    Parameters

    ----------

    user_layer: Embedding

        User layer from TripletModel

    item_layer: Embedding

        Item layer from TripletModel

    mlp: MLP

        MLP layer from TripletModel



    Arguments

    ---------

    user_embedding: Embedding

        Embedding layer of user 

    item_embedding: Embedding

        Embedding layer of item

    mlp: MLP

        MLP layer

    flatten: Flatten

        Flatten layer

    concat: Concatenate

        Concatenate layer

    """

    def __init__(self, user_layer, item_layer, mlp):

        super().__init__(name='MatchModel')



        # Reuse the layer from the triplet model

        self.user_embedding = user_layer

        self.item_embedding = item_layer

        self.mlp = mlp

        

        self.flatten = Flatten()

        self.concat = Concatenate()

        

    def call(self, inputs, **kwargs):

        """

        Parameters

        ----------

        inputs: list with three elements

            First element corresponds to the users

            Second element corresponds to the positive items

        """

        user_input = inputs[0]

        item_pos_input = inputs[1]

        

        # Create embeddings

        user_embedding = self.user_embedding(user_input)

        user_embedding = self.flatten(user_embedding)

        

        item_pos_embedding = self.item_embedding(item_pos_input)

        item_pos_embedding = self.flatten(item_pos_embedding)

        

        pos_embeddings_pair = self.concat([user_embedding, item_pos_embedding])

        

        # Similarity computation between embeddings

        pos_similarity = self.mlp(pos_embeddings_pair)

        

        return pos_similarity
# Define and train the model

HYPER_PARAM = dict( 

    embedding_size_user=32, 

    embedding_size_item=64, 

    n_hidden=1, 

    hidden_size=[128], 

    l2_reg=0., 

    margin=0.5, 

    p_dropout=0.1)



deep_triplet_model = DeepTripletModel(N_USERS, N_ITEMS, **HYPER_PARAM)

deep_match_model = DeepMatchModel(deep_triplet_model.user_embedding, 

                                  deep_triplet_model.item_embedding, 

                                  deep_triplet_model.mlp)
print(f'Average ROC AUC on the untrained model: {average_roc_auc(deep_match_model, train_pos, test_pos)}.')
# Define parameters

N_EPOCHS = 20



# We plug the identity loss and a fake target variable ignored by 

# the model to be able to use the Keras API to train the model.

fake_y = np.ones_like(train_pos["new_user_id"], dtype='int64')

    

deep_triplet_model.compile(loss=identity_loss, optimizer='adam')

    

for i in range(N_EPOCHS):

    # Sample new negative items to build different triplets at each epoch

    triplet_inputs = sample_triplets(train_pos, MAX_ITEM_ID, random_seed=i)

        

    # Fit the model incrementally by doing a single pass over the sampled triplets

    deep_triplet_model.fit(x=triplet_inputs, y=fake_y,

                      shuffle=True, batch_size=BATCH_SIZE, epochs=1)
# Evaluate the convergence of the model. Ideally, we should prepare a

# validation set and compute this at each epoch but this is too slow.

test_auc = average_roc_auc(deep_match_model, train_pos, test_pos)

print(f'Average ROC AUC on the trained model: {test_auc}.')
deep_triplet_model.summary()
deep_match_model.summary()