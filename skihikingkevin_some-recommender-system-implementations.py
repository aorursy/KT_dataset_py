import pandas as pd

import numpy as np
from pandas.api.types import CategoricalDtype



df_favorites = pd.read_csv('../input/meta_favorites.csv')#, chunksize = 2000000)

#for df_favorites in df_favorites:

#    break

df_tracks = pd.read_csv('../input/meta_tracks.csv')



c_track = CategoricalDtype(sorted(df_favorites['track_id'].unique()), ordered=True)

c_user = CategoricalDtype(sorted(df_favorites['username'].unique()), ordered=True)



df_favorites['track_id'] = df_favorites['track_id'].astype(c_track)

df_favorites['username'] = df_favorites['username'].astype(c_user)
print(len(c_track.categories))

print(len(c_user.categories))
from sklearn.model_selection import train_test_split

train, test = train_test_split(df_favorites, test_size=0.15, random_state=420)

print(train.shape)

print(test.shape)
from scipy.sparse import  csr_matrix

train_data = csr_matrix((np.ones(len(train)), (train['username'].cat.codes, train['track_id'].cat.codes)), 

                  shape=( len(c_user.categories), len(c_track.categories)))

print(f'{100*train_data.sum()/(train_data.shape[0]*train_data.shape[1])}% Sparsity')
# Simple Cosine Similarity

from scipy.stats import percentileofscore

def model_simple(data, pred):

    from sklearn.metrics.pairwise import cosine_similarity

    # Model Simple: Cosine Similarity to obtain a matrix and then find closest based on the similarity matrix

    

    # normalize the matrix for each user (% importance to user)

    norm_data = data.multiply(1/data.sum(axis=1)).tocsr()

    # apply cosine similarity

    sim = cosine_similarity(norm_data.transpose(), dense_output=False)

    sim[np.diag_indices(sim.shape[0])] = 0

    denom = np.asarray(sim.sum(axis=1)).reshape(-1)



    # do it in chunks else we get memory error

    u_idx = pred['username'].cat.codes.values

    i_idx = pred['track_id'].cat.codes.values



    n_chunks = 30

    chunks = np.array_split(np.arange(norm_data.shape[0]), n_chunks)

    res = []

    previous_max = 0

    for i,idx in enumerate(chunks):

        print(f'Doing Chunk {i+1}/{n_chunks}')

        score = (norm_data[idx].dot(sim)) / denom

        score = (-score).argsort() / denom.shape[0]

        sel = (u_idx >= idx.min()) & (u_idx <= idx.max())

        chunk_score = np.asarray(score[u_idx[sel] - previous_max, i_idx[sel]]).reshape(-1)

        res.append(chunk_score)

        previous_max = idx.max() + 1

    return np.concatenate(res)



# Evaluate model

model_pred = model_simple(train_data, test)

mpr = model_pred.sum()/len(test)

print(f'MPR Score: {mpr:.5f}')

rec = (model_pred < 0.5).sum()/len(test)

print(f'Recall Score: {rec:.5f}')
from sklearn.decomposition import NMF, TruncatedSVD



def model_mf(data, pred, model):

    # normalize the matrix for each user (% importance to user)

    #norm_data = data.multiply(1/data.sum(axis=1)).tocsr()

    W = model.fit_transform(data)

    H = model.components_



    # do it in chunks else we get memory error

    u_idx = pred['username'].cat.codes.values

    i_idx = pred['track_id'].cat.codes.values

    n_chunks = 25

    chunks = np.array_split(np.arange(W.shape[0]), n_chunks)

    res = []

    previous_max = 0

    for i,idx in enumerate(chunks):

        print(f'Doing Chunk {i+1}/{n_chunks}')

        score = (W[idx].dot(H))

        score = (-score).argsort() / score.shape[1]

        sel = (u_idx >= idx.min()) & (u_idx <= idx.max())

        chunk_score = np.asarray(score[u_idx[sel] - previous_max, i_idx[sel]]).reshape(-1)

        res.append(chunk_score)

        previous_max = idx.max() + 1

    return np.concatenate(res)



K = 20



# NMF

model = NMF(n_components = K, init = 'nndsvd')

model_pred = model_mf(train_data, test, model)

mpr = model_pred.sum()/len(test)

print(f'NMF MPR Score: {mpr:.5f}')

rec = (model_pred < 0.5).sum()/len(test)

print(f'Recall Score: {rec:.5f}')



# TruncatedSVD

model = TruncatedSVD(n_components = K)

model_pred = model_mf(train_data, test, model)

mpr = model_pred.sum()/len(test)

print(f'SVD MPR Score: {mpr:.5f}')

rec = (model_pred < 0.5).sum()/len(test)

print(f'Recall Score: {rec:.5f}')
# Create the Training Set

APPROX_NEGATIVE_SAMPLE_SIZE = int(len(train)*1.2)

n_users = c_user.categories.shape[0]

n_tracks = c_track.categories.shape[0]

# Create Training Set

train_users = train['username'].cat.codes.values

train_tracks = train['track_id'].cat.codes.values

train_labels = np.ones(len(train_users))

# insert negative samples

u = np.random.randint(n_users, size=APPROX_NEGATIVE_SAMPLE_SIZE)

i = np.random.randint(n_tracks, size=APPROX_NEGATIVE_SAMPLE_SIZE)

non_neg_idx = np.where(train_data[u,i] == 0)

train_users = np.concatenate([train_users, u[non_neg_idx[1]]])

train_tracks = np.concatenate([train_tracks, i[non_neg_idx[1]]])

train_labels = np.concatenate([train_labels, np.zeros(u[non_neg_idx[1]].shape[0])])

print((train_users.shape, train_tracks.shape, train_labels.shape))



# random shuffle the data (because Keras takes last 10% as validation split)

X = np.stack([train_users, train_tracks, train_labels], axis=1)

np.random.shuffle(X)
import keras

import tensorflow as tf

from keras.callbacks import EarlyStopping

from keras import backend as K

from keras.models import Sequential, Model

from keras.layers.core import Dense, Lambda, Activation

from keras.layers import Embedding, Input, Dense, Reshape, Flatten, merge

from keras.optimizers import RMSprop

from keras.regularizers import l2



def ncf_model(num_users, num_items, latent_dim, regs=[0,0]):

    # Input variables

    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')

    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')



    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'user_embedding', embeddings_regularizer = l2(regs[0]), input_length=1)

    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding', embeddings_regularizer = l2(regs[1]), input_length=1)   

    

    # Crucial to flatten an embedding vector!

    user_latent = Flatten()(MF_Embedding_User(user_input))

    item_latent = Flatten()(MF_Embedding_Item(item_input))

    

    # Element-wise product of user and item embeddings 

    predict_vector = merge.Multiply()([user_latent, item_latent])

    

    # Final prediction layer

    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction')(predict_vector)

    

    model = Model(inputs=[user_input, item_input], outputs=prediction)



    return model



# create the model

def recall(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall



model = ncf_model(n_users, n_tracks, 50, regs = [0,0])

model.compile(optimizer=RMSprop(lr=0.001), metrics = ['accuracy', recall],  loss='binary_crossentropy')



# Train Model

ES = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=1, verbose=1)

hist = model.fit([X[:,0], X[:,1]], X[:,2], batch_size=150000, epochs=20, validation_split = 0.1, verbose=1, callbacks = [ES])

score = model.evaluate([test['username'].cat.codes.values, test['track_id'].cat.codes.values], np.ones(test.shape[0]), verbose=1, batch_size=100000)

print(f'Test Loss: {score[0]}   |   Test Recall: {score[1]}')
# Plot it

import matplotlib.pyplot as plt

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(16,4))

ax1.plot(hist.history['loss'])

ax1.plot(hist.history['val_loss'])

ax1.set_title('Loss')

ax1.legend(['Train', 'Test'], loc='upper left')

ax2.plot(hist.history['recall'])

ax2.plot(hist.history['val_recall'])

ax2.set_title('Recall')

ax2.legend(['Train', 'Test'], loc='upper left')

plt.show()
del X,train_users,train_tracks,train_labels

# Calculate MPR like before

# do it in chunks else we get memory error

u_idx = test['username'].cat.codes.values

i_idx = test['track_id'].cat.codes.values

n_chunks = 100

chunks = np.array_split(np.arange(n_users), n_chunks)

res = []

previous_max = 0

for i,idx in enumerate(chunks):

    print(f'Doing Chunk {i+1}/{n_chunks}')

    cross_product = np.transpose([np.tile(np.arange(n_tracks), len(idx)), np.repeat(idx, len(np.arange(n_tracks)))])

    score = model.predict([cross_product[:,1], cross_product[:,0]], batch_size=500000, verbose=1).reshape(idx.shape[0], n_tracks)

    score = (-score).argsort() / score.shape[1]

    sel = (u_idx >= idx.min()) & (u_idx <= idx.max())

    chunk_score = np.asarray(score[u_idx[sel] - previous_max, i_idx[sel]]).reshape(-1)

    res.append(chunk_score)

    previous_max = idx.max() + 1

mpr = np.concatenate(res).sum()/len(test)

print(f'NCF MPR Score: {mpr:.5f}')

rec = (np.concatenate(res) < 0.5).sum()/len(test)

print(f'NCF Recall Score: {rec:.5f}')
import implicit # doesn't work with GPU active
def model_als(data, pred, n_factors=50):

    # initialize a model

    CONFIDENCE = 130

    model = implicit.als.AlternatingLeastSquares(factors=n_factors, calculate_training_loss=True)

    model.fit(CONFIDENCE*data.transpose())

    W = model.user_factors

    H = model.item_factors.transpose()

    

    # do it in chunks else we get memory error

    u_idx = pred['username'].cat.codes.values

    i_idx = pred['track_id'].cat.codes.values

    n_chunks = 30

    chunks = np.array_split(np.arange(W.shape[0]), n_chunks)

    res = []

    previous_max = 0

    for i,idx in enumerate(chunks):

        print(f'Doing Chunk {i+1}/{n_chunks}')

        score = (W[idx].dot(H))

        score = (-score).argsort() / score.shape[1]

        sel = (u_idx >= idx.min()) & (u_idx <= idx.max())

        chunk_score = np.asarray(score[u_idx[sel] - previous_max, i_idx[sel]]).reshape(-1)

        res.append(chunk_score)

        previous_max = idx.max() + 1

    return np.concatenate(res)



K = 50

model_pred = model_als(train_data, test, K)

mpr = model_pred.sum()/len(test)

print(f'ALS MPR Score: {mpr:.5f}')

rec = (model_pred < 0.5).sum()/len(test)

print(f'Recall Score: {rec:.5f}')
def model_pbr(data, pred, n_factors=50):

    # initialize a model

    model = implicit.bpr.BayesianPersonalizedRanking(factors=n_factors, iterations = 50)

    model.fit(data.transpose())

    W = model.user_factors

    H = model.item_factors.transpose()

    

    # do it in chunks else we get memory error

    u_idx = pred['username'].cat.codes.values

    i_idx = pred['track_id'].cat.codes.values

    n_chunks = 30

    chunks = np.array_split(np.arange(W.shape[0]), n_chunks)

    res = []

    previous_max = 0

    for i,idx in enumerate(chunks):

        print(f'Doing Chunk {i+1}/{n_chunks}')

        score = (W[idx].dot(H))

        score = (-score).argsort() / score.shape[1]

        sel = (u_idx >= idx.min()) & (u_idx <= idx.max())

        chunk_score = np.asarray(score[u_idx[sel] - previous_max, i_idx[sel]]).reshape(-1)

        res.append(chunk_score)

        previous_max = idx.max() + 1

    return np.concatenate(res)



K = 50

model_pred = model_pbr(train_data, test, K)

mpr = model_pred.sum()/len(test)

print(f'PBR MPR Score: {mpr:.5f}')

rec = (model_pred < 0.5).sum()/len(test)

print(f'Recall Score: {rec:.5f}')