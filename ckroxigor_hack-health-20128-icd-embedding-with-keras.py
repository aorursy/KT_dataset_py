# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Input, Lambda
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.losses import mse
from keras import backend as K

from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree
from sklearn import metrics

import matplotlib.pyplot as plt # plotting
import seaborn as sns
sns.set(style="darkgrid")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

df = pd.read_csv('../input/Hack2018_ES_cleaned.csv', parse_dates=[9])

# Clean data
# Parse GPS values
def parse_coord(value):
    if type(value) == str:
        return float(value.replace(',', '.'))
    return value

df.longitud_corregida = df.longitud_corregida.apply(parse_coord)
df.latitude_corregida = df.latitude_corregida.apply(parse_coord)

# Discarding ungeolocated values
lat_limit = 20
lon_limit = 0

discarded = df[(df.latitude_corregida <= lat_limit) | (df.longitud_corregida <= lon_limit)]
print("Discarding {} entries outside Catalonia".format(len(discarded)))
df = df[(df.latitude_corregida > lat_limit) & (df.longitud_corregida > lon_limit)]

def twoDigits(n):
    if n < 10:
        return '0' + str(n)
    return str(n)

def format_month(d):
    month = "{}-{}-01".format(d.year, twoDigits(d.month))
    return pd.Timestamp(month)

def format_date(d):
    month = "{}-{}-{}".format(d.year, twoDigits(d.month), twoDigits(d.day))
    return pd.Timestamp(month)

# Add month value
def format_n_month(d):
    return "{}".format(d.month if d.month > 9 else "0" + str(d.month))

# Filter by date
len_before = len(df)
df = df[(df.Fecha >= '2015-09-01 00:00:00') & (df.Fecha < '2018-09-01 00:00:00')]
print("Discarding {} because date out of range".format(len_before - len(df)))

df['month'] = df.Fecha.apply(format_month)
df['n_month'] = df.Fecha.apply(format_n_month)
df['date'] = df.Fecha.apply(format_date)

# Any results you write to the current directory are saved as output.
df.head()
vectors = df.pivot_table(values='nasistencias', index='date', columns=['E_class', 'id_poblacion'], aggfunc=np.sum).reset_index()
vectors = vectors.fillna(0)
vectors.head()
X = vectors.values[:,1:]
X = normalize(X, axis=1, norm='l1')
X.shape
EMBEDDING_SIZE = 8
HIDDEN_SIZE = 64
batch_size = 32
epochs = 1000
PATIENCE = 50
_, n_features = X.shape
# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

inputs = Input(shape=(n_features,), name='encoder_input')
x = Dense(HIDDEN_SIZE, activation='relu')(inputs)
z_mean = Dense(EMBEDDING_SIZE, name='z_mean')(x)
z_log_var = Dense(EMBEDDING_SIZE, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(EMBEDDING_SIZE,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# build decoder model
latent_inputs = Input(shape=(EMBEDDING_SIZE,), name='z_sampling')
x = Dense(HIDDEN_SIZE, activation='tanh')(latent_inputs)
outputs = Dense(n_features, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
print("Model structure")
decoder.summary()

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

# Loss
reconstruction_loss = mse(inputs, outputs)
reconstruction_loss *= n_features
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam', metrics=['mse'])

model_path = "vae1.hdf5"
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,  
          write_graph=True, write_images=True)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1),
    ModelCheckpoint(filepath=model_path, verbose=0, save_best_only=True, monitor='val_loss'),
    tbCallBack
]

# Open with: tensorboard --logdir ./Graph 

vae.fit(X, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True, validation_split=0.2,
    callbacks=callbacks)#, callbacks=callbacks) # , callbacks=[callbacks]
model = load_model(model_path, compile=False)
encoder = model.layers[1]

#print(encoder.summary())
embedding = encoder.predict(X)[0]
n_assistencias = np.sum(vectors.values[:,1:], axis=1)
np.transpose(np.transpose(X) * n_assistencias)
metrics.mean_absolute_error(np.transpose(np.transpose(X) * n_assistencias), vectors.values[:,1:])
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("normalized MSE: {}".format(metrics.mean_squared_error(X, model.predict(X))))
print("normalized MAE: {}".format(metrics.mean_absolute_error(X, model.predict(X))))

print("cell value MSE: {}".format(metrics.mean_squared_error(vectors.values[:,1:], np.transpose(np.transpose(model.predict(X)) * n_assistencias))))
print("cell value MAE: {}".format(metrics.mean_absolute_error(vectors.values[:,1:], np.transpose(np.transpose((model.predict(X))) * n_assistencias))))

print("day level value MSE: {}".format(metrics.mean_squared_error(np.sum(vectors.values[:,1:], axis=1), np.sum(np.transpose(np.transpose(model.predict(X)) * n_assistencias), axis=1))))
print("day level value MAE: {}".format(metrics.mean_absolute_error(np.sum(vectors.values[:,1:], axis=1), np.sum(np.transpose(np.transpose((model.predict(X))) * n_assistencias), axis=1))))
print("day level value MAPE: {}".format(mean_absolute_percentage_error(np.sum(vectors.values[:,1:], axis=1), np.sum(np.transpose(np.transpose((model.predict(X))) * n_assistencias), axis=1))))
tree = BallTree(embedding, leaf_size=10)
dist, ind = tree.query(embedding[:1], k=4) 
vectors.values[ind]
plt.figure(figsize=(12,5))
embedding_pca = PCA(n_components=2).fit_transform(X)
fig = sns.scatterplot(x=embedding_pca[:,0], y=embedding_pca[:,1], hue=vectors.date.apply(lambda d:  abs(6 - d.month)))
fig.set_title("VAE + PCA embedding")
plt.figure(figsize=(12,5))
tsne_model = TSNE(perplexity=40, n_components=2, init='random', n_iter=2500, random_state=42, verbose=1)
embedding_tsne = tsne_model.fit_transform(embedding)
fig = sns.scatterplot(x=embedding_tsne[:,0], y=embedding_tsne[:,1], hue=vectors.date.apply(lambda d:  abs(6 - d.month)))
fig.set_title("VAE + T-SNE embedding")