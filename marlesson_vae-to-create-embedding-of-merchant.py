import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import missingno as msno
import matplotlib.pyplot as plt
import os

%matplotlib inline
df = pd.read_csv('../input/merchants.csv')

print("Size of the dataframe: ", df.shape); display(df.head(5))
df.info()
# Filter onlu nissing values
null_columns=df.columns[df.isnull().any()]
msno.bar(df[null_columns])
for c in ['avg_sales_lag3', 'avg_sales_lag6', 'avg_sales_lag12']:
    df[c] = df[c].fillna(df[c].mean())
# add other category 
df['category_2'] = df.category_2.fillna(df.category_2.max()+1)
# replace inf to zero
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
#merchant_group_id
categorical_columns = ['merchant_category_id','subsector_id',
                       'category_1', 'most_recent_sales_range', 'most_recent_purchases_range',
                       'category_4', 'city_id', 'state_id', 'category_2']

df_enc = pd.get_dummies(df, columns=categorical_columns)
print(df_enc.shape)
df_enc.head()
from sklearn.preprocessing import MinMaxScaler

scaler    = MinMaxScaler()
df_values = df_enc.drop('merchant_id', axis=1)
df_norm   = scaler.fit_transform(df_values)
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential

import argparse
import os
# network parameters
original_dim= df_enc.shape[1]-1
input_shape = (original_dim, )
intermediate_dim = int(original_dim/2)
batch_size = 128
latent_dim = 64
epochs     = 80
epsilon_std = 1.0
class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs
# VAE Architecture
# * original_dim - Original Input Dimension
# * intermediate_dim - Hidden Layer Dimension
# * latent_dim - Latent/Embedding Dimension
def vae_arc(original_dim, intermediate_dim, latent_dim):
    # Decode
    decoder = Sequential([
        Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
        Dense(original_dim, activation='sigmoid')
    ])

    # Encode
    x = Input(shape=(original_dim,))
    h = Dense(intermediate_dim, activation='relu')(x)

    z_mu = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
    z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)

    eps = Input(tensor=K.random_normal(stddev=epsilon_std,
                                       shape=(K.shape(x)[0], latent_dim)))
    z_eps = Multiply()([z_sigma, eps])
    z = Add()([z_mu, z_eps])

    x_pred = decoder(z)
    
    return x, eps, z_mu, x_pred
def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
x, eps, z_mu, x_pred = vae_arc(original_dim, intermediate_dim, latent_dim)
vae            = Model(inputs=[x, eps], outputs=x_pred)
vae.compile(optimizer='adam', loss=nll)
vae.summary()
from sklearn.model_selection import train_test_split

# 
X_train, X_test, y_train, y_test = train_test_split(df_norm, df_norm, 
                                                    test_size=0.33, random_state=42)
filepath   ="weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# train
hist = vae.fit(X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        validation_data=(X_test, X_test))
def plt_hist(hist):
    # summarize history for loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
plt_hist(hist)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plt_reduce(x, color='merchant_category_id'):
    '''
    Plot Scatter with color
    '''
    plt.figure(figsize=(6, 6))
    plt.scatter(x[:, 0], x[:, 1], c=df[color],
            alpha=.4, s=3**2, cmap='viridis')
    #plt.colorbar()
    plt.show()
# Predict Embedding values
encoder = Model(x, z_mu)
z_df    = encoder.predict(df_norm, batch_size=batch_size)
# Reduce dimmension
pca      = PCA(n_components=2)
x_reduce = pca.fit_transform(z_df)
# Plot with merchant_category_id color
plt_reduce(x_reduce, 'merchant_category_id')
# Plot with subsector_id color
plt_reduce(x_reduce, 'subsector_id')
# Plot with city_id color
plt_reduce(x_reduce, 'city_id')
df_embedding = pd.DataFrame(z_df)
df_embedding['merchant_id'] = df.merchant_id
df_embedding.head(5)
df_embedding.to_csv('merchant_id_embedding.csv')