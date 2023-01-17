# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
df = pd.read_csv('../input/creditcard.csv')
df.head()
df.describe()
df.isnull().sum()
df = df.drop('Time',axis=1)
X = df.drop('Class',axis=1).values 
y = df['Class'].values
X.shape
X -= X.min(axis=0)
X /= X.max(axis=0)
X.mean()
X.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.1)
from keras.models import Model
from keras.layers import Input, Dense
data_in = Input(shape=(29,))
encoded = Dense(12,activation='tanh')(data_in)
decoded = Dense(29,activation='sigmoid')(encoded)
autoencoder = Model(data_in,decoded)
autoencoder.compile(optimizer='adam',loss='mean_squared_error')
autoencoder.fit(X_train,
                X_train,
                epochs = 20, 
                batch_size=128, 
                validation_data=(X_test,X_test))
X_test.mean()
pred = autoencoder.predict(X_test[0:10])
import matplotlib.pyplot as plt
import numpy as np

width = 0.8

prediction   = pred[9]
true_value    = X_test[9]

indices = np.arange(len(prediction))

fig = plt.figure(figsize=(10,7))

plt.bar(indices, prediction, width=width, 
        color='b', label='Predicted Value')

plt.bar([i+0.25*width for i in indices], true_value, 
        width=0.5*width, color='r', alpha=0.5, label='True Value')

plt.xticks(indices+width/2., 
           ['V{}'.format(i) for i in range(len(prediction))] )

plt.legend()

plt.show()
encoder = Model(data_in,encoded)
enc = encoder.predict(X_test)
np.savez('enc.npz',enc,y_test)
#from sklearn.manifold import TSNE
#tsne = TSNE(verbose=1,n_iter=300)
#res = tsne.fit_transform(enc)
'''
fig = plt.figure(figsize=(10,7))
scatter =plt.scatter(res[:,0],res[:,1],c=y_test,cmap='coolwarm', s=0.6)
scatter.axes.get_xaxis().set_visible(False)
scatter.axes.get_yaxis().set_visible(False)
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
batch_size = 100
original_dim = 29
latent_dim = 6
intermediate_dim = 16
epochs = 50
epsilon_std = 1.0

x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
h_decoded = decoder_h(z)

decoder_mean = Dense(original_dim)
x_decoded_mean = decoder_mean(h_decoded)
# instantiate VAE model
vae = Model(x, x_decoded_mean)
# Compute VAE loss
xent_loss = original_dim * metrics.mean_squared_error(x, x_decoded_mean)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)
vae.add_loss(vae_loss)
from keras.optimizers import RMSprop
vae.compile(optimizer=RMSprop(lr=0.1))
#vae.summary()

vae.fit(X_train,
        shuffle=True,
        epochs=epochs,
        batch_size=256,
        validation_data=(X_test, None))

pred = autoencoder.predict(X_test[0:10])

import matplotlib.pyplot as plt
import numpy as np

width = 0.8

prediction   = pred[1]
true_value    = X_test[1]

indices = np.arange(len(highPower))

fig = plt.figure(figsize=(10,7))

plt.bar(indices, prediction, width=width, 
        color='b', label='Predicted Value')

plt.bar([i+0.25*width for i in indices], true_value, 
        width=0.5*width, color='r', alpha=0.5, label='True Value')

plt.xticks(indices+width/2., 
           ['T{}'.format(i) for i in range(len(pred))] )

plt.legend()

plt.show()
frauds = np.where(y_train == 1)
encoder = Model(x,z_mean)
fraud_encodings = encoder.predict(X_train[frauds],batch_size=128)
fraud_encodings.shape
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)
more_frauds = generator.predict(fraud_encodings)
