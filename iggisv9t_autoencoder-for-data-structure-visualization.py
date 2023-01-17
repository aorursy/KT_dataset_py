import os

print(os.listdir("../input"))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from tqdm import tqdm
train = pd.read_csv('../input/train.csv')

train.head()
cols = list(train.columns)[2:]
from sklearn.preprocessing import StandardScaler



from keras.models import Sequential, Model

from keras.layers import Dense, BatchNormalization, Dropout, Flatten, Input

from keras import backend as K

import keras

from matplotlib.colors import LogNorm
%%time

ss = StandardScaler(copy=False)

data_ss = ss.fit_transform(np.nan_to_num(train[cols].apply(lambda x: round(x, 2)).values))
n_features = data_ss.shape[1]



dim = 15



def build_model(dropout_rate=0.15, activation='tanh'):

    main_input = Input(shape=(n_features, ), name='main_input')

    

    x = Dense(dim*2, activation=activation)(main_input)

    x = BatchNormalization()(x)

    x = Dropout(dropout_rate)(x)

    

    x = Dense(dim*2, activation=activation)(x)

    x = BatchNormalization()(x)

    x = Dropout(dropout_rate/2)(x)

    

    x = Dense(dim, activation=activation)(x)

    x = Dropout(dropout_rate/4)(x)



    encoded = Dense(2, activation='tanh')(x)



    input_encoded = Input(shape=(2, ))

    

    x = Dense(dim, activation=activation)(input_encoded)

    x = Dense(dim, activation=activation)(x)

    x = Dense(dim*2, activation=activation)(x)

    

    decoded = x = Dense(n_features, activation='linear')(x)



    encoder = Model(main_input, encoded, name="encoder")

    decoder = Model(input_encoded, decoded, name="decoder")

    autoencoder = Model(main_input, decoder(encoder(main_input)), name="autoencoder")

    return encoder, decoder, autoencoder



K.clear_session()

c_encoder, c_decoder, c_autoencoder = build_model()

c_autoencoder.compile(optimizer='nadam', loss='mse')



c_autoencoder.summary()
%%time

loss_history = []

for i in tqdm(range(20)):

    epochs = 20

    batch_size = 2048

    history = c_autoencoder.fit(data_ss + np.random.normal(scale=0.01, size=data_ss.shape), data_ss,

                        epochs=epochs,

                        batch_size=batch_size,

                            shuffle=True,

                            verbose=0)

    

    loss_history += history.history['loss']

plt.figure(figsize=(10, 5))

plt.plot(loss_history);
%%time

emb = c_encoder.predict(data_ss)
plt.figure(figsize=(10, 10))

plt.hist2d(emb[:, 0], emb[:, 1], bins=256, norm=LogNorm());
plt.figure(figsize=(10, 10))

plt.scatter(emb[:, 0], emb[:, 1], marker='.', c=train['target'].values, alpha=0.1);