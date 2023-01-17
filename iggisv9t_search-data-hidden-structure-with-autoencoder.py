import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.preprocessing import StandardScaler



from keras.models import Sequential, Model

from keras.layers import Dense, BatchNormalization, Dropout, Flatten, Input

from keras import backend as K

import keras

from matplotlib.colors import LogNorm
folder_path = '/kaggle/input/ieee-fraud-detection/'
train = pd.read_csv(f'{folder_path}train_transaction.csv')

test = pd.read_csv(f'{folder_path}test_transaction.csv')
cats = ['ProductCD',

    'card1',

    'card2',

    'card3',

    'card4',

    'card5',

    'card6',

    'P_emaildomain',

    'R_emaildomain',

    'M1',

    'M2',

    'M3',

    'M4',

    'M5',

    'M6',

    'M7',

    'M8',

    'M9',

    'addr1',

    'addr2']



cols = list(train.columns)[3:]

nocats = [c for c in cols if (not c in cats)]
%%time

ss = StandardScaler(copy=False)

data_ss = ss.fit_transform(np.nan_to_num(train[nocats].values))
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
data_ss = np.clip(data_ss, -10, 10)
%%time

epochs = 50

batch_size = 9548

history = c_autoencoder.fit(data_ss, data_ss,

                    epochs=epochs,

                    batch_size=batch_size,

                    shuffle=True,

                    verbose=1)



loss_history = history.history['loss']

plt.figure(figsize=(10, 5))

plt.plot(loss_history);
%%time

emb = c_encoder.predict(data_ss)
plt.figure(figsize=(10, 10))

plt.hist2d(emb[:, 0], emb[:, 1], bins=256, norm=LogNorm());
plt.figure(figsize=(20, 20))

plt.scatter(emb[:, 0], emb[:, 1], c=train['isFraud'].values,

           marker='.', alpha=0.1);
plt.figure(figsize=(20, 20))

plt.scatter(emb[:, 0], emb[:, 1], c=train['TransactionDT'].values,

           marker='.', alpha=0.1, cmap='jet');
prd_d = {p: i for i, p in enumerate(train['ProductCD'].unique())}
plt.figure(figsize=(20, 20))

plt.scatter(emb[:, 0], emb[:, 1],

            c=train['ProductCD'].apply(lambda x: prd_d[x]).values,

           marker='.', alpha=0.1, cmap='jet');
%%time

test_ss = ss.transform(np.nan_to_num(test[nocats].values))
%%time

test_emb = c_encoder.predict(test_ss)
plt.figure(figsize=(20, 20))

plt.scatter(test_emb[:, 0], test_emb[:, 1],

            c=test['TransactionDT'].values,

           marker='.', alpha=0.1, cmap='jet');