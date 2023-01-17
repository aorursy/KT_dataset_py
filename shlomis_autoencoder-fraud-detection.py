# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler

from keras.layers import Input, Dense

from keras.models import Model

import matplotlib.pyplot as plt



data = pd.read_csv('../input/creditcard.csv')

data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

data = data.drop(['Time','Amount'],axis=1)



data = data[data.Class != 1]

X = data.loc[:, data.columns != 'Class']



encodingDim = 8

inputShape = X.shape[1]

inputData = Input(shape=(inputShape,))



X = X.as_matrix()



encoded = Dense(encodingDim, activation='relu')(inputData)

encoded = Dense(int(encodingDim/2), activation='relu')(encoded)

encoded = Dense(int(encodingDim/4), activation='relu')(encoded)



decoded = Dense(int(encodingDim/4), activation='relu')(encoded)

decoded = Dense(int(encodingDim/2), activation='relu')(decoded)

decoded = Dense(inputShape, activation='sigmoid')(decoded)



autoencoder = Model(inputData, decoded)

encoder = Model(inputData, encoded)

encodedInput = Input(shape=(encodingDim,))

decoderLayer = autoencoder.layers[-1]

decoder = Model(encodedInput, decoderLayer(encodedInput))



autoencoder.summary()



autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

history = autoencoder.fit(X, X,

                epochs=50,

                batch_size=10,

                validation_split=0.33)



print(history.history.keys())

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()








