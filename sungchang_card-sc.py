import numpy as np

import pandas as pd

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, cohen_kappa_score

from sklearn.metrics import f1_score, recall_score

import sys

import os

import pprint

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/creditcardfraud/creditcard.csv',sep=',')

print(data.columns)
#data = data[data.Class != 1]

#Class = data1['Class']

#data = data.drop(['Class'], axis = 1)

#data = data.drop(['Time'], axis = 1)
correlation_matrix = data.corr()

fig = plt.figure(figsize=(12,9))

sns.heatmap(correlation_matrix,vmax=0.3, vmin=-0.3,linewidths=1)

plt.show()
data = data.drop(['V1'], axis = 1)

data = data.drop(['V2'], axis = 1)

data = data.drop(['V3'], axis = 1)

data = data.drop(['V4'], axis = 1)

data = data.drop(['V5'], axis = 1)

data = data.drop(['V6'], axis = 1)

data = data.drop(['V7'], axis = 1)

data = data.drop(['V8'], axis = 1)

data = data.drop(['V9'], axis = 1)

#data = data.drop(['V10'], axis = 1)

#data = data.drop(['V11'], axis = 1)

#data = data.drop(['V12'], axis = 1)

data = data.drop(['V13'], axis = 1)

#data = data.drop(['V14'], axis = 1)

data = data.drop(['V15'], axis = 1)

#data = data.drop(['V16'], axis = 1)

#data = data.drop(['V17'], axis = 1)

#data = data.drop(['V18'], axis = 1)

data = data.drop(['V19'], axis = 1)

data = data.drop(['V20'], axis = 1)

data = data.drop(['V21'], axis = 1)

data = data.drop(['V22'], axis = 1)

data = data.drop(['V23'], axis = 1)

data = data.drop(['V24'], axis = 1)

data = data.drop(['V25'], axis = 1)

data = data.drop(['V26'], axis = 1)

data = data.drop(['V27'], axis = 1)

data = data.drop(['V28'], axis = 1)

data = data.drop(['Time'], axis = 1)

data = data.drop(['Amount'], axis = 1)
train_x, test_x = train_test_split(data,test_size = 0.3,random_state=1)

train_x = train_x[train_x.Class == 0] 

train_x = train_x.drop(['Class'], axis=1) 



test_y = test_x['Class']

test_x = test_x.drop(['Class'], axis=1)
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

#data['Time'] = StandardScaler().fit_transform(data['Time'].values.reshape(-1, 1))

#data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

train_x = sc.fit_transform(train_x)

test_x = sc.transform(test_x)
from keras.models import Model, load_model

from keras.layers import Input, Dense

from keras.callbacks import ModelCheckpoint, TensorBoard

from keras import regularizers

import tensorflow as tf



input_dim = train_x.shape[1]

encoding_dim = int(input_dim * 2) - 1

hidden_dim = int(encoding_dim * 2)

learning_rate = 1e-7



input_layer = Input(shape=(input_dim, ))

encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l2(learning_rate))(input_layer)

encoder = Dense(hidden_dim, activation="relu")(encoder)

decoder = Dense(hidden_dim, activation='tanh')(encoder)

decoder = Dense(input_dim, activation='relu')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
nb_epoch = 30

batch_size = 128
autoencoder.compile(metrics=['accuracy'],

                    loss='mean_squared_error',

                    optimizer='adam')



cp = ModelCheckpoint(filepath="autoencoder_fraud.h5",

                               save_best_only=True,

                               verbose=0)



tb = TensorBoard(log_dir='./logs',

                histogram_freq=0,

                write_graph=True,

                write_images=True)



history = autoencoder.fit(train_x, train_x,

                    epochs=nb_epoch,

                    batch_size=batch_size,

                    shuffle=False,

                    validation_data=(test_x, test_x),

                    verbose=1,

                    callbacks=[cp, tb]).history
autoencoder = load_model('autoencoder_fraud.h5')
plt.plot(history['loss'], linewidth=2, label='Train')

plt.plot(history['val_loss'], linewidth=2, label='Test')

plt.legend(loc='upper right')

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.show()
pred = autoencoder.predict(test_x)
mse = np.mean(np.power(test_x - pred, 2), axis=1)

error_df = pd.DataFrame({'Reconstruction_error': mse,

                        'True_class': test_y})
error_df.head(30)
error_df.Reconstruction_error.values
threshold_fixed = 2.55

pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]

matrix = confusion_matrix(error_df.True_class, pred_y)

print(matrix)
tpos = matrix[0][0]

fneg = matrix[1][1]

fpos = matrix[0][1]

tneg = matrix[1][0]

print((tpos / (fpos + tpos)))

print((fneg / (tneg + fneg)))

print(((tpos / (fpos + tpos)) + (fneg / (tneg + fneg))) / 2)