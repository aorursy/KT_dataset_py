%pylab inline

import os

import numpy as np

import pandas as pd

from scipy.misc import imread

from sklearn.metrics import accuracy_score



import tensorflow as tf

import keras
# To stop potential randomness

seed = 128

rng = np.random.RandomState(seed)
root_dir = os.path.abspath('..')

data_dir = os.path.join(root_dir, 'input')

#sub_dir = os.path.join(root_dir, 'sub')

# check for existence

os.path.exists(root_dir)

os.path.exists(data_dir)

#os.path.exists(sub_dir)
train = pd.read_csv(os.path.join(data_dir, 'train.csv'))

test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
train_x = train.iloc[0:,1:]

test_x = test.iloc[0:,1:]



split_size = int(train_x.shape[0]*0.7)



train_x, val_x = train_x[:split_size], train_x[split_size:]

train_y, val_y = train.label.ix[1:split_size], train.label.ix[split_size:]

train_y = train_y.reshape((-1, 1))

train_x = train_x.as_matrix()

val_x.shape
# define vars

input_num_units = 784

hidden_num_units = 50

output_num_units = 10



epochs = 5

batch_size = 128



# import keras modules



from keras.models import Sequential

from keras.layers import Dense



# create model

model = Sequential([

  Dense(output_dim=hidden_num_units, input_dim=input_num_units, activation='relu'),

  Dense(output_dim=output_num_units, input_dim=hidden_num_units, activation='softmax'),

])



# compile the model with necessary attributes

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
trained_model = model.fit(train_x, train_y, nb_epoch=epochs, \

                          batch_size=batch_size, validation_data=(val_x, val_y))