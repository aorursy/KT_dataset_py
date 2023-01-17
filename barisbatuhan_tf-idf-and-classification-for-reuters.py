import pandas as pd

import numpy as np
from keras.datasets import reuters

(x_train, y_train), (x_test, y_test) = reuters.load_data(path="reuters.npz", num_words=None, 

                                                         skip_top=0,maxlen=None, test_split=0.2, 

                                                         seed=113, start_char=1,oov_char=2, 

                                                         index_from=3)
from keras import utils

from keras.preprocessing.text import Tokenizer



t = Tokenizer(num_words=10000)

seq = np.concatenate((x_train, x_test), axis=0)

t.fit_on_sequences(seq)



xt_train = t.sequences_to_matrix(x_train, mode='tfidf')

xt_test = t.sequences_to_matrix(x_test, mode='tfidf')



yt_train = utils.to_categorical(y_train, max(y_train) + 1)

yt_test = utils.to_categorical(y_test, max(y_train) + 1)
from keras import layers, models, callbacks

from keras.layers.core import Dense, Dropout # Dropout is being used for not to overfit
# 3 elements, 1st is for first layer, 2nd for last layer

activations = ['relu', 'softmax'] 

layer_sizes = [512, 46] # for each layer a size is given

optimizer_method = 'adadelta'

batch_size = 128

epochs = 5

callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, 

                                   patience=2, mode='auto')
model = models.Sequential()

model.add(layers.Dense(layer_sizes[0], input_shape=(10000,), activation=activations[0]))

model.add(Dropout(0.5))

model.add(layers.Dense(layer_sizes[1], activation=activations[1]))

model.compile(optimizer=optimizer_method, loss='cosine_proximity', metrics=['accuracy'])
model.fit(x=xt_train, y=yt_train, batch_size=batch_size, epochs=epochs, verbose=1, 

          shuffle=True, validation_split=0.15, use_multiprocessing=True,

          workers=8, steps_per_epoch=None, callbacks=[callback])
print(model.metrics_names)

model.evaluate(xt_test, yt_test, verbose=0)