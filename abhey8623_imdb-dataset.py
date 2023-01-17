from keras.datasets import imdb

import keras

import numpy as np

keras.__version__
type(imdb)
print(dir(imdb))
# save np.load

np_load_old = np.load



# modify the default parameters of np.load

np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# call load_data with allow_pickle implicitly set to true

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)



# restore np.load for future normal usage

np.load = np_load_old
print(train_data[0])
word_to_ind = imdb.get_word_index()
ind_to_word = dict([(value, key) for key,value in word_to_ind.items()])
print("The {}th and {}th word in the vocabulary are \{}/ and \{}/ respectively.".format(16,22,ind_to_word[16],ind_to_word[22]))

print("The words \happy/ and \sad/ in the vocabulary have {}th and {}th index respectively".format(word_to_ind['happy'],word_to_ind['sad']))
def vectorize_sequences(sequences, dimension=10000):

	results = np.zeros((len(sequences), dimension))

	for i, sequence in enumerate(sequences):

		results[i, sequence] = 1.

	return results
x_train = vectorize_sequences(train_data)

x_test = vectorize_sequences(test_data)



y_train = np.asarray(train_labels).astype('float32')

y_test = np.asarray(test_labels).astype('float32')
from keras.layers import Dense

from keras.models import Sequential
model = Sequential()

model.add(Dense(108, activation = 'relu', input_shape = [10000,]))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))
model.summary()
class my_callback(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs = {}):

        if(logs.get('acc') > 0.99):

            print("Stopping to prevent overfitting")

            self.model.stop_training = True
callback = my_callback()

model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 20, callbacks = [callback])
model.evaluate(x_test, y_test)
from keras.layers import Input

from keras.models import Model
X = Input(shape = (10000,))

Y = Dense(108, activation = 'relu')(X)

Y = Dense(10, activation = 'relu')(Y)

Y = Dense(1, activation = 'sigmoid')(Y)
model = Model(inputs = [X], outputs = [Y])

model.summary()
model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 15, callbacks = [callback])
model.evaluate(x_test, y_test)