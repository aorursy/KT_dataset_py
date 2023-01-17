#Required Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os # library for file and folder traversal

from keras.datasets import imdb # keras.datasets library has a list of popular data

from keras import models # this library calls model

from keras import layers # required for building dense layers

from keras import optimizers # define Dense layer optimizers

import matplotlib.pyplot as plt # required for ploting loss vs epochs



from keras.utils import to_categorical # library to encode labels
print(os.listdir("../input/imdb-data/"))
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
#train_data

#test_data

#train_labels

test_labels
word_index = imdb.get_word_index() # word_index is a dictionary mapping words to an integer index

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) # reverse it, mapping integer indices to words

decoded_review = ' '.join([reverse_word_index.get(i -3, '?') for i in train_data[2]]) # decode the review, first 3 indices are not relevent
decoded_review
def vectorize_sequences(sequences, dimension = 10000): 

    # sequence arguement denotes number of review i.e. number of rows in dataset

    # dimension argument denotes number of columns

    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):

        results[i, sequence] = 1

    return results # Returns, a tensor of shape (len(sequence), dimension)
x_train = vectorize_sequences(train_data) # 

x_test = vectorize_sequences(test_data)
x_train
x_test
y_train = np.asarray(train_labels).astype('float32')

y_test = np.asarray(test_labels).astype('float32')
y_train
y_test
print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
model = models.Sequential()

model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))

model.add(layers.Dense(16, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer= optimizers.RMSprop(lr=0.001),

             loss='binary_crossentropy',

             metrics=['accuracy'])
history = model.fit(x_train,

                   y_train,

                   epochs=4,

                   batch_size=512,

                   validation_data=(x_test, y_test))
history_dict = history.history

history_dict.keys()
history_dict = history.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']



epochs = range(1, len(loss_values) + 1)



plt.plot(epochs, loss_values, 'bo', label='Training loss')

plt.plot(epochs, val_loss_values, 'b', label='Validation loss') 

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
plt.clf()

acc = history_dict['accuracy']

val_acc = history_dict['val_accuracy']



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.show()
results = model.evaluate(x_test, y_test)
results
model = models.Sequential()

model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))

model.add(layers.Dense(16, activation='relu'))

model.add(layers.Dense(16, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer= optimizers.RMSprop(lr=0.001),

             loss='binary_crossentropy',

             metrics=['accuracy'])
history = model.fit(x_train,

                   y_train,

                   epochs=4,

                   batch_size=512,

                   validation_data=(x_test, y_test))
history_dict = history.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']



epochs = range(1, len(loss_values) + 1)



plt.plot(epochs, loss_values, 'bo', label='Training loss')

plt.plot(epochs, val_loss_values, 'b', label='Validation loss') 

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
plt.clf()

acc = history_dict['accuracy']

val_acc = history_dict['val_accuracy']



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.show()
results = model.evaluate(x_test, y_test)

results
model = models.Sequential()

model.add(layers.Dense(32, activation='relu', input_shape=(10000,)))

model.add(layers.Dense(32, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer= optimizers.RMSprop(lr=0.001),

             loss='binary_crossentropy',

             metrics=['accuracy'])



history = model.fit(x_train,

                   y_train,

                   epochs=4,

                   batch_size=512,

                   validation_data=(x_test, y_test))
history_dict = history.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']



epochs = range(1, len(loss_values) + 1)



plt.plot(epochs, loss_values, 'bo', label='Training loss')

plt.plot(epochs, val_loss_values, 'b', label='Validation loss') 

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
plt.clf()

acc = history_dict['accuracy']

val_acc = history_dict['val_accuracy']



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.show()
results = model.evaluate(x_test, y_test)

results
model = models.Sequential()

model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.summary()
model.compile(optimizer= optimizers.RMSprop(lr=0.001),

             loss='binary_crossentropy',

             metrics=['accuracy'])



history = model.fit(x_train,

                   y_train,

                   epochs=4,

                   batch_size=512,

                   validation_data=(x_test, y_test))
history_dict = history.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']



epochs = range(1, len(loss_values) + 1)



plt.plot(epochs, loss_values, 'bo', label='Training loss')

plt.plot(epochs, val_loss_values, 'b', label='Validation loss') 

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
plt.clf()

acc = history_dict['accuracy']

val_acc = history_dict['val_accuracy']



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.show()
results = model.evaluate(x_test, y_test)

results
model = models.Sequential()

model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))

model.add(layers.Dense(16, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.summary()
model.compile(optimizer= optimizers.RMSprop(lr=0.001),

             loss='MSE',

             metrics=['accuracy'])



history = model.fit(x_train,

                   y_train,

                   epochs=4,

                   batch_size=512,

                   validation_data=(x_test, y_test))
history_dict = history.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']



epochs = range(1, len(loss_values) + 1)



plt.plot(epochs, loss_values, 'bo', label='Training loss')

plt.plot(epochs, val_loss_values, 'b', label='Validation loss') 

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
plt.clf()

acc = history_dict['accuracy']

val_acc = history_dict['val_accuracy']



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.show()
results = model.evaluate(x_test, y_test)

results
model = models.Sequential()

model.add(layers.Dense(16, activation='tanh', input_shape=(10000,)))

model.add(layers.Dense(16, activation='tanh'))

model.add(layers.Dense(1, activation='sigmoid'))



model.summary()
model.compile(optimizer= optimizers.RMSprop(lr=0.001),

             loss='binary_crossentropy',

             metrics=['accuracy'])



history = model.fit(x_train,

                   y_train,

                   epochs=4,

                   batch_size=512,

                   validation_data=(x_test, y_test))
history_dict = history.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']



epochs = range(1, len(loss_values) + 1)



plt.plot(epochs, loss_values, 'bo', label='Training loss')

plt.plot(epochs, val_loss_values, 'b', label='Validation loss') 

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
plt.clf()

acc = history_dict['accuracy']

val_acc = history_dict['val_accuracy']



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.show()
results = model.evaluate(x_test, y_test)

results
from keras.layers import LeakyReLU
model = models.Sequential()

model.add(layers.Dense(16, input_shape=(10000,)))

model.add(LeakyReLU(alpha=0.05))

model.add(layers.Dense(16))

model.add(LeakyReLU(alpha=0.05))

model.add(layers.Dense(1, activation='sigmoid'))



model.summary()
model.compile(optimizer= optimizers.RMSprop(lr=0.001),

             loss='binary_crossentropy',

             metrics=['accuracy'])



history = model.fit(x_train,

                   y_train,

                   epochs=4,

                   batch_size=512,

                   validation_data=(x_test, y_test))
history_dict = history.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']



epochs = range(1, len(loss_values) + 1)



plt.plot(epochs, loss_values, 'bo', label='Training loss')

plt.plot(epochs, val_loss_values, 'b', label='Validation loss') 

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
plt.clf()

acc = history_dict['accuracy']

val_acc = history_dict['val_accuracy']



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.show()
results = model.evaluate(x_test, y_test)

results
from keras.datasets import reuters



(train_data, train_labels),(test_data, test_labels)= reuters.load_data(num_words=10000)
print(len(train_data))

print(len(test_data))
print(train_data[2])
print(test_data[2])
word_index = reuters.get_word_index()



reverse_word_index = dict([(value,key) for (key, value) in word_index.items()])

decoded_newswire = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])
print(decoded_newswire)
decoded_newswire = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[1]])
print(decoded_newswire)
def vectorize_sequences(sequences, dimension=10000):

    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):

        results[i, sequence]=1

    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
x_test
x_train
def to_one_hot(labels, dimension=46):

    results = np.zeros((len(labels), dimension))

    for i, label in enumerate(labels):

        results[i, label] =1

    return results
one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)
print(one_hot_test_labels)
print(one_hot_train_labels)
# using keras.utils to import to_categorical



one_hot_train_labels= to_categorical(train_labels)

one_hot_test_labels = to_categorical(test_labels)
print(one_hot_train_labels)
print(one_hot_test_labels)
model = models.Sequential()

model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(46, activation='softmax'))
model.summary()
model.compile(optimizer='rmsprop',

             loss = 'categorical_crossentropy',

             metrics=['accuracy'])
history = model.fit(x_train,

                   one_hot_train_labels,

                    epochs = 20,

                    batch_size=16,

                    validation_data=(x_test, one_hot_test_labels)

                   )
loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(loss) + 1)



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
plt.clf()                     



acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.show()
model = models.Sequential()

model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop',

             loss = 'categorical_crossentropy',

             metrics=['accuracy'])
history = model.fit(x_train,

                   one_hot_train_labels,

                    epochs = 9,

                    batch_size=16,

                    validation_data=(x_test, one_hot_test_labels)

                   )
results= model.evaluate(x_test, one_hot_test_labels)
print(results)
predictions = model.predict(x_test)
predictions[0]
np.argmax(predictions[50])
model = models.Sequential()

model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))

model.add(layers.Dense(4, activation='relu'))

model.add(layers.Dense(46, activation='softmax'))
model.summary()
model.compile(optimizer='rmsprop',

             loss='categorical_crossentropy',

             metrics=['accuracy'])
model.fit(x_train,

         one_hot_train_labels,

         epochs=9,

          batch_size=16,

         validation_data=(x_test, one_hot_test_labels))
loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(loss) + 1)



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
plt.clf()                     



acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.show()
evaluations= model.evaluate(x_test, one_hot_test_labels)
print(evaluations)
model = models.Sequential()

model.add(layers.Dense(32, activation='relu', input_shape=(10000,)))

model.add(layers.Dense(32, activation='relu'))

model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop',

             loss='categorical_crossentropy',

             metrics=['accuracy'])
model.fit(x_train,

         one_hot_train_labels,

         epochs=20,

          batch_size=16,

         validation_data=(x_test, one_hot_test_labels))
loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(loss) + 1)



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
plt.clf()                     



acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.show()
evaluations= model.evaluate(x_test, one_hot_test_labels)

print(evaluations)
model = models.Sequential()

model.add(layers.Dense(128, activation='relu', input_shape=(10000,)))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop',

             loss='categorical_crossentropy',

             metrics=['accuracy'])



model.fit(x_train,

         one_hot_train_labels,

         epochs=20,

          batch_size=16,

         validation_data=(x_test, one_hot_test_labels))
loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(loss) + 1)



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
plt.clf()                     



acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.show()
evaluations= model.evaluate(x_test, one_hot_test_labels)

print(evaluations)
model = models.Sequential()

model.add(layers.Dense(256, activation='relu', input_shape=(10000,)))

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop',

             loss='categorical_crossentropy',

             metrics=['accuracy'])



model.fit(x_train,

         one_hot_train_labels,

         epochs=20,

          batch_size=16,

         validation_data=(x_test, one_hot_test_labels))
loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(loss) + 1)



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
plt.clf()                     



acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.show()
evaluations= model.evaluate(x_test, one_hot_test_labels)

print(evaluations)
model = models.Sequential()

model.add(layers.Dense(256, activation='relu', input_shape=(10000,)))

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop',

             loss='categorical_crossentropy',

             metrics=['accuracy'])



model.fit(x_train,

         one_hot_train_labels,

         epochs=20,

          batch_size=16,

         validation_data=(x_test, one_hot_test_labels))
loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(loss) + 1)



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
plt.clf()                     



acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.show()
evaluations= model.evaluate(x_test, one_hot_test_labels)

print(evaluations)
model = models.Sequential()

model.add(layers.Dense(256, activation='relu', input_shape=(10000,)))

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop',

             loss='categorical_crossentropy',

             metrics=['accuracy'])



model.fit(x_train,

         one_hot_train_labels,

         epochs=20,

          batch_size=16,

         validation_data=(x_test, one_hot_test_labels))
loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(loss) + 1)



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
plt.clf()                     



acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.show()
evaluations= model.evaluate(x_test, one_hot_test_labels)

print(evaluations)
model = models.Sequential()

model.add(layers.Dense(256, activation='relu', input_shape=(10000,)))

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop',

             loss='categorical_crossentropy',

             metrics=['accuracy'])



model.fit(x_train,

         one_hot_train_labels,

         epochs=20,

          batch_size=16,

         validation_data=(x_test, one_hot_test_labels))
loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(loss) + 1)



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
plt.clf()                     



acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.show()
evaluations= model.evaluate(x_test, one_hot_test_labels)

print(evaluations)