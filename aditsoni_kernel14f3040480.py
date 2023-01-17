from keras.datasets import mnist



# use Keras to import pre-shuffled MNIST database

(X_train, y_train), (X_test, y_test) = mnist.load_data()



print("The MNIST database has a training set of %d examples." % len(X_train))

print("The MNIST database has a test set of %d examples." % len(X_test))
import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.cm as cm

import numpy as np



# plot first six training images

fig = plt.figure(figsize=(20,20))

for i in range(6):

    ax = fig.add_subplot(1, 6, i+1, xticks=[], yticks=[])

    ax.imshow(X_train[i], cmap='gray')

    ax.set_title(str(y_train[i]))
def visualize_input(img, ax):

    ax.imshow(img, cmap='gray')

    width, height = img.shape

    thresh = img.max()/2.5

    for x in range(width):

        for y in range(height):

            ax.annotate(str(round(img[x][y],2)), xy=(y,x),

                        horizontalalignment='center',

                        verticalalignment='center',

                        color='white' if img[x][y]<thresh else 'black')



fig = plt.figure(figsize = (12,12)) 

ax = fig.add_subplot(111)

visualize_input(X_train[0], ax)
# rescale [0,255] --> [0,1]

X_train = X_train.astype('float32')/255

X_test = X_test.astype('float32')/255 
from keras.utils import np_utils



# print first ten (integer-valued) training labels

print('Integer-valued labels:')

print(y_train[:10])



# one-hot encode the labels

y_train = np_utils.to_categorical(y_train, 10)

y_test = np_utils.to_categorical(y_test, 10)



# print first ten (one-hot) training labels

print('One-hot labels:')

print(y_train[:10])
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten



# define the model

model = Sequential()

model.add(Flatten(input_shape=X_train.shape[1:]))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))



# summarize the model

model.summary()
# compile the model

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 

              metrics=['accuracy'])
# evaluate test accuracy

score = model.evaluate(X_test, y_test, verbose=0)

accuracy = 100*score[1]



# print test accuracy

print('Test accuracy: %.4f%%' % accuracy)
from keras.callbacks import ModelCheckpoint   



# train the model

checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', 

                               verbose=1, save_best_only=True)

hist = model.fit(X_train, y_train, batch_size=128, epochs=10,

          validation_split=0.2, callbacks=[checkpointer],

          verbose=1, shuffle=True)
# load the weights that yielded the best validation accuracy

model.load_weights('mnist.model.best.hdf5')
# evaluate test accuracy

score = model.evaluate(X_test, y_test, verbose=0)

accuracy = 100*score[1]



# print test accuracy

print('Test accuracy: %.4f%%' % accuracy)