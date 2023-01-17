from keras.datasets import mnist

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from keras.utils import np_utils



%matplotlib inline
# use Keras to import pre-shuffled MNIST database

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = np.vstack((X_train, X_test))

y_train = np.concatenate([y_train, y_test])

X_train = X_train.reshape(-1, 28, 28, 1)

print(X_train.shape, y_train.shape)
train = pd.read_csv('../input/train.csv').values

y_val = train[:,0].astype('int32')

X_val = train[:,1:].astype('float32')

X_val = X_val.reshape(-1,28,28,1)

print(X_val.shape, y_val.shape)
X_test = pd.read_csv('../input/test.csv').values.astype('float32')

X_test = X_test.reshape(-1, 28, 28, 1)
# plot first six training images

fig = plt.figure(figsize=(20,20))

for i in range(6):

    ax = fig.add_subplot(1, 6, i+1, xticks=[], yticks=[])

    ax.imshow(X_train[i].reshape(28,28), cmap='gray')

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

visualize_input(X_train[0].reshape(28,28), ax)
X_train = X_train.astype('float32')/255

X_val = X_val.astype('float32')/255

X_test = X_test.astype('float32')/255 
# print first ten (integer-valued) training labels

print('Integer-valued labels:')

print(y_train[:10])



# one-hot encode the labels

y_train = np_utils.to_categorical(y_train, 10)

y_val = np_utils.to_categorical(y_val, 10)



# print first ten (one-hot) training labels

print('One-hot labels:')

print(y_train[:10])
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D



# define the model

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))

model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))

model.add(Conv2D(filters=192, kernel_size=3, padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=192, kernel_size=5, padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=2, padding='same'))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dense(10, activation='softmax'))

# summarize the model

model.summary()
# compile the model

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint   

from keras.callbacks import ReduceLROnPlateau



# train the model

#checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', 

#                               verbose=1, save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, verbose=1,

                              patience=2, min_lr=0.00000001)

hist = model.fit(X_train, y_train, batch_size=100, epochs=25,

          validation_data=(X_val, y_val), callbacks=[reduce_lr],

          verbose=1, shuffle=True)
testY = model.predict_classes(X_test, verbose=2)
model.save_weights("mnist.h5")
sub = pd.read_csv('../input/sample_submission.csv')

sub['Label'] = testY

sub.to_csv('submission.csv',index=False)