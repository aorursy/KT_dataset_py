# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Function given in the mnist dataset on kaggle
def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)

# use Keras to import pre-shuffled MNIST database
(X_train, y_train), (X_test, y_test) = load_data('../input/mnist.npz')

print("The MNIST database has a training set of %d examples." % len(X_train))
print("The MNIST database has a test set of %d examples." % len(X_test))
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.cm as cm

# plot first 5 training images
fig = plt.figure(figsize=(20,20))
for i in range(5):
    ax = fig.add_subplot(1, 6, i+1, xticks=[], yticks=[])
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(str(y_train[i]))
def detailed_image(img, ax):
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y], 2)), xy=(y,x),
                        horizontalalignment = 'center',
                        verticalalignment = 'center',
                       color = 'white' if img[x][y]<thresh else 'black')
            
fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)
detailed_image(X_train[0], ax)            
# rescale [0,255] --> [0,1]
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255 
from keras.utils import np_utils

#First 10 integer valued training labels
print('Integer values training labels: ')
print(y_train[:10])

#one-hot encode labels
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

#print ffirst 10 encoded values
print('One-hot Labels: ')
print(y_train[:10])
X_train.shape
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

#Defining the model
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

#summarize the model
model.summary()
#Compile the model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#evaluate score accuracy
score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]

#print test accuracy
print('Test accruracy: %.4f%%' %accuracy)
from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5',
                            verbose = 1, save_best_only = True)
hist = model.fit(X_train, y_train, batch_size=78, epochs=10, 
                 validation_split=0.2, callbacks=[checkpointer], verbose=1, shuffle=True)

# load the weights that yielded the best validation accuracy
model.load_weights('mnist.model.best.hdf5')
#evaluate score accuracy
score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]

#print test accuracy
print('Test accruracy: %.4f%%' %accuracy)