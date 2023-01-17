# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from keras.datasets import fashion_mnist

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.utils import np_utils

# Any results you write to the current directory are saved as output.
A = pd.read_csv("../input/fashion-mnist_test.csv")

B = pd.read_csv("../input/fashion-mnist_train.csv")
X_train = B.drop(columns=['label'], axis=1).values.reshape(60000, 28, 28)

X_train = X_train/255

y_train = B.label.values

y_train
X_test = A.drop(columns=['label'], axis=1).values.reshape(10000, 28, 28)

X_test = X_test/255

y_test = A.label.values

y_test
import matplotlib.pyplot as plt

plt.imshow(X_train[1,:,:], cmap='Greys')

plt.show
for i in range(9):

    plt.subplot(3,3,i+1)

    plt.imshow(X_train[i], cmap='gray', interpolation='none')

    plt.title("Class {}".format(y_train[i]))
X_train = X_train.reshape(60000, 784)

X_test = X_test.reshape(10000, 784)

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

print("Training matrix shape", X_train.shape)

print("Testing matrix shape", X_test.shape)
nb_classes = 10

Y_train = np_utils.to_categorical(y_train, nb_classes)

Y_test = np_utils.to_categorical(y_test, nb_classes)
model = Sequential()

model.add(Dense(512, input_shape=(784,)))

model.add(Activation('relu')) # An "activation" (non linear function)to the output

# if any value <0 we set it at 0



model.add(Dropout(0.2)) # Dropout = ??? crazy thing I dont understand at all wtf wtf wtf

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(Dense(10))

model.add(Activation('softmax')) # softmax = makes the output positive and ensures the sum is 1 (probability function)
# here we compile the model

model.compile(loss='categorical_crossentropy', optimizer='adam')
#here we train the model

model.fit(X_train, Y_train,

batch_size=256, epochs=10, verbose=1, #we can put any number of epochs, YOLO

validation_data=(X_test, Y_test))
#Now we test the shit and evaluate it performance

score= model.evaluate(X_test, Y_test, verbose=0)

print ('Test loss:', score)

# The predict_classes function outputs the highest probability class

# according to the trained classifier for each input example.

predicted_classes = model.predict_classes(X_test)



# Check which items we got right / wrong

correct_indices = np.nonzero(predicted_classes == y_test)[0]

incorrect_indices = np.nonzero(predicted_classes != y_test)[0]




name = {0: 'T-shirt/top',

1: 'Trouser',

2 :'Pullover',

3 :'Dress',

4 :'Coat',

5 :'Sandal',

6 :'Shirt',

7 :'Sneaker',

8 :'Bag',

9 :'Ankle boot'}
plt.figure()

for i, correct in enumerate(correct_indices[:9]):

    plt.subplot(3,3,i+1)

    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Predicted {}, Class {}".format(name[predicted_classes[correct]],name[ y_test[correct]]))



plt.figure()

for i, incorrect in enumerate(incorrect_indices[:9]):

    plt.subplot(3,3,i+1)

    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Predicted {}, Class {}".format(name[predicted_classes[incorrect]], name[y_test[incorrect]]))