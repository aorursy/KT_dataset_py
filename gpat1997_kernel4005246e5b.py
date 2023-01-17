# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model

from keras.layers import MaxPooling2D, Conv2D, Flatten, Dropout, Dense

from keras.utils import to_categorical

from keras import optimizers

%matplotlib inline
dftrain = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_train.csv')
dftrain.head()
train = np.array(dftrain)
train.shape
train = train[:,1:]
train.shape
train = train/255
plt.imshow(np.reshape(train[1,:],(28,28)),'gray')

plt.show()
tr = np.reshape(train,(27455,28,28,1))
X_train = tr

Y_train = np.array(dftrain['label'])
Y_train = to_categorical(Y_train)
test = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_test.csv')
test.head()
X_test = np.array(test)
Y_test = np.array(test['label'])

Y_test = to_categorical(Y_test)
X_test = X_test[:,1:]
X_test = X_test/255
X_test = np.reshape(X_test,(7172,28,28,1))
print(X_train.shape,'\t',Y_train.shape)

print(X_test.shape,'\t',Y_test.shape)
# Fitting CNN



model = Sequential()

model.add(Conv2D(8, kernel_size=5, activation='relu', input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(16, kernel_size=3, activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())

model.add(Dense(25, activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, Y_train,validation_split = 0.33, epochs = 7, verbose=1)
model.evaluate(X_test,Y_test)
plt.plot(history.history['accuracy'])

plt.plot(history.history['loss'])

plt.plot(history.history['val_accuracy'])

plt.plot(history.history['val_loss'])

plt.xlabel('Epochs')

plt.ylabel('Values for Accuracy and Loss')

plt.legend(['Training Accuracy','Training Loss','Validation Accuracy','Validation Loss'])
model.summary()