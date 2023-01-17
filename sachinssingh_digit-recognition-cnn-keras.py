# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline

import tensorflow as tf

np.random.seed(2)



from sklearn.model_selection import  train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train.shape
test.shape
X_train = train.drop('label', axis=1)

y_train = train['label']

X_test = test
X_train=X_train/255

X_test=X_test/255
y_train
y_train=to_categorical(y_train, num_classes = 10)
X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)
X_train.shape,y_train.shape
x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 2)
model = Sequential()



model.add(Conv2D(filters = 20, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters = 40, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters = 150, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Flatten())

model.add(Dense(10, activation = "softmax"))
model.compile(optimizer = 'rmsprop' , loss = "categorical_crossentropy", metrics=["accuracy"])
history = model.fit(x_train, y_train,

                    epochs=20,

                    batch_size = 128,

                    validation_data = (x_val, y_val))
loss = history.history['loss']

val_loss = history.history['val_loss']

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

epochs = range(1, 21)



plt.plot(epochs, loss, 'ko', label = 'Training Loss')

plt.plot(epochs, val_loss, 'k', label = 'Validation Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('Training and Validation Loss')

plt.legend()
plt.plot(epochs, acc, 'yo', label = 'Training Accuracy')

plt.plot(epochs, val_acc, 'y', label = 'Validation Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.title('Training and Validation Accuracy')

plt.legend()
# predict results

results = model.predict(X_test)
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
results
submission = pd.concat([pd.Series(range(1, 28001), name = 'ImageId'), results], axis = 1)

submission.to_csv("MNIST_Dataset_Submissions.csv", index = False)