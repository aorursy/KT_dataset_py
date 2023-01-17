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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

import keras

from keras.models import Sequential

from keras.layers import Dense, MaxPool2D,MaxPooling2D, Conv2D, Flatten, Dropout, BatchNormalization

from keras.losses import categorical_crossentropy

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
train = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")

test = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

validation = pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv")
train.head()
train_dataset = np.loadtxt('/kaggle/input/Kannada-MNIST/train.csv', skiprows=1, delimiter=',')
train_dataset[0:5]
x_train = train_dataset[:, 1:]

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

input_shape = (28, 28, 1)
x_train /= 255.0
x_train[1].shape
y_train = train_dataset[:, 0]
y_train[:5]
from tensorflow.keras import utils

y_train = utils.to_categorical(y_train)
y_train
random_seed = 2

X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=random_seed)
X_train.shape
datagen = ImageDataGenerator(

        rotation_range=10,  

        zoom_range = 0.10,  

        width_shift_range=0.1, 

        height_shift_range=0.1)
i = 0

data = X_train[0]

data = np.expand_dims(data, axis=0)

for batch in datagen.flow(data, batch_size=1):

    plt.figure(i)

    imgplot = plt.imshow(batch[0][:,:,0])

    i += 1

    if i % 6 == 0:

        break

plt.show()
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print(model.summary())
batch_size=96

сheckpoint = ModelCheckpoint('mnist-cnn.h5', 

                              monitor='val_acc', 

                              save_best_only=True,

                              verbose=1)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
history = model.fit(datagen.flow(X_train,Y_train, batch_size=batch_size), 

                    epochs=30,

                    validation_data=(X_val, Y_val),

                    steps_per_epoch=X_train.shape[0] // batch_size,

                    verbose=1)
plt.plot(history.history['accuracy'], 

         label='Accuracy on Training Data ')

plt.plot(history.history['val_accuracy'], 

         label='Accuracy on Test Data')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()

plt.show()
test_dataset = np.loadtxt('/kaggle/input/Kannada-MNIST/test.csv', skiprows=1, delimiter=",")
test_dataset[:5]


x_test = test_dataset[:, 1:]

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test / 255.0
predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis=1)
predictions[:5]
out = np.column_stack((range(1, predictions.shape[0]+1), predictions))
sub = np.savetxt('submission.csv', out, header="id,label", 

            comments="", fmt="%d,%d")