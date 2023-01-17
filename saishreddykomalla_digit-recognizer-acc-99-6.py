import tensorflow as tf
tf.version
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
path = "/kaggle/input/digit-recognizer/"

train = pd.read_csv(path + "train.csv")

print(train.shape)
X_train = train.drop(["label"], axis =1).values.astype('float32')

Y_train = train["label"].values.astype('int32')

test = pd.read_csv(path + "test.csv")

X_test = test.values.astype('float32')

print("X_train_dim: ", X_train.shape,"Y_train_dim: ", Y_train.shape, "X_test_dim: ", X_test.shape)
X_train.shape
X_train = X_train.reshape((X_train.shape[0], 28, 28)) 

X_test = X_test.reshape((X_test.shape[0], 28, 28)) 

X_train = X_train / 255.0

X_test = X_test / 255.0
## Add one more dimension for color 

X_train = X_train.reshape((X_train.shape[0],28,28,1))

X_test = X_test.reshape((X_test.shape[0],28, 28,1))

print(f"X_train shape: {X_train.shape}")

print(f"X_test shape: {X_test.shape}")
import tensorflow as tf

class mcb(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs = {}):

        if (logs.get('acc') > 0.999):

            print("\nReached a good accuracy\n")

            self.model.stop_training = True

callbacks = mcb()
from tensorflow.keras.utils import to_categorical

Y_train = to_categorical(Y_train)
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout, Dense

from tensorflow.keras.models import Sequential
model2 = Sequential([

    Conv2D(64, (3,3), activation = 'relu', input_shape = (28,28,1), padding = 'same'),

    BatchNormalization(),

    

    Conv2D(64, (3,3), activation= 'relu', padding = 'same'),

    BatchNormalization(),

    MaxPooling2D(pool_size = (2,2)),

    

    Dropout(0.35),

    

    Conv2D(64, (3,3), activation = 'relu'),

    BatchNormalization(),

    

    Conv2D(64 , (3,3), activation = 'relu', padding = 'same'),

    BatchNormalization(),

    MaxPooling2D(pool_size = (2,2), strides = (2,2)),

    

    Dropout(0.35),

    

    Flatten(),

    Dense(256, activation = 'relu'),

    BatchNormalization(),

    

    Dropout(0.25),

    

    Dense(10, activation = 'sigmoid')

])
model2.compile(optimizer= 'adam', loss = 'categorical_crossentropy', metrics=['acc'])

model2.summary()
from tensorflow.keras.utils import plot_model

plot_model(model2, to_file='model2.png', show_shapes=True, show_layer_names=True)

from IPython.display import Image

Image("model2.png")
modelt = model2.fit(X_train, Y_train,epochs =30, callbacks = [callbacks], validation_split = 0.1)
model3 = Sequential([

    Conv2D(64, (3,3), activation = 'relu', input_shape = (28,28,1), padding = 'same'),

    BatchNormalization(),

    

    Conv2D(64, (3,3), activation= 'relu', padding = 'same'),

    BatchNormalization(),

    MaxPooling2D(pool_size = (2,2)),

    

    Dropout(0.35),

    

    Conv2D(64, (3,3), activation = 'relu'),

    BatchNormalization(),

    

    Conv2D(64 , (3,3), activation = 'relu', padding = 'same'),

    BatchNormalization(),

    MaxPooling2D(pool_size = (2,2), strides = (2,2)),

    

    Dropout(0.35),

    

    Flatten(),

    Dense(256, activation = 'relu'),

    BatchNormalization(),

    

    Dropout(0.25),

    

    Dense(10, activation = 'sigmoid')

])
model3.compile(loss = 'categorical_crossentropy', metrics = ['acc'], optimizer = 'adam')

batch_size = 128

steps_per_epoch = X_train.shape[0]//batch_size
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

    rotation_range = 10,

    zoom_range = 0.1,

    width_shift_range = 0.1,

    height_shift_range = 0.1

)

datagen.fit(X_train)
modelt2 = model3.fit(datagen.flow(X_train, Y_train, batch_size = 128), steps_per_epoch = steps_per_epoch, callbacks = [callbacks], epochs = 30)
%matplotlib inline

import matplotlib.pyplot as plt



fig, ax = plt.subplots(2,1)

ax[0].plot(modelt.history['acc'], color = 'b', label = "Training_accuracy")

ax[0].plot(modelt.history['val_acc'], color = 'r', label = "Dev_accuracy")

legend = ax[0].legend(loc = 'best', shadow = True)



ax[1].plot(modelt.history['loss'], color = 'b', label = "Training_loss")

ax[1].plot(modelt.history['val_loss'], color =  'r', label = "Dev_loss")

legend = ax[1].legend(loc = 'best', shadow = True)



fig, ax = plt.subplots(2,1)

ax[0].plot(modelt2.history['acc'], color = 'b', label = "Training accuracy")

ax[0].legend(loc = 'best', shadow = True)

ax[1].plot(modelt2.history['loss'], color = 'r', label = "Training loss")

ax[1].legend(loc = 'best', shadow = True)
y_pred = model3.predict(X_test, verbose = 1)
print(y_pred[0])
predictions=[]

for i in range(len(X_test)):

    a=np.where(y_pred[i] == max(y_pred[i]))

    predictions.append(a[0][0])
len(X_test)
import matplotlib.pyplot as plt

import random

i = random.randint(0,28000)

plt.imshow(X_test[i].reshape(28,28), cmap = plt.get_cmap('gray'))

plt.title(predictions[i])
# model.save(_________)

# model.save_weights('model')
import pandas as pd

counter = range(1, len(predictions) + 1)

solution = pd.DataFrame({"ImageId": counter, "label": list(predictions)})
solution.to_csv("digit_recognizer8.csv", index = False)