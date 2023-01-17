# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np 

import pandas as pd

import tensorflow as tf
mnist_train = pd.read_csv("../input/digit-recognizer/train.csv")

mnist_test = pd.read_csv("../input/digit-recognizer/test.csv")
mnist_train_data = mnist_train.loc[:, "pixel0":]

mnist_train_label = mnist_train.loc[:, "label"]

mnist_train_data = mnist_train_data/255.0

mnist_test = mnist_test/255.0
mnist_train_data = np.array(mnist_train_data)

mnist_train_label = np.array(mnist_train_label)

mnist_train_data = mnist_train_data.reshape(mnist_train_data.shape[0], 28, 28, 1)

print(mnist_train_data.shape, mnist_train_label.shape)
mnist_test_arr = np.array(mnist_test)

mnist_test_arr = mnist_test_arr.reshape(mnist_test_arr.shape[0], 28, 28, 1)

print(mnist_test_arr.shape)
from keras.utils import np_utils

nclasses = mnist_train_label.max() - mnist_train_label.min() + 1

mnist_train_label = np_utils.to_categorical(mnist_train_label, num_classes = nclasses)

print("Shape of y_train after encoding: ", mnist_train_label.shape)
x_train=mnist_train_data

x_test=mnist_test_arr

y_train=mnist_train_label

x_train.shape
from keras.models import Sequential

from keras.layers import Dropout,Activation,Conv2D,GlobalAveragePooling2D

from keras.optimizers import SGD

import tensorflow as tf
def allcnn():

    model =Sequential()

    model.add(Conv2D(64,(3, 3), padding = 'same' , input_shape=(x_train.shape[1:])))

    model.add(Activation ('relu'))

    model.add(Conv2D(32,(3, 3), padding = 'same' ))

    model.add(Activation ('relu'))

    model.add(Conv2D(64,(3, 3), padding = 'same', strides=(2,2) ))

    model.add(Dropout(0.5))

    

    model.add(Conv2D(64,(3, 3), padding = 'same' ))

    model.add(Activation ('relu'))

    model.add(Conv2D(64,(3, 3), padding = 'same' ))

    model.add(Activation ('relu'))

    model.add(Conv2D(64,(3, 3), padding = 'same', strides=(2,2) ))

    model.add(Dropout(0.5))

    

    

    model.add(Conv2D(64,(3, 3), padding = 'same' ))

    model.add(Activation ('relu'))

    model.add(Conv2D(64,(1, 1), padding = 'valid' ))

    model.add(Activation ('relu'))

    model.add(Conv2D(10,(1, 1), padding = 'same', strides=(2,2) ))

    

    

    model.add(GlobalAveragePooling2D())

    model.add(Activation('softmax'))

    

    

    return model
learning_r= 0.01

weight_decay=1e-6

mometum=0.5

model =allcnn()

sgd= SGD(lr=learning_r,decay=weight_decay,momentum=mometum, nesterov=True)

model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])



print(model.summary())
class myCallback(tf.keras.callbacks.Callback):

      def on_epoch_end(self, epoch, logs={}):

            if(logs.get('accuracy')>0.99):

                print("\nReached 90% accuracy so cancelling training!")

                self.model.stop_training = True
callbacks = myCallback()

hist=model.fit(x_train, y_train,epochs=50,    validation_split=0.2, verbose=1)
plt.plot(hist.history['accuracy'],label = 'ACCURACY')

plt.plot(hist.history['val_accuracy'],label = 'VALIDATION ACCURACY')

plt.legend()
plt.plot(hist.history['loss'],label = 'TRAINING LOSS')

plt.plot(hist.history['val_loss'],label = 'VALIDATION LOSS')

plt.legend()
p = model.predict(x_test)
p_t = []



for i in p:

    p_t.append(np.argmax(i))
submission =  pd.DataFrame({

        "ImageId": mnist_test.index+1,

        "Label": p_t

    })



submission.to_csv('my_submission.csv', index=False)