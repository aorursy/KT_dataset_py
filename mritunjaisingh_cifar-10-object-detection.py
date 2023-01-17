# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from tensorflow.keras.datasets import cifar10
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
train, test = cifar10.load_data()
x_train, y_train = train
x_test, y_test = test
x_train.shape, y_train.shape, x_test.shape, y_test.shape
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train.dtype, y_train.dtype
from sklearn.model_selection import train_test_split
x_train_new, x_val, y_train_new, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 101)
x_train_new.shape, y_train_new.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape
#x_train.shape[1:]
#y_train = tf.keras.utils.to_categorical(y_train, num_classes= 10)
y_train_new = tf.keras.utils.to_categorical(y_train_new, num_classes= 10)
y_val = tf.keras.utils.to_categorical(y_val, num_classes= 10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes= 10)
def build_model():
    
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), padding = "same", activation = "relu",
                                    input_shape = (32,32,3)))
    model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), padding = "same", activation = "relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size = (2,2)))
    model.add(tf.keras.layers.Dropout(0.25))
              
    model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = "same", activation = "relu"))
    model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = "same", activation = "relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size = (2,2)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units = 512, activation = "relu"))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(units = 10, activation = "softmax"))
    
    return model
              
model = build_model()
model.summary()
model.compile(optimizer= "adam", loss= "categorical_crossentropy", metrics= ["accuracy"])
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor= "val_loss", patience= 2, verbose= 1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor= "val_loss", patience= 3, restore_best_weights= True)
history = model.fit(x_train_new, y_train_new, batch_size= 32,
                   epochs= 30, validation_data= (x_val, y_val), shuffle= True, 
                    callbacks= [reduce_lr, early_stop])
plt.plot(history.history['loss'], color='b')
plt.plot(history.history['val_loss'], color='r')
plt.show()
plt.plot(history.history['accuracy'], color='b')
plt.plot(history.history['val_accuracy'], color='r')
plt.show()
pred = model.predict(x_test)
pred = np.argmax(pred, axis= 1)
y_test
pred = tf.keras.utils.to_categorical(pred, num_classes= 10)
#y_test = tf.keras.utils.to_categorical(y_test, num_classes= 10)
pred.shape, y_test.shape
from sklearn.metrics import classification_report
y_test.shape, pred.shape
print(classification_report(y_test, pred))
