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
import tensorflow as tf
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,BatchNormalization,Dropout,Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_data_directory='../input/intel-image-classification/seg_train/seg_train/'
val_data_directory='../input/intel-image-classification/seg_test/seg_test/'
train_iterator=ImageDataGenerator(rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True).flow_from_directory(
    train_data_directory, target_size=(150, 150), color_mode='rgb', classes=None,
    class_mode='sparse', shuffle=True, seed=None)

validation_iterator=ImageDataGenerator(rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True).flow_from_directory(
    val_data_directory, target_size=(150, 150), color_mode='rgb', classes=None,
    class_mode='sparse', shuffle=True, seed=None)
def create_a_Conv_Model():
    model=Sequential([
        Conv2D(128,(5,5),padding='same',strides=(1,1),input_shape=([150,150,3])),
        tf.keras.layers.LeakyReLU(),
        
        Conv2D(64,(3,3),padding='same',strides=(1,1)),
        tf.keras.layers.LeakyReLU(),
        MaxPool2D(pool_size=(2, 2), strides=None, padding='valid'),
        BatchNormalization(),
        
        Conv2D(64,(3,3),padding='same',strides=(1,1)),
        tf.keras.layers.LeakyReLU(),
        MaxPool2D(pool_size=(2, 2), strides=None, padding='valid'),
        
        Dropout(0.4),

        Conv2D(32,(3,3),padding='same',strides=(1,1)),
        tf.keras.layers.LeakyReLU(),
        
        Conv2D(16,(3,3),padding='same',strides=(1,1)),
        tf.keras.layers.LeakyReLU(),
        MaxPool2D(pool_size=(2, 2), strides=None, padding='valid'),
        BatchNormalization(),
        
        Conv2D(16,(3,3),padding='same',strides=(1,1)),
        tf.keras.layers.LeakyReLU(),
        
        Flatten(),
        Dense(32),
        tf.keras.layers.LeakyReLU(),
        Dropout(0.3),
        Dense(6)
    ])
        
    return model
Model=create_a_Conv_Model()
Model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy']) 
Model.summary()
#ES_callback=tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
CHpt_callback=tf.keras.callbacks.ModelCheckpoint('./Chkpt_folder',save_weights_only=True,monitor='val_acc',save_freq=20)
Model.fit_generator(train_iterator,epochs=60,validation_data=validation_iterator,callbacks=[CHpt_callback])
import matplotlib.pyplot as plt
plt.plot(Model.history.history['accuracy'])
plt.plot(Model.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.show()
from sklearn.metrics import classification_report
classification_report(validation_iterator.classes,np.argmax(Model.predict_generator(validation_iterator)))