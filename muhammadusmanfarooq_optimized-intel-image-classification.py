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
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
%matplotlib inline
train = '../input/intel-image-classification/seg_train/seg_train'

data = ImageDataGenerator(rescale=1./255)
Train = data.flow_from_directory(
    train,
    target_size = (150,150),
    class_mode = 'categorical'
)
test = '../input/intel-image-classification/seg_test/seg_test'

Val = data.flow_from_directory(
    test,
    target_size = (150,150),
    class_mode = 'categorical'
)
dense_layers = [256, 512]
conv_layers = [128, 256]
for dense in dense_layers:
    for conv in conv_layers:
        model = tf.keras.models.Sequential([

            tf.keras.layers.Conv2D( conv, (3,3), activation = 'relu', input_shape = (150,150,3)),
            tf.keras.layers.MaxPooling2D(2,2),

            tf.keras.layers.Conv2D( conv, (3,3), activation = 'relu'),
            tf.keras.layers.MaxPooling2D(2,2),

            tf.keras.layers.Conv2D( 64, (3,3), activation = 'relu'),
            tf.keras.layers.MaxPooling2D(2,2),

            tf.keras.layers.Conv2D( 32, (3,3), activation = 'relu'),
            tf.keras.layers.MaxPooling2D(2,2),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Dense(dense, activation = 'relu'),
            tf.keras.layers.Dense(6, activation = 'softmax')

        ])
        
        Name = 'cnv {}  -  dense {} '.format(conv,dense,int(time.time()))
        tensorboard = TensorBoard(log_dir = 'logs/'.format(Name))
        
        model.summary()

        model.compile(optimizer= keras.optimizers.Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

        history = model.fit_generator(Train, epochs=25, validation_data = Val, verbose = 1, callbacks= [tensorboard])
        
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        plt.plot(epochs, acc, 'r', label='Training accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend(loc=0)
        plt.figure()
        
        plt.plot(epochs, loss, 'r', label='Trainig Loss')
        plt.plot(epochs, val_loss, 'b', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend(loc=0)
        plt.figure()


        plt.show()