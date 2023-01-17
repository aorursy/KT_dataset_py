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
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import random
import seaborn as sns
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Flatten,Dropout,AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
train_ben = "../input/skin-cancer-malignant-vs-benign/train/benign"
train_mal = "../input/skin-cancer-malignant-vs-benign/train/malignant"
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['normal', 'danger']
students = [len(train_ben),len(train_mal)]
ax.bar(langs,students)
plt.show()
training_dir = "../input/skin-cancer-malignant-vs-benign/train"
training_generator = ImageDataGenerator(rescale=1./255,
                                   rotation_range=15,
                                   shear_range=0.2,
                                   zoom_range=0.2 )
train_generator = training_generator.flow_from_directory(training_dir,target_size=(200,200),batch_size=4,class_mode='binary')
testing_dir = "../input/skin-cancer-malignant-vs-benign/test"
testing_generator = ImageDataGenerator(rescale=1./255)
test_generator = testing_generator.flow_from_directory(testing_dir,target_size=(200,200),batch_size=4,class_mode='binary')

ak=tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),input_shape=(200,200,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),

    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
    
])
ak.summary()
ak.compile(optimizer = tf.keras.optimizers.Adam(lr=0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])
history = ak.fit_generator(train_generator,
                           validation_data = test_generator,
                           epochs = 5,
                           verbose = 1)
%matplotlib inline
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'r', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.plot(epochs, loss, 'g', label='loss')
plt.plot(epochs, val_loss, 'y', label='Validation loss')
plt.title('Training,validation accuracy and loss')
plt.legend(loc=0)
plt.figure()
print("Loss of the model is - " , ak.evaluate(test_generator)[0]*100 , "%")
print("Accuracy of the model is - " , ak.evaluate(test_generator)[1]*100 , "%")
