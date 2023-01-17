import numpy as np 
import pandas as pd 
import os
import itertools
import tensorflow as tf 
from glob import glob
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import matplotlib.pyplot as plt






test_path = '/kaggle/input/intel-image-classification/seg_test/seg_test'
train_path = '/kaggle/input/intel-image-classification/seg_train/seg_train'

data_gen = ImageDataGenerator(rescale = 1./255,
                              shear_range = 0.2,
                              zoom_range = 0.2,
                              horizontal_flip = True)
training_data = data_gen.flow_from_directory(train_path,
                                             target_size = (100,100),
                                             classes = ["buildings","forest","glacier","mountain","sea","street"],
                                             batch_size = 32,
                                             class_mode = 'categorical')

test_data = data_gen.flow_from_directory(test_path,
                                         target_size = (100,100),
                                         classes = ["buildings","forest","glacier","mountain","sea","street"],
                                         batch_size = 32,
                                         class_mode = 'categorical')
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
model = tf.keras.models.Sequential()
model.add(Conv2D(32,(3,3),input_shape = (100,100,3),activation = 'relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128,(3,3),activation = 'relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128,activation = 'relu'))
model.add(Dropout(0.25))

model.add(Dense(256,activation = 'relu'))
model.add(Dropout(0.25))

model.add(Dense(6,activation = 'softmax'))
model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
model.summary()
model = model.fit(training_data,epochs = 25,batch_size = 32,validation_data = test_data)
plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()
plt.plot(model.history['val_loss'])
plt.plot(model.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()