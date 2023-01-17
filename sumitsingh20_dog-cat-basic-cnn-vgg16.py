import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

train_data = ImageDataGenerator(rescale = 1./255,
                                shear_range = 0.2,
                                zoom_range = 0.2,
                                horizontal_flip = True)
train_df = train_data.flow_from_directory('../input/cat-and-dog/training_set/training_set',
                                          target_size = (64,64),
                                          batch_size = 32,
                                          class_mode = 'binary')
test_data = ImageDataGenerator(rescale = 1./255)
test_df = test_data.flow_from_directory('../input/cat-and-dog/test_set/test_set',
                                        target_size = (64,64),
                                        batch_size = 32,
                                        class_mode = 'binary')
cnn = tf.keras.models.Sequential()
cnn.add(Conv2D(filters = 32,kernel_size = 3,activation = 'relu',input_shape = [64,64,3]))
cnn.add(MaxPool2D(pool_size = 2,strides = 2))
cnn.add(Conv2D(filters = 32,kernel_size = 3,activation = 'relu'))
cnn.add(MaxPool2D(pool_size = 2,strides = 2))
cnn.add(Flatten())
cnn.add(Dense(units = 128,activation = 'relu'))
cnn.add(Dense(units = 128,activation = 'relu'))
cnn.add(Dense(units = 1,activation = 'sigmoid'))
cnn.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = 'accuracy')
cnn.summary()
cnn.fit(x = train_df,epochs = 30,validation_data = test_df)
plt.plot(cnn.history.history['accuracy'])
plt.plot(cnn.history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()
plt.plot(cnn.history.history['loss'])
plt.plot(cnn.history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory('../input/cat-and-dog/training_set/training_set',
                         target_size=(224,224),
                         classes=['cats', 'dogs'],
                         batch_size=10)

test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory('../input/cat-and-dog/test_set/test_set', target_size=(224,224), classes=['cats', 'dogs'], batch_size=10, shuffle=False)
vgg16_model = tf.keras.applications.vgg16.VGG16()
model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)
for layer in model.layers:
    layer.trainable = False
model.add(Dense(units=2, activation='softmax'))
model.summary()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x = train_batches, 
          steps_per_epoch = len(train_batches),
          epochs = 5,
          validation_data = test_batches
         )
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show();
plt.plot(model.history.history['val_loss'])
plt.plot(model.history.history['loss'])
plt.title('model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()
scores = pd.DataFrame({'Basic Cnn': cnn.evaluate(test_df),
                         'VGG16': model.evaluate(test_batches)})
scores
scores.plot(kind = 'bar',
              figsize = (10,10),
              color = ['lightgreen','lightblue'])
plt.title('Scores & Loss Comparision')
plt.xlabel('0 = Loss,1 = Accuracy')
plt.ylabel('Percent')
plt.legend(['Bsic Cnn','VGG16'])
plt.xticks(rotation = 0);

