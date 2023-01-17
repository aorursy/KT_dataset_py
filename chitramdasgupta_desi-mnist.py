import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns

sns.set_style('dark')

import sklearn

import tensorflow as tf

from tensorflow import keras
import os



train = pd.read_csv('../input/Kannada-MNIST/train.csv')

test = pd.read_csv('../input/Kannada-MNIST/test.csv')

dig_mnist = pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')
print(train.shape)

print(dig_mnist.shape)

print(test.shape)
train.columns
train_labels = train['label'].values

train_data = train.drop('label',axis=1).values



val_labels = dig_mnist['label'].values

val_data = dig_mnist.drop('label',axis=1).values



test_data = test.drop('id', axis=1).values



print(train_labels)

print(train_data)
train_data = (train_data - np.mean(train_data)) / 255.

val_data = (val_data - np.mean(val_data)) / 255.

test_data = (test_data - np.mean(test_data)) / 255.



train_data
from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

      rotation_range=40,

      width_shift_range=0.2,

      height_shift_range=0.2,

      shear_range=0.2,

      zoom_range=0.2,

      horizontal_flip=True,

      fill_mode='nearest')



# Let's reshape our 2d arrays because input to datagen.fit() should have rank 4.



train_data = train_data.reshape(-1,28,28,1)

val_data = val_data.reshape(-1,28,28,1)

test_data = test_data.reshape(-1,28,28,1)



datagen.fit(train_data)
model = keras.models.Sequential([

    keras.layers.Conv2D(64, (3,3), padding='same', input_shape=(28, 28, 1)),

    keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    keras.layers.LeakyReLU(alpha=0.1),

    

    keras.layers.Conv2D(64, (3,3), padding='same'),

    keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    keras.layers.LeakyReLU(alpha=0.1),

    

    keras.layers.Conv2D(64, (3,3), padding='same'),

    keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    keras.layers.LeakyReLU(alpha=0.1),

    

    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Dropout(0.2),

    

    keras.layers.Conv2D(128, (3,3), padding='same', input_shape=(28, 28, 1)),

    keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    keras.layers.LeakyReLU(alpha=0.1),

    

    keras.layers.Conv2D(128, (3,3), padding='same', input_shape=(28, 28, 1)),

    keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    keras.layers.LeakyReLU(alpha=0.1),

    

    keras.layers.Conv2D(128, (3,3), padding='same', input_shape=(28, 28, 1)),

    keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    keras.layers.LeakyReLU(alpha=0.1),

    

    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Dropout(0.2),

    

    

    keras.layers.Conv2D(256, (3,3), padding='same', input_shape=(28, 28, 1)),

    keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    keras.layers.LeakyReLU(alpha=0.1),

    

    keras.layers.Conv2D(256, (3,3), padding='same', input_shape=(28, 28, 1)),

    keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    keras.layers.LeakyReLU(alpha=0.1),

    

    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Dropout(0.2),

    

    keras.layers.Flatten(),

    

    keras.layers.Dense(256),

    keras.layers.LeakyReLU(alpha=0.1),

    keras.layers.BatchNormalization(),

    

    keras.layers.Dense(256),

    keras.layers.LeakyReLU(alpha=0.1),

    keras.layers.BatchNormalization(),

    

    keras.layers.Dense(10, activation='softmax')

    

])





optimizer = keras.optimizers.RMSprop(learning_rate=0.002,rho=0.9)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])



model.summary()
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau( 

    monitor='loss',    

    factor=0.2,       

    patience=2,        

    verbose=1,         

    mode="min",       

    min_delta=0.0001,  

    cooldown=0, 

    min_lr=0.00001)



es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=3, restore_best_weights=True)



history = model.fit(datagen.flow(train_data, train_labels, batch_size=1024),

                              steps_per_epoch = len(train_data) // 1024,

                              epochs = 100,

                              validation_data = (np.array(val_data),np.array(val_labels)),

                              validation_steps = 50,

                              callbacks = [learning_rate_reduction, es])
print(history.history.keys())

epochs = len(history.history['loss'])
y1 = history.history['loss']

y2 = history.history['val_loss']

x = np.arange(1, epochs+1)



plt.plot(x, y1, y2)

plt.legend(['loss', 'val_loss'])

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.tight_layout()
y1 = history.history['accuracy']

y2 = history.history['val_accuracy']

x = np.arange(1, epochs+1)



plt.plot(x, y1, y2)

plt.legend(['accuracy', 'val_accuracy'])

plt.xlabel('Epochs')

plt.ylabel('Acc')

plt.tight_layout()
score = model.evaluate( np.array(val_data), np.array(val_labels),batch_size = 1024)
predictions = model.predict(test_data)
pred = np.argmax(predictions, axis=1)

pred
# test_id = test_data['Id']

test_id = np.arange(pred.shape[0])

test_id
submission = pd.DataFrame({'id': test_id, 'label': pred})

submission.to_csv('submission.csv', index = False)