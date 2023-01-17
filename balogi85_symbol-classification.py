import pandas as pd
import numpy as np
import os
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras.callbacks import EarlyStopping
path_pict = '../input/currency-symbol-datasets/datasets/datasets/' 
path_csv = '../input/currency-symbol-datasets/datasets.csv'
data = pd.read_csv(path_csv)
data.head()
img_plot = plt.imshow(Image.open('../input/currency-symbol-datasets/datasets/datasets/dollar/dollar1.jpg'))
pict = cv2.imread('../input/currency-symbol-datasets/datasets/datasets/dollar/dollar100.jpg')
pic = pict.shape
img_height, img_width = pic[0], pic[1]
batch_size = 32
nb_epochs = 60
opt_adam = Adam(learning_rate=0.001)
opt_sgd = SGD(learning_rate=0.001)
opt_rms = RMSprop(learning_rate=0.001)
train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    rotation_range=0.4,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    path_pict,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    path_pict,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epochs, logs={}):
    if logs.get('val_accuracy') > 0.998:
         print('\nCallback is stopped the training!')
         model.stop_training=True


callback = myCallback()
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer=opt_adam,
              metrics=['accuracy'])
results = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs = nb_epochs, callbacks=[callback])
pd.DataFrame(results.history).plot(figsize=(18, 5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

