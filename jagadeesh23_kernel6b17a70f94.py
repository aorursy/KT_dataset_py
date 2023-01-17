import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
import shutil
from shutil import copyfile
import keras
df=pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv")
test_df=pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv")
df['image_id']=df['image_id']+'.jpg'
test_df['image_id']=test_df['image_id']+'.jpg'
test_df.head()
from sklearn.model_selection import train_test_split
train, valid = train_test_split(df, test_size = 0.2)
from keras_preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=.1,
    rescale=1/255,
    fill_mode='nearest',
    shear_range=0.1,
    brightness_range=[0.5, 1.5])

valid_datagen = ImageDataGenerator(rescale=1/255.0)
test_datagen= ImageDataGenerator(rescale=1/255.0)   
train_generator=train_datagen.flow_from_dataframe(train,directory='/kaggle/input/plant-pathology-2020-fgvc7/images/',
                                                      target_size=(250,250),x_col="image_id",y_col=['healthy','multiple_diseases','rust','scab'],
                                                      class_mode='raw',shuffle=False, batch_size=128)
valid_generator=valid_datagen.flow_from_dataframe(valid,directory='/kaggle/input/plant-pathology-2020-fgvc7/images/',
                                                      target_size=(250,250),x_col="image_id",y_col=['healthy','multiple_diseases','rust','scab'],
                                                      class_mode='raw',shuffle=False, batch_size=128)
test_generator=test_datagen.flow_from_dataframe(test_df,directory='/kaggle/input/plant-pathology-2020-fgvc7/images/',
                                                      target_size=(250,250),x_col="image_id",y_col=None,
                                                      class_mode=None,shuffle=False, batch_size=128)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy',factor=.5,patience=10,min_lr=.000001,verbose=1)
model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16,(5,5),padding='valid',activation='relu',input_shape=(250,250,3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((4,4)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(32,(5,5),padding='valid',activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((3,3)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64,(5,5),padding='valid',activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((3,3)),
        tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Conv2D(64,(15,15),padding='same',activation='relu'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(4,activation="softmax")
    ])


optimizer=tf.keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
model.summary()  
history =model.fit_generator(train_generator,epochs=15,validation_data=valid_generator,callbacks=[reduce_lr],verbose=2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'])
plt.show()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train','Test'])
plt.show()