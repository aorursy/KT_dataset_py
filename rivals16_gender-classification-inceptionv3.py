import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
pre_trained_model = InceptionV3(include_top = False,
                               input_shape = (150,150,3),
                               weights='imagenet')
for layers in pre_trained_model.layers:
    layers.trainable = False
last_layer = pre_trained_model.get_layer('mixed8')
last_output = last_layer.output
x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024,activation = 'relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(1,activation = 'sigmoid')(x)

model = tf.keras.Model(pre_trained_model.input , x)
hiss = model.compile(optimizer = 'Adam',loss = tf.keras.losses.BinaryCrossentropy(),metrics = ['accuracy'])
data_gen = ImageDataGenerator(rescale = 1./255 ,
                              width_shift_range = 0.2 ,
                              validation_split=0.1,
                              height_shift_range = 0.2 ,
                              shear_range = 0.2 ,
                              horizontal_flip = True ,
                              vertical_flip = True,
                              zoom_range = 0.2)
training_data = data_gen.flow_from_directory('/kaggle/input/men-women-classification/data',
                                            target_size = (150,150),
                                            class_mode='binary',
                                            batch_size = 32,
                                            subset = 'training'
                                            )
validation_data = data_gen.flow_from_directory('/kaggle/input/men-women-classification/data',
                                              target_size = (150,150),
                                              class_mode='binary',
                                              batch_size = 32,
                                              subset = 'validation')
history = model.fit_generator(training_data,epochs = 30 , steps_per_epoch =  93,validation_data = validation_data)
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


plt.show()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()


plt.show()