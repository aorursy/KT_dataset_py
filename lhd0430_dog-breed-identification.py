import tensorflow as tf



from tensorflow.keras import layers

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator



import os

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
# Hyperparameters

batch_size = 128

epochs = 10

IMG_HEIGHT = 150

IMG_WIDTH = 150

validation_split=0.2



rescale=1./255

rotation_range=0

width_shift_range=0

height_shift_range=0

horizontal_flip=True

zoom_range=0
# Load file-labels data

train_dir = '../input/dog-breed-identification/train/'

df = pd.read_csv('../input/dog-breed-identification/labels.csv')

df['id']=df['id'].apply(lambda x: x+'.jpg')

class_names = np.array(sorted(df['breed'].unique()))

num_classes = class_names.shape[0]
# Create generator with augmentation

image_generator = ImageDataGenerator( rescale=rescale, 

                                      validation_split=validation_split,

                                      rotation_range=rotation_range,

                                      width_shift_range=width_shift_range,

                                      height_shift_range=height_shift_range,

                                      horizontal_flip=horizontal_flip,

                                      zoom_range=zoom_range)

train_data_gen = image_generator.flow_from_dataframe(dataframe=df,

                                                     subset='training',

                                                       batch_size=batch_size,

                                                       x_col='id',

                                                       y_col='breed',

                                                       directory=train_dir,

                                                       shuffle=True,

                                                       target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                       class_mode='categorical')

validation_data_gen = image_generator.flow_from_dataframe(dataframe=df,

                                                          subset='validation',

                                                           batch_size=batch_size,

                                                           x_col='id',

                                                           y_col='breed',

                                                           directory=train_dir,

                                                           shuffle=True,

                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                           class_mode='categorical')
# Visualize sample training data

sample_training_images, sample_training_labels = next(train_data_gen)

sample_training_label_id = np.argmax(sample_training_labels, axis=-1)

sample_training_labels = class_names[sample_training_label_id]

plt.figure(figsize=(20,20))

plt.subplots_adjust(hspace=.5)

for n in range(30):

  plt.subplot(6,5,n+1)

  plt.imshow(sample_training_images[n])

  plt.title(sample_training_labels[n].title())

  plt.axis('off')
# Create the base model from the pre-trained model

'''base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT,IMG_WIDTH,3),

                                              include_top=False,

                                             weights='imagenet')'''



base_model = tf.keras.applications.VGG16(input_shape=(IMG_HEIGHT,IMG_WIDTH,3),

                                               include_top=False,

                                               weights='imagenet')



# Freeze the convolutional base

base_model.trainable = False



base_model.summary()
# Add classification head

model = tf.keras.Sequential([

  base_model,

  layers.GlobalAveragePooling2D(),

  layers.Dense(num_classes, activation='softmax')

])



model.summary()
# Complile the model

model.compile(

  optimizer=tf.keras.optimizers.Adam(),

  loss='categorical_crossentropy',

  metrics=['acc'])
# Start training

steps_per_epoch = np.ceil(train_data_gen.samples/train_data_gen.batch_size)

validation_steps = np.ceil(validation_data_gen.samples/validation_data_gen.batch_size)



history = model.fit_generator(

    train_data_gen,

    steps_per_epoch=steps_per_epoch,

    epochs=epochs,

    validation_data=validation_data_gen,

    validation_steps=validation_steps

)
# Visualize the learning curve

acc = history.history['acc']

val_acc = history.history['val_acc']



loss = history.history['loss']

val_loss = history.history['val_loss']



plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)

plt.plot(acc, label='Training Accuracy')

plt.plot(val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.ylabel('Accuracy')

plt.ylim([min(plt.ylim()),1])

plt.title('Training and Validation Accuracy')



plt.subplot(2, 1, 2)

plt.plot(loss, label='Training Loss')

plt.plot(val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.ylabel('Cross Entropy')

plt.title('Training and Validation Loss')

plt.xlabel('epoch')

plt.show()