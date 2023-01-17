import os

import numpy as np

import glob

import shutil



import tensorflow as tf



import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"



zip_file = tf.keras.utils.get_file(origin=_URL,

                                   fname="flower_photos.tgz",

                                   extract=True)



base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']
for cl in classes:

  img_path = os.path.join(base_dir, cl)

  images = glob.glob(img_path + '/*.jpg')

  print("{}: {} Images".format(cl, len(images)))

  train, val = images[:round(len(images)*0.8)], images[round(len(images)*0.8):]



  for t in train:

    if not os.path.exists(os.path.join(base_dir, 'train', cl)):

      os.makedirs(os.path.join(base_dir, 'train', cl))

    shutil.move(t, os.path.join(base_dir, 'train', cl))



  for v in val:

    if not os.path.exists(os.path.join(base_dir, 'val', cl)):

      os.makedirs(os.path.join(base_dir, 'val', cl))

    shutil.move(v, os.path.join(base_dir, 'val', cl))
train_dir = os.path.join(base_dir, 'train')

val_dir = os.path.join(base_dir, 'val')
batch_size = 100

IMG_SHAPE = 150
image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)

train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,

                                               directory=train_dir,

                                               shuffle=True,

                                               target_size=(IMG_SHAPE,IMG_SHAPE))
image_gen = ImageDataGenerator(rescale=1./255,rotation_range=45)



train_data_gen = image_gen.flow_from_directory(

    batch_size=batch_size,

    directory=train_dir,

    shuffle=True,

    target_size=(IMG_SHAPE,IMG_SHAPE)

)
image_gen = ImageDataGenerator(rescale=1./255,zoom_range=0.5)



train_data_gen = image_gen.flow_from_directory(

    batch_size=batch_size,

    directory=train_dir,

    shuffle=True,

    target_size=(IMG_SHAPE,IMG_SHAPE)

)
image_gen_train = ImageDataGenerator(rescale=1./255,rotation_range=45,horizontal_flip=True,width_shift_range=0.15,height_shift_range=0.15) 





train_data_gen = image_gen_train.flow_from_directory(

    batch_size=batch_size,

    directory=train_dir,

    shuffle=True,

    target_size=(IMG_SHAPE,IMG_SHAPE),

    class_mode='binary'

)


image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,

                                                 directory=val_dir,

                                                 target_size=(IMG_SHAPE, IMG_SHAPE),

                                                 class_mode='binary')
model = tf.keras.models.Sequential([

                                    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),

                                    tf.keras.layers.MaxPooling2D(2,2),



                                    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),

                                    tf.keras.layers.MaxPooling2D(2,2),



                                    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),

                                    tf.keras.layers.MaxPooling2D(2,2),



                                    tf.keras.layers.Dropout(0.2),

                                    tf.keras.layers.Flatten(),

                                    tf.keras.layers.Dense(512, activation='relu'),



                                    tf.keras.layers.Dense(5,activation='softmax')

])
# Compile the model

model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
epochs = 30



history = model.fit_generator(

    train_data_gen,

    epochs=epochs,

    validation_data=val_data_gen

)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_range = range(epochs)





plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()