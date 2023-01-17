import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
PATH = os.path.join('..', 'input', 'catndog','cats_and_dogs_filtered') 

train_dir = os.path.join(PATH, 'train')

validation_dir = os.path.join(PATH, 'validation')

# directory with our training cat pictures

train_cats_dir = os.path.join(train_dir, 'cats')

# directory with our training dog pictures

train_dogs_dir = os.path.join(train_dir, 'dogs')

# directory with our validation cat pictures

validation_cats_dir = os.path.join(validation_dir, 'cats')

# directory with our validation dog pictures

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
num_cats_tr = len(os.listdir(train_cats_dir))

num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))

num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr

total_val = num_cats_val + num_dogs_val

print('total training cat images:', num_cats_tr)

print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)

print('total validation dog images:', num_dogs_val)

print('—')

print('Total training images:', total_train)

print('Total validation images:', total_val)
batch_size = 128

epochs = 15

IMG_HEIGHT = 150

IMG_WIDTH = 150
# Generator for our training data

train_image_generator = ImageDataGenerator(rescale=1./255)

# Generator for our validation data

validation_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(

    batch_size=batch_size, 

    directory=train_dir,

    shuffle=True,

    target_size=(IMG_HEIGHT, IMG_WIDTH),

    class_mode='binary')



val_data_gen = validation_image_generator.flow_from_directory(

batch_size=batch_size,

directory=validation_dir,

target_size=(IMG_HEIGHT, IMG_WIDTH),

class_mode='binary')

sample_training_images, _ = next(train_data_gen)

def plotImages(images_arr):

    fig, axes = plt.subplots(1, 5, figsize=(20,20))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

        ax.axis('off')

    plt.tight_layout()

    plt.show()

plotImages(sample_training_images[:5])

# Create model

model = Sequential([

    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),

    MaxPooling2D(),

    Conv2D(32, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Conv2D(64, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Flatten(),

    Dense(512, activation='relu'),

    Dense(1)

])
# Compile model

model.compile(

    optimizer='adam',

    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

    metrics=['accuracy'])
# Model summary

model.summary()
# Train the model

history = model.fit_generator(

    train_data_gen,

    steps_per_epoch=total_train // batch_size,

    epochs=epochs,

    validation_data=val_data_gen,

    validation_steps=total_val // batch_size

)
# Predict

results = model.predict(sample_training_images[:5])
model.save('my_model.h5')

# 重新创建完全相同的模型，包括其权重和优化程序

new_model = tf.keras.models.load_model('my_model.h5')

# 显示网络结构

new_model.summary()
#!pip install tensorflowjs
!tensorflowjs_converter --input_format=keras ./my_model.h5 tfjs_model
!zip -r downloadmeeeee.zip tfjs_model  