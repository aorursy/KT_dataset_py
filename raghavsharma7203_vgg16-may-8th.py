# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

import matplotlib.image as img

import tensorflow.keras as keras

import numpy as np





import os

print(os.listdir("../input/"))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from tensorflow.keras.preprocessing.image import ImageDataGenerator



batch_size = 32



def generators(shape, preprocessing): 

    '''Create the training and validation datasets for 

    a given image shape.

    '''

    imgdatagen = ImageDataGenerator(

        rescale=1./255,

        rotation_range=20,

        width_shift_range=0.2,

        height_shift_range=0.2,

        horizontal_flip=True,

        fill_mode='nearest',

        preprocessing_function = preprocessing,

        validation_split = 0.1,

    )

    

    

    height, width = shape



    train_dataset = imgdatagen.flow_from_directory(

        '../input/newhandwriting/generated/',

        target_size = (height, width), 

        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'),

        batch_size = batch_size,

        subset = 'training', 

    )



    val_dataset = imgdatagen.flow_from_directory(

        '../input/newhandwriting/generated/',

        target_size = (height, width), 

        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'),

        batch_size = batch_size,

        subset = 'validation'

    )

    return train_dataset, val_dataset
vgg = keras.applications.vgg16
train_dataset, val_dataset = generators((224,224), preprocessing=vgg.preprocess_input)
vgg_conv = vgg.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Freeze the layers except the last 4 layers

for layer in vgg_conv.layers[:-4]:

    layer.trainable = False



# Check the trainable status of the individual layers

for layer in vgg_conv.layers:

    print(layer, layer.trainable)
# Create the model

model = keras.Sequential()



# Add the vgg convolutional base model

model.add(vgg_conv)



# Add new layers

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(1024, activation='relu'))

model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(10, activation='softmax'))



# Show a summary of the model. Check the number of trainable parameters

model.summary()
from keras.callbacks import ModelCheckpoint



checkpoint = ModelCheckpoint("../working/model2.h5", monitor = 'val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

# Compile the model

model.compile(loss='categorical_crossentropy',

              optimizer=keras.optimizers.Adam(lr=1e-4),

              metrics=['acc'])

# Train the model

history = model.fit_generator(

      train_dataset,

      steps_per_epoch=train_dataset.samples/batch_size ,

      epochs=20,

      validation_data=val_dataset,

      validation_steps=val_dataset.samples/batch_size,

      verbose=1,

      callbacks=[checkpoint])
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.figure()

plt.plot(epochs, acc, 'b', label = 'Training accuracy')

plt.plot(epochs, val_acc, 'r', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

# plt.savefig('Accuracy.jpg')

plt.figure()

plt.plot(epochs, loss, 'b', label = 'Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

# plt.savefig('Loss.jpg')

import cv2

import tensorflow as tf

CATEGORIES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

def prepare(file):

    IMG_SIZE = 224

    img_array = cv2.imread(file)

    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE),3)

    return new_array.reshape(1,IMG_SIZE, IMG_SIZE,-1)

model = tf.keras.models.load_model("../working/model2.h5")

image = prepare("../input/testing/testing/4_1.png") #your image path



# image = image.reshape(image.shape[0],224,224,3)

# print(image.shape)

prediction = model.predict([image])

prediction = list(prediction[0])

print(prediction)

print("And the prediction is:")

result = CATEGORIES[prediction.index(max(prediction))]

print(result)

print("rghv23")