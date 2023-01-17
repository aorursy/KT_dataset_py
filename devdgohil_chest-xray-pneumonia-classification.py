import math

import os

import cv2

from random import randint

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

%matplotlib inline



import keras

from keras.preprocessing.image import ImageDataGenerator

from keras import Model, layers



from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Dense, Dropout, Flatten

from keras.regularizers import l2
DIR = '../input/chest_xray/chest_xray'

SIZE = 64

TARGET_SIZE = (SIZE, SIZE)
train_datagen = ImageDataGenerator( 

    zoom_range=0.2,

    rotation_range=10,

    horizontal_flip=False,

    rescale=1.0/255)



train_generator = train_datagen.flow_from_directory(

    DIR + '/train',

    batch_size=100,

    class_mode='categorical',

    target_size=TARGET_SIZE)

 

val_datagen = ImageDataGenerator(

    zoom_range=0.2,

    rotation_range=10,

    horizontal_flip=False,

    rescale=1.0/255)

 

val_generator = val_datagen.flow_from_directory(

    DIR + '/test',

    batch_size=100,

    class_mode='categorical',

    target_size=TARGET_SIZE)
def findKey(indices, search_value):

    for key, value in indices.items():

        if(value == search_value):

            return key

    return -1
train_generator.class_indices
for X_batch, y_batch in train_datagen.flow_from_directory(DIR + '/train', batch_size=100, class_mode='categorical', target_size=TARGET_SIZE):

    plt.figure(figsize=(20,20))

    # create a grid of 3x3 images

    for i in range(0, 16):

        ax = plt.subplot(4, 4, i+1)

        ax.set_title(findKey(train_generator.class_indices, np.argmax(y_batch[i])))

        plt.imshow((X_batch[i].reshape(SIZE, SIZE, 3)*255).astype(np.uint8))

    # show the plot

    plt.show()

    break
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(SIZE, SIZE, 3), name="conv1"))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', name="conv2"))

model.add(Dropout(0.25))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', name="conv3"))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', name="conv4"))

model.add(Dropout(0.25))

model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', name="conv5"))

model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', name="conv6"))

model.add(Dropout(0.25))

model.add(Conv2D(8, kernel_size=(5, 5), activation='relu', name="conv7"))

model.add(Conv2D(8, kernel_size=(5, 5), activation='relu', name="conv8"))

model.add(Dropout(0.25))

model.add(Conv2D(4, kernel_size=(7, 7), activation='relu', name="conv9"))

model.add(Flatten())

model.add(Dropout(0.25))

model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))

model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',

              optimizer=keras.optimizers.Adam(lr=0.000035),

              metrics=['accuracy'])
class_weight = {0: 3, 1: 1}
history = model.fit_generator(generator=train_generator,steps_per_epoch=53, epochs=70, validation_data=val_generator, validation_steps=6, use_multiprocessing=True, class_weight=class_weight) 
plt.figure(figsize=(10,5))

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model acc')

plt.ylabel('acc')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
plt.figure(figsize=(10,5))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
score = model.evaluate_generator(val_generator, steps=6, verbose=0)

print('Test loss:', score[0]*100)

print('Test accuracy:', score[1]*100)
# serialize model to JSON

model_json = model.to_json()

with open("PneumoniaClassification.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("PneumoniaClassification.h5")

print("Saved model to disk")
def get_images(directory):

    Images = []

    for image_file in os.listdir(directory): #Extracting the file name of the image from Class Label folder

        if(image_file != '.DS_Store'):

            image = cv2.imread(directory+'/'+image_file) #Reading the image (OpenCV)

            image = cv2.resize(image,TARGET_SIZE) #Resize the image, Some images are different sizes. (Resizing is very Important)

            Images.append(image)

    return Images
def show_prediction(pred_images, prediction):

    row_size = 2

    col_size = 4

    index = 0

    plt.figure(figsize=(20,20))

    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*5,col_size*3))

    for row in range(0,row_size):

        for col in range(0,col_size):

            ax[row][col].imshow(pred_images[index, :, :, :])

            ax[row][col].set_title(findKey(train_generator.class_indices, np.argmax(prediction[index])))

            ax[row][col].axis('off')

            index += 1
pred_images = get_images(DIR + '/val/NORMAL')

pred_images = np.array(pred_images) * 1.0 / 255.0

pred_images.shape
prediction = model.predict(pred_images, verbose=1)

prediction
show_prediction(pred_images, prediction)
pred_images = get_images(DIR + '/val/PNEUMONIA')

pred_images = np.array(pred_images) * 1.0 / 255.0

pred_images.shape
prediction = model.predict(pred_images, verbose=1)

prediction
show_prediction(pred_images, prediction)