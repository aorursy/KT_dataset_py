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
train_datagen = ImageDataGenerator( 

    zoom_range=0.2,

    rotation_range=10,

    horizontal_flip=True,

    rescale=1.0/255)



train_generator = train_datagen.flow_from_directory(

    '../input/seg_train/seg_train',

    batch_size=100,

    class_mode='sparse',

    target_size=(150, 150))

 

test_datagen = ImageDataGenerator(

    zoom_range=0.2,

    rotation_range=10,

    horizontal_flip=True,

    rescale=1.0/255)

 

test_generator = test_datagen.flow_from_directory(

    '../input/seg_test/seg_test',

    class_mode='sparse',

    target_size=(150, 150))
def findKey(indices, search_value):

    for key, value in indices.items():

        if(value == search_value):

            return key

    return -1
for X_batch, y_batch in train_datagen.flow_from_directory('../input/seg_train/seg_train', batch_size=100, class_mode='sparse', target_size=(150, 150)):

    plt.figure(figsize=(20,20))

    # create a grid of 3x3 images

    for i in range(0, 16):

        ax = plt.subplot(4, 4, i+1)

        ax.set_title(findKey(train_generator.class_indices, y_batch[i]))

        plt.imshow((X_batch[i].reshape(150, 150, 3)*255).astype(np.uint8))

    # show the plot

    plt.show()

    break
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3), name="conv1"))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', name="conv2"))

model.add(MaxPooling2D(pool_size=(3, 3), name="maxpool1"))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', name="conv3"))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', name="conv4"))

model.add(MaxPooling2D(pool_size=(3, 3), name="maxpool2"))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', name="conv5"))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', name="conv6"))

model.add(MaxPooling2D(pool_size=(3, 3), name="maxpool3"))

model.add(Flatten())

model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))

model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))

model.add(Dropout(0.25))

model.add(Dense(6, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',

              optimizer=keras.optimizers.Adam(lr=0.0001),

              metrics=['accuracy'])
model.summary()
history = model.fit_generator(generator=train_generator,steps_per_epoch=140, epochs=40, validation_data=test_generator, validation_steps=30, use_multiprocessing=True)    
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
score = model.evaluate_generator(test_generator, steps=30, verbose=0)

print('Test loss:', score[0]*100)

print('Test accuracy:', score[1]*100)
from keras.models import Model

for X_batch, y_batch in train_datagen.flow_from_directory('../input/seg_train/seg_train', batch_size=1000, class_mode='categorical', target_size=(150, 150)):

    layer_outputs = [layer.output for layer in model.layers]

    activation_model = Model(inputs=model.input, outputs=layer_outputs)

    activations = activation_model.predict(X_batch[0].reshape(1,150,150,3))

    break
def display_activation(activations, col_size, row_size, act_index): 

    activation = activations[act_index]

    activation_index=0

    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*5,col_size*3))

    for row in range(0,row_size):

        for col in range(0,col_size):

            ax[row][col].imshow(activation[0, :, :, activation_index])

            activation_index += 1
plt.imshow((X_batch[0].reshape(150, 150, 3)*255).astype(np.uint8))

print(y_batch[0])
display_activation(activations, 8, 8, 2)
display_activation(activations, 16, 8, 5)
# serialize model to JSON

model_json = model.to_json()

with open("IntelImageClassification.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("IntelImageClassification.h5")

print("Saved model to disk")
def get_images(directory):

    Images = []

    for image_file in os.listdir(directory): #Extracting the file name of the image from Class Label folder

        image = cv2.imread(directory+'/'+image_file) #Reading the image (OpenCV)

        image = cv2.resize(image,(150,150)) #Resize the image, Some images are different sizes. (Resizing is very Important)

        Images.append(image)

    return Images
pred_images = get_images('../input/seg_pred/seg_pred')

pred_images = np.array(pred_images) * 1.0 / 255.0

pred_images.shape
prediction = model.predict(pred_images, verbose=1)
prediction.shape
row_size = 5

col_size = 5

index = 0

fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*5,col_size*3))

for row in range(0,row_size):

    for col in range(0,col_size):

        rnd_number = randint(0,len(pred_images))

        ax[row][col].imshow(pred_images[rnd_number, :, :, :])

        ax[row][col].set_title(findKey(train_generator.class_indices, np.argmax(prediction[rnd_number])))

        ax[row][col].axis('off')

        index += 1