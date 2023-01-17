from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.preprocessing.image import load_img

import numpy as np

import matplotlib.pyplot as plt

imagepath = '../input/animal-image-datasetdog-cat-and-panda/images/dog.jpg'

image = load_img(imagepath)

plt.imshow(image)

plt.show()
image1 = img_to_array(image)

image2 = np.expand_dims(image,axis=0)

aug = ImageDataGenerator(rotation_range=0.2,horizontal_flip=True,vertical_flip=True,width_shift_range=0.1

                        ,height_shift_range=0.1,fill_mode='nearest',zoom_range=0.2)

plt.figure(figsize=(10,10))

# load the image

img = load_img(imagepath)

# convert to numpy array

data = img_to_array(img)

# expand dimension to one sample

samples = np.expand_dims(data, 0)

# create image data augmentation generator

datagen = ImageDataGenerator(rotation_range=0.2,horizontal_flip=True,vertical_flip=True,width_shift_range=0.1

                        ,height_shift_range=0.1,fill_mode='nearest',zoom_range=0.2)

# prepare iterator

it = datagen.flow(samples, batch_size=1)

# generate samples and plot

for i in range(9):

    # define subplot

    plt.subplot(330 + 1 + i)

    # generate batch of images

    batch = it.next()

    # convert to unsigned integers for viewing

    image = batch[0].astype('uint8')

    # plot raw pixel data

    plt.imshow(image)

# show the figure

plt.show()
! pip install imutils
import cv2

import os

from imutils import paths

## Defining model

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (Conv2D,Activation,BatchNormalization,MaxPooling2D,Flatten,Dense,Dropout)

from tensorflow.keras.optimizers import SGD

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



def imagesloder(imagepaths,width,height):

    data=[]

    labels=[]

    for (i,images) in enumerate(imagepaths):

        image = cv2.imread(images)

        image = cv2.resize(image,(height,width),interpolation=cv2.INTER_AREA)

        image = img_to_array(image)

        # image = np.expand_dims(image,axis=0)

        label = int(images.split(os.path.sep)[-2])

        data.append(image)

        labels.append(label)

    return np.array(data,dtype='float')/255.0,np.array(labels)





def minivgg(width,height,depth,classes):

    input_shape=(height,width,depth)

    ch_dim = -1

    model= Sequential()

    model.add(Conv2D(32,(3,3),padding='same',input_shape=input_shape))

    model.add(Activation('relu'))

    model.add(BatchNormalization(axis=ch_dim))



    model.add(Conv2D(64,(3,3),padding='same'))

    model.add(Activation('relu'))

    model.add(BatchNormalization(axis=ch_dim))



    model.add(MaxPooling2D((2,2)))

    model.add(Dropout(0.25))



    model.add(Conv2D(32,(3,3),padding='same'))

    model.add(Activation('relu'))

    model.add(BatchNormalization(axis=ch_dim))



    model.add(Conv2D(64,(3,3),padding='same'))

    model.add(Activation('relu'))

    model.add(BatchNormalization(axis=ch_dim))



    model.add(MaxPooling2D((2,2)))

    model.add(Dropout(0.25))



    model.add(Flatten())

    model.add(Activation('relu'))

    model.add(BatchNormalization(axis=ch_dim))



    model.add(Flatten())

    model.add(Dense(512))



    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(classes))

    model.add(Activation('softmax'))



    model.compile(loss='sparse_categorical_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy'])

    return model





# config file

flower_dataset_path = '../input/flowers17/17flowers/jpg'

epochs = 100

batch_size = 32



flowerimagepaths = list(paths.list_images(flower_dataset_path))

    

data,labels=imagesloder(flowerimagepaths,64,64)



train_X,test_X,train_y,test_y = train_test_split(data,labels,test_size=0.2,random_state=41)



model=minivgg(64,64,3,17)



model.summary()
H = model.fit(train_X,train_y,validation_data=(test_X,test_y),

              batch_size=batch_size,epochs=epochs,steps_per_epoch=len(train_X) // 32)
plt.figure(figsize=(10,8))

plt.plot(np.arange(0,epochs),H.history['accuracy'],label='Training_accuracy')

plt.plot(np.arange(0,epochs),H.history['loss'],label='Training_loss')

plt.plot(np.arange(0,epochs),H.history['val_accuracy'],label='Validation_accuracy')

plt.plot(np.arange(0,epochs),H.history['val_loss'],label='validation_loss')

plt.xlabel('#Epochs')

plt.ylabel('Percentage')

plt.legend()

plt.title('Training accuracy and Loss plot')

plt.show()
aug = ImageDataGenerator(rotation_range=0.2,horizontal_flip=True,vertical_flip=True,width_shift_range=0.1

                        ,height_shift_range=0.1,fill_mode='nearest',zoom_range=0.2)

H = model.fit_generator(aug.flow(train_X, train_y, batch_size=32),

                        validation_data=(test_X, test_y), steps_per_epoch=len(train_X) // 32,epochs=100, verbose=1)
plt.figure(figsize=(10,8))

plt.plot(np.arange(0,epochs),H.history['accuracy'],label='Training_accuracy')

plt.plot(np.arange(0,epochs),H.history['loss'],label='Training_loss')

plt.plot(np.arange(0,epochs),H.history['val_accuracy'],label='Validation_accuracy')

plt.plot(np.arange(0,epochs),H.history['val_loss'],label='validation_loss')

plt.xlabel('#Epochs')

plt.ylabel('Percentage')

plt.legend()

plt.title('Training accuracy and Loss plot')

plt.show()