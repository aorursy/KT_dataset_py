# Importing LIBRARIES

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras import backend as K

import os

import numpy as np

import pandas as np

import matplotlib.pyplot as plt

%matplotlib inline
# Specifying the path of the data(train,test,validaton)

train = '../input/chest_xray/chest_xray/train'

test = '../input/chest_xray/chest_xray/test' 

val = '../input/chest_xray/chest_xray/val'
#Here we have backended keras to tensorflow ,so we go for channel_last ie to specify the channel value as the last dimension in shape of the input.

img_width,img_height= 150,150

input_shape = (img_width,img_height,3)





#It’s just a thing function that you use to get the output of node. It is also known as Transfer Function.





model = Sequential()

# The number of filters are 32 and the kernal_size is (3,3)

model.add(Conv2D(32, (3, 3), input_shape=input_shape))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dropout(50))

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(50))

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(50))

model.add(Dense(1))

model.add(Activation('sigmoid'))
#Here we use RMSPROP optimizer and BINARY_CROSSENTROPY as loss function  

model.compile(loss='binary_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])
model.summary()
#ImageDataGenerator-Generate batches of tensor image data with real-time data augmentation. The data will be looped over (in batches).



train_datagen = ImageDataGenerator(rescale=1. / 255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Here we import images directly from Directory by using flow_from_directory method.

#flow_from_directory() automatically infers the labels from the directory structure of the folders containing images

train_generator = train_datagen.flow_from_directory(

    train,

    target_size=(img_width, img_height),

    batch_size=16,

    class_mode='binary')



test_generator = test_datagen.flow_from_directory(

    test,

    target_size=(img_width, img_height),

    batch_size=16,

    class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

    val,

    target_size=(img_width, img_height),

    batch_size=16,

    class_mode='binary')
#We Fit the model here using fit_generator as we are dealing with large datasets.

model.fit_generator(

    train_generator,

    steps_per_epoch=5217 // 16,

    epochs=20,

    validation_data=validation_generator,

    validation_steps=17 // 16)
#Accuracy of test data.

scores = model.evaluate_generator(test_generator,624/16)

print("\nAccuracy:"+" %.2f%%" % ( scores[1]*100))









# saving model in H5 format.

model.save('vison_v1.0.h5')
# saving model in Json format.

model_json = model.to_json()

with open("model.json","w") as json_file:

    json_file.write(model_json)
# Displaying images of Normal and Pneumonia 

img_n = load_img('../input/chest_xray/chest_xray/train/NORMAL/NORMAL2-IM-0927-0001.jpeg') 

plt.imshow(img_n)

plt.title("Normal")

plt.show()

img_p = load_img('../input/chest_xray/chest_xray/train/PNEUMONIA/person755_bacteria_2659.jpeg') 

plt.imshow(img_p) 

plt.title("Pneumonia")

plt.show()