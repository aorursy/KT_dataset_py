#Importing necessary modules and API



# For affray related conversions

import numpy as np



# To handle CSV data effciently in form of DataFrames

import pandas as pd



# To plot images in data base

from matplotlib import pyplot as plt



# file listing

from glob import glob



# Necessary modules fromm keras API

from keras.models import Sequential,load_model

from keras.layers import Dense,Dropout,Flatten,ZeroPadding2D,Conv2D,MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import to_categorical

from keras.optimizers import Adam,SGD,RMSprop

from keras.callbacks import EarlyStopping

from keras.preprocessing.image import load_img

from PIL import Image







# I will explain why and where these modules are being used 

# as we progress in this notebook
# listing the input data 



# *** You need to change this path by simply replacing the 

# path argument with the path containing the data in your machine ****



# code will work fine if you are using kaggle notebooks



import os

paths = os.listdir(path="../input")

print(paths)
# defining paths to train,test and validation directories



path_train = "../input/chest-xray-pneumonia/chest_xray/train"

path_val = "../input/chest-xray-pneumonia/chest_xray/val"

path_test = "../input/chest-xray-pneumonia/chest_xray/test"
#getting paths of all images in desired directory



img = glob(path_train+"/PNEUMONIA/*.jpeg")

print(img[0:6])
# printing the number of images 



print(len(img))
# plotting an image in dataset

# you may replace 0 in the index of img (list containing paths to all images) by any number in between 0-3875

# to have a look at different images in data



image = np.asarray(plt.imread(img[0]))

plt.imshow(image)



print("you can clearly see this x-ray is of pneumoniatic lungs")
image.shape



#we will be downscaling these images for our model so not a big deal with the size
# defining classes in our classification predictions

classes = ["NORMAL", "PNEUMONIA"]



# preparing the train_data list to contain all the images 

# showing symptoms of Pneumonia and normal

# to be provided to model for traing



train_data = glob(path_train+"/NORMAL/*.jpeg")

train_data += glob(path_train+"/PNEUMONIA/*.jpeg")





# initiating the ImageDataGenerator() object to

# augment the images and making pipeline to the training of model

data_gen = ImageDataGenerator() 
# preparing the batches for training



train_batches = data_gen.flow_from_directory(path_train, target_size = (226, 226), classes = classes, class_mode = "categorical")

val_batches = data_gen.flow_from_directory(path_val, target_size = (226, 226), classes = classes, class_mode = "categorical")

test_batches = data_gen.flow_from_directory(path_test, target_size = (226, 226), classes = classes, class_mode = "categorical")
print(train_batches.image_shape)

print(len(train_batches))

print(len(val_batches))
#This is a Convolutional Artificial Neural Network

#VGG16 Model

model = Sequential()

model.add(ZeroPadding2D((1,1),input_shape=train_batches.image_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D((2,2), strides=(2,2)))



model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D((2,2), strides=(2,2)))



model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(256, (3, 3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(256, (3, 3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(256, (3, 3), activation='relu'))

model.add(MaxPooling2D((2,2), strides=(2,2)))



model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(512, (3, 3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(512, (3, 3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(512, (3, 3), activation='relu'))

model.add(MaxPooling2D((2,2), strides=(2,2)))



model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(512, (3, 3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(512, (3, 3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(512, (3, 3), activation='relu'))

model.add(MaxPooling2D((2,2), strides=(2,2)))



model.add(Flatten())

model.add(Dense(4096, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1000, activation='relu'))

model.add(Dropout(0.5))



# last layer is added with only two nodes as we need to classify between two classes



model.add(Dense(2, activation='softmax'))
model.summary()
optimizer = Adam(lr = 0.0001)

early_stopping_monitor = EarlyStopping(patience = 3, monitor = "val_acc", mode="max", verbose = 2)

model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)

history = model.fit_generator(epochs=5, callbacks=[early_stopping_monitor], shuffle=True, validation_data=val_batches, generator=train_batches, steps_per_epoch=163, validation_steps=1,verbose=1)

prediction = model.predict_generator(generator=train_batches, verbose=2, steps=100)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='best')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='best')

plt.show()
model.save("Pnemonia_detection.h5")
del model
#loading  the saved model

prev_model = load_model("./Pnemonia_detection.h5")
# checking the type of object of prev_model 

print(prev_model)
# load the image using the load_img fucntion in keras so that

# the image can be resized at the time of loading as the size (226,226) is required by the model



img = load_img(path_test+"/PNEUMONIA/person100_bacteria_475.jpeg",target_size = (226,226))



# as the grayscale image would have three dimenions even it contains only one channel 

# we need to add one more dimension in image before sending it to model



img = np.expand_dims(np.asarray(img),axis=0)
print(prev_model.predict(img))