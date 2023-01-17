import os

print(os.listdir("../input/ye358311_fender_apron/YE358311_Fender_apron"))
from os import walk
'''
for (dirpath, dirnames, filenames) in walk("../input"):
    print("Directory path: ", dirpath)
    print("Folder name: ", dirnames)
    print("File name: ", filenames)'''

##Steps - Create Keras CNN to detect,Remove Background
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt #Ploting charts
from glob import glob #retriving an array of files in directories
from keras.models import Sequential #for neural network models
from keras.layers import Dense, Dropout, Flatten, ZeroPadding2D, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator #Data augmentation and preprocessing
from keras.utils import to_categorical #For One-hot Encoding
from keras.optimizers import Adam, SGD, RMSprop #For Optimizing the Neural Network
from keras.callbacks import EarlyStopping
YE358311_Healthy_train = glob("../input/ye358311_fender_apron/YE358311_Fender_apron/train/YE358311_Healthy/*.jpg")
YE358311_defects_train = glob("../input/ye358311_fender_apron/YE358311_Fender_apron/train/YE358311_defects/*.jpg")
YE358311_Healthy_test = glob("../input/ye358311_fender_apron/YE358311_Fender_apron/test/YE358311_Healthy/*.jpg")
YE358311_defects_test = glob("../input/ye358311_fender_apron/YE358311_Fender_apron/test/YE358311_defects/test/*.jpg")
img = np.asarray(plt.imread(YE358311_Healthy_train[0]))
plt.imshow(img)

img = np.asarray(plt.imread(YE358311_defects_train[0]))
plt.imshow(img)
nb_train_samples = 206
epochs = 20
batch_size = 16
classes = ["YE358311_Healthy", "YE358311_defects"]
path_train = "../input/ye358311_fender_apron/YE358311_Fender_apron/train"
path_test = "../input/ye358311_fender_apron/YE358311_Fender_apron/test"
data_gen = ImageDataGenerator() #Augmentation happens here

#target_size = (350, 350)
train_batches = data_gen.flow_from_directory(path_train,target_size = (350, 350), batch_size=16, classes = classes, class_mode = "categorical")
test_batches = data_gen.flow_from_directory(path_test,target_size = (350, 350), batch_size=1, classes = classes, class_mode = "categorical")
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
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.summary()
optimizer = Adam(lr = 0.001)
model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)
history = model.fit_generator(epochs=20, shuffle=True,  generator=train_batches, steps_per_epoch=206/16, verbose=2)
prediction = model.predict_generator(generator=test_batches, verbose=2, steps=44)
prediction = model.predict_generator(generator=test_batches, verbose=2, steps=44)