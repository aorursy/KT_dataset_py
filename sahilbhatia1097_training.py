
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.applications.vgg16 import VGG16
#from  import keras_modules_injection
from os import listdir
from os.path import join
from sklearn.preprocessing import LabelEncoder
print(os.listdir("../input/model-1/training_set/training_set"))
from sklearn.preprocessing import OneHotEncoder
import matplotlib.image as mpimg
import cv2
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D,Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.callbacks import ModelCheckpoint

# Any results you write to the current directory are saved as output.
train_image_paths=[]
train_image_labels=[]
mypath=("../input/model-1/training_set/training_set")
for f in listdir(mypath):
    for imagename in listdir(join(mypath,f)):
        print(join(join(mypath,f),imagename))
        train_image_paths.append(join(join(mypath,f),imagename))
        train_image_labels.append(f)

        

    

    
test_image_paths=[]
test_image_labels=[]
mypath=("../input/model-1/test_set/test_set")
for f in listdir(mypath):
    for imagename in listdir(join(mypath,f)):
        print(join(join(mypath,f),imagename))
        test_image_paths.append(join(join(mypath,f),imagename))
        test_image_labels.append(f)
        
lb_make = LabelEncoder()
train_image_labels=lb_make.fit_transform(train_image_labels)
test_image_labels=lb_make.fit_transform(test_image_labels)
onehot_encoder = OneHotEncoder(sparse=False)
test_image_labels = test_image_labels.reshape(len(test_image_labels), 1)
test_image_labels = onehot_encoder.fit_transform(test_image_labels)

train_image_labels = train_image_labels.reshape(len(train_image_labels), 1)
train_image_labels = onehot_encoder.fit_transform(train_image_labels)
print(train_image_labels[0])
print(len(train_image_labels))
print(len(train_image_paths))
print(np.random.permutation(len(train_image_paths)))
print(train_image_labels[4209])
ImageHeight = 224
ImageWidth = 224
ImageChannels = 3
def batch_generator(image_paths, imagelabels, batch_size, isTraining):
    images = np.empty([batch_size, ImageHeight, ImageWidth, ImageChannels])
    labels = np.empty([batch_size, 6])
    
    while True:
        i=0
        for index in np.random.permutation(len(image_paths)):
            imagePath = image_paths[index]
            label = imagelabels[index]
            
            image = Load_Image(imagePath)
            
            images[i] = preprocess(image)
            labels[i] = label
            i +=1
            if i== batch_size:
                break
        yield images, labels

def Load_Image(image_file):
    return mpimg.imread(image_file.strip())
def preprocess(image):
    return cv2.resize(image,(ImageHeight, ImageWidth), cv2.INTER_AREA)

# classifier = VGG16()

# x = classifier.output
# x = Flatten()(x)
# predictions = Dense(6, activation='softmax')(x)

# # # This is the model we will train
# model = Model(inputs=classifier.input, outputs=predictions)


# model.compile(loss='categorical_crossentropy', 
#             optimizer='adam', 
#             metrics=['accuracy'])
# model.summary()

# classifier = Sequential()
# classifier.add(Convolution2D(64, (3, 3), input_shape = (64, 64, 3), activation = 'relu',padding='same'))
# classifier.add(MaxPooling2D(pool_size = (2, 2)))
# classifier.add(Convolution2D(128, (3, 3), activation = 'relu',padding='same'))
# classifier.add(MaxPooling2D(pool_size = (2, 2)))
# classifier.add(Convolution2D(256, (3, 3), activation = 'relu',padding='same'))
# classifier.add(MaxPooling2D(pool_size = (2, 2)))
# classifier.add(Convolution2D(512, (3, 3), activation = 'relu',padding='same'))
# classifier.add(MaxPooling2D(pool_size = (2, 2)))
# classifier.add(Convolution2D(512, (3, 3), activation = 'relu',padding='same'))
# classifier.add(Flatten())
# classifier.add(Dense(units = 256, activation = 'relu'))
# classifier.add(Dense(units = 64, activation = 'relu'))
# classifier.add(Dense(units = 6, activation = 'softmax'))
# optimizer = Adam(lr=1e-3)
# classifier.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
# classifier.summary()
classifier = Sequential()
classifier.add(Convolution2D(64, (3, 3), padding="same",input_shape=(224,224,3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(128, (3, 3), padding="same", activation = "relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(256, (3, 3), padding="same", activation = "relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(512, (3, 3), padding="same", activation = "relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Convolution2D(512, (3, 3), padding="same", activation = "relu"))

classifier.add(Flatten())
classifier.add(Dense(256,activation = "relu"))
classifier.add(Dense(64,activation = "relu"))
classifier.add(Dense(6, activation = "softmax"))

classifier.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
SETS = len(train_image_paths)/32
checkpoint=ModelCheckpoint('IndoorModel.h5',
                          monitor='val_loss',
                          verbose=1,
                          save_best_only=True,
                          mode='categorical_crossentropy')
#model.compile(loss='mean_squared_error',optimizer=Adam(lr=le-4),metrics=['accuracy'])
classifier.fit_generator(batch_generator(train_image_paths,train_image_labels,32,True),
                   SETS,
                   5,
                   max_q_size=24,
                   validation_data=batch_generator(test_image_paths,test_image_labels,32,False),
                   nb_val_samples=len(test_image_paths)/10,
                   callbacks=[checkpoint],
                   verbose=1)
from keras.preprocessing import image

test_image = image.load_img('../input/model-1/test_set/test_set/AquaZone/resizedaqua00060.png',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
result
IMGPATH = "../input/model-1/test_set/test_set/BlueZone/"
filenames = os.listdir(IMGPATH)
images = []
for i in range(100):
    test_image = image.load_img(IMGPATH + filenames[i],target_size=(64,64))
    test_image = image.img_to_array(test_image)
    images.append(test_image)
images = np.array(images)
classifier.predict(images)
from keras.preprocessing import image

test_image = image.load_img('../input/model-1/test_set/test_set/BlueZone/resizedblue00205.png',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
result
from keras.preprocessing import image

test_image = image.load_img('../input/model-1/test_set/test_set/AquaZone/resizedaqua00311.png',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
result
import os, glob
test_image_paths=[]
test_image_labels=[]
mypath=("../input/model-1/test_set/test_set")
for imagename in glob.glob(os.path.join(mypath,"*.png")):
    print(imagename)
#         test_image = image.load_img(mypath,target_size=(64,64))
#         test_image = image.img_to_array(test_image)
#         test_image = np.expand_dims(test_image, axis = 0)
#         result = classifier.predict(test_image)
#         result
def GenerateModel():
    input  = Input(shape = (ImageHeight, ImageWidth, 3), name = 'image_input')
    x = Conv2D(32, (3, 3), input_shape= (ImageHeight, ImageWidth, 3), padding='same',
           activation='relu')(input)
    x = MaxPooling2D(pool_size = (2, 2))(x)
    x = Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu')(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation = 'relu')(x)
    x = Dense(6, activation = 'softmax')(x)
    
    my_model = Model(input = input, output = x)
    my_model.summary()
    return my_model
    
def TrainModel(model, X_train,X_test,y_train,y_test):
    checkpoint=ModelCheckpoint('TrainedModels/Modelv1[EP-{epoch:03d},valloss-{val_loss.07f}].h5',
                              monitor='val_loss',
                              verbose=1,
                              save_best_only=False,
                              mode='categorical_crossentropy')
    model.compile(loss='mean_squared_error',optimizer=Adam(lr=le-4),metrics=['accuracy'])
    model.fit_generator(batch_generator(X_train,y_train,BATCH_SIZE,True),
                       SETS,
                       EPOCHS,
                       max_q_size=24,
                       validation_data=batch_generator(DIR,X_test,y_test,BATCH_SIZE,False),
                       nb_val_samples=len(X_test)/10,
                       callbacks=[checkpoint],
                       verbose=1)
SETS = len(train_image_paths)/32
model = GenerateModel()
TrainModel(model, train_image_paths, train_image_labels, test_image_paths, test_image_labels)

