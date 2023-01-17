# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/exzeodata/sample/train"))



# Any results you write to the current directory are saved as output.
import sys

import os

from os.path import isfile, join

from os import listdir

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers

from keras.models import Sequential, Model

from keras.layers import Dropout, Flatten, Dense, Activation

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras import callbacks

import pandas as pd

import numpy as np

import keras

import matplotlib.pyplot as plt

from keras.layers import Dense, GlobalAveragePooling2D

from keras.applications import MobileNet

from keras.preprocessing import image

from keras.applications.mobilenet import preprocess_input

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model

from keras.optimizers import Adam

from keras.applications.resnet50 import ResNet50

from keras.preprocessing import image

from keras.preprocessing.image import load_img, img_to_array

from keras.applications.resnet50 import preprocess_input

from sklearn.metrics import accuracy_score, f1_score

from keras.optimizers import Adam, SGD
# dataset="../input/exzeodata/sample"

train_data_dir="../input/exzeodata/sample/train"

test="../input/exzeodata/sample/test"
img_width, img_height = 150, 150

nb_train_samples = 350

nb_validation_samples = 150

nb_filters1 = 32

nb_filters2 = 64

conv1_size = 3

conv2_size = 2

pool_size = 2

classes_num = 2

batch_size = 32

lr = 0.0001

epochs = 5
def img_to_tensor(image_path, target_size):

    img = load_img(image_path, target_size=target_size)

    tensor = img_to_array(img)

    tensor = np.expand_dims(tensor, axis=0)

    print("Image """ + str(image_path) +

          " "" converted to tensor with shape " + str(tensor.shape))

    return tensor
def getCustommodel():

    model1 = Sequential()

    model1.add(Conv2D(32, (3, 3), padding="same", input_shape=(img_width, img_height, 3), activation='relu'))

    model1.add(Conv2D(32, (3, 3), activation='relu'))

    model1.add(MaxPooling2D(pool_size=(2, 2)))

    model1.add(Dropout(0.2))



    model1.add(Conv2D(64, (3, 3), padding="same", activation='relu'))

    model1.add(Conv2D(64, (3, 3), activation='relu'))

    model1.add(MaxPooling2D(pool_size=(2, 2)))



    model1.add(Flatten())

    model1.add(Dense(512, activation='relu'))

    model1.add(Dropout(0.2))

    model1.add(Dense(2, activation='softmax'))



    model1.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), 

                  metrics=['accuracy'])

    model1.summary()

    

    return model1
m=getCustommodel()

m.summary()
def getpretrainedmodel():

    base_model = MobileNet(weights='imagenet', include_top=False)

    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)

    x = Dense(1024, activation='relu')(x)  # dense layer 2

    x = Dense(512, activation='relu')(x)  # dense layer 3

    preds = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=preds)

    for layer in model.layers[:20]:

        layer.trainable = False

    for layer in model.layers[20:]:

        layer.trainable = True

    model.compile(optimizer='Adam', loss='categorical_crossentropy',

                  metrics=['accuracy'])



    return model



p=getpretrainedmodel()

p.summary()
def datagenerators(mode="pretrained"):

    

    if mode=="custom":

        train_datagen = ImageDataGenerator(

            rescale=1. / 255,

            shear_range=0.2,

            zoom_range=0.2,

            horizontal_flip=True,

            validation_split=0.3

            )

    else:

        train_datagen = ImageDataGenerator(

            rescale=1. / 255,

            shear_range=0.2,

            zoom_range=0.2,

            horizontal_flip=True,

            validation_split=0.3,

        preprocessing_function=preprocess_input)



    train_generator = train_datagen.flow_from_directory(

        train_data_dir,

        target_size=(img_height, img_width),

        batch_size=batch_size,

        class_mode='categorical',

        subset="training")



    validation_generator = train_datagen.flow_from_directory(

        train_data_dir,

        target_size=(img_height, img_width),

        batch_size=batch_size,

        class_mode='categorical',

        subset="validation")



    return train_generator, validation_generator
def getTestImages(folder):

    a = []

    b = []

    for f in listdir(folder):

        temp = (join(folder, f))

        print(temp)

        b.append(f)

        a.append(img_to_tensor(temp, target_size=(img_width, img_height)))

    return (a, b)
(a,b)=getTestImages(test)

print(a)

print(b)
def ensemble_predictor(trained_ensemble, test_folder, method="vote"):



    test_tensors, test_images = getTestImages(test_folder)

    # test_images = np.array(test_images)

    # print(test_images.shape)



    p1 = []

    p3 = []

    output = {}

    if method == "average":

        for i in range(len(test_tensors)):

            t1 = trained_ensemble[0].predict(test_tensors[i])

            t2 = trained_ensemble[1].predict(test_tensors[i])

            tt = t2

            output[test_images[i]] = np.argmax(tt, axis=1)



    # for i in range(len(p1)):

        # if(np.argmax(p1[i], axis=1) > np.argmax(p2[i], axis=1)):

            # output[test_images[i]] = 1



    return output
model1 = getCustommodel()

model3 = getpretrainedmodel()



ensemble = []

ensemble.append(model1)

ensemble.append(model3)



trained_ensemble = []

for i in ensemble:

    a, b = datagenerators()

    i.fit_generator(

        a,

        samples_per_epoch=nb_train_samples,

        epochs=7,

        validation_data=b,

        validation_steps=nb_validation_samples)



    trained_ensemble.append(i)

output = ensemble_predictor(ensemble, test, method="average")

print(output)
predictions={}

for i,j in output.items():

    predictions[i]=j[0]
print(predictions)
df=pd.DataFrame()

df['filename']=predictions.keys()

df['label']= predictions.values()
print(df.head(3))
df.T.to_csv("output.csv")