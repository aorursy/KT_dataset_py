# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

from tensorflow import keras
# https://medium.com/@divyanshuDeveloper/a-simple-animal-classifier-from-scratch-using-keras-61ef0edfcb1f

from PIL import Image

import cv2

print(os.listdir("../input/images/Images"))



def cargarCategoria (categoria, label, directorio = "train"):

    data = []

    labels = []

    path = "../input/images/Images/"+ directorio +"/" + categoria

    items = os.listdir(path)

    for item in items:

        imag = cv2.imread(path +"/"+ item)

        img_from_ar = Image.fromarray(imag, 'RGB')

        resized_image = img_from_ar.resize((50, 50))

        data.append(np.array(resized_image))

        labels.append(label)

    return (data, labels)

data = []

labels = [] 



def AgregarCategoria(categoria, label, directorio = "train"):

    (dataC, labelsC) = cargarCategoria(categoria,label, directorio)

    print (" catidad datos "+ categoria + ": "+ str(len(dataC)))

    return (dataC, labelsC)

    



#Murcielago

(dataC, labelsC) = AgregarCategoria("bat",0)

data += dataC

labels += labelsC

#Castor

(dataC,labelsC) = AgregarCategoria("beaver",1)

data += dataC

labels += labelsC

#Hipopotamo

(dataC,labelsC) = AgregarCategoria("hippopotamus",2)

data += dataC

labels += labelsC

#Caballo

(dataC,labelsC) = AgregarCategoria("horse",3)

data += dataC

labels += labelsC

#Ardilla

(dataC,labelsC) = AgregarCategoria("squirrel",4)

data += dataC

labels += labelsC

print("cantidad datos Final " + str(len(data)))



animals=np.array(data)

labels=np.array(labels)
num_classes=len(np.unique(labels))

data_length=len(animals)



print("Cantidad de datos " + str(data_length))

print("Cantegorias " + str(num_classes))
# import sequential model and all the required layers

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout



model = Sequential()

model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(500,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(5,activation="softmax"))

model.summary()
from keras.utils import np_utils



animals = animals.astype('float32')/255

#One hot encoding

labels=keras.utils.to_categorical(labels,5)



print(labels)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model.fit(animals,labels,batch_size=50 ,epochs=100,verbose=1)
dataTest = []

labelsTest = []



#Murcielago

(dataC, labelsC) = AgregarCategoria("bat",0, "val")

dataTest += dataC

labelsTest += labelsC

#Castor

(dataC,labelsC) = AgregarCategoria("beaver",1, "val")

dataTest += dataC

labelsTest += labelsC

#Hipopotamo

(dataC,labelsC) = AgregarCategoria("hippopotamus",2, "val")

dataTest += dataC

labelsTest += labelsC

#Caballo

(dataC,labelsC) = AgregarCategoria("horse",3, "val")

dataTest += dataC

labelsTest += labelsC

#Ardilla

(dataC,labelsC) = AgregarCategoria("squirrel",4, "val")

dataTest += dataC

labelsTest += labelsC

print("cantidad datos Final " + str(len(dataTest)))



animalsTest=np.array(dataTest)

labelsTest=np.array(labelsTest)



animalsTest = animalsTest.astype('float32')/255

labelsTest=keras.utils.to_categorical(labelsTest,5)



print(labelsTest)



score = model.evaluate(animalsTest, labelsTest, verbose=1)

print('\n', 'Test accuracy:', score[1])
model.save("clasificador_animales.h5")