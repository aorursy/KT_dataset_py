# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import keras

from keras.callbacks import ReduceLROnPlateau

from keras import layers

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPool2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.models import Model

from keras.models import Sequential

from keras.preprocessing import image

from keras.utils import layer_utils

from keras import regularizers

from keras.utils.data_utils import get_file

from keras.applications.imagenet_utils import preprocess_input

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model

from keras.utils import to_categorical

from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

import keras.backend as K

import tensorflow as tf

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator

K.set_image_data_format('channels_last')

from matplotlib.pyplot import imshow

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Cargamos los Datos y verificamos

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

df = train.copy()

df_test = test.copy()
print("Train: ", df.shape)

print("Test: ", df_test.shape)
#Verificamos que no existan valores nulos en el train

df.isnull().any().sum()
#Verificamos que no existan valores nulos en el test

df_test.isnull().any().sum()
#Conversión de el Data Frame a una matriz numpy

x = train.drop(labels = ["label"],axis = 1)

y = train["label"]
#Hacems One Hot Encoding

# y=to_categorical(y,10) 
#Verificamos que se lean los numeros, podemos cambiar el numero de muestra alterando el numero dentro de 

x = np.array(x)

e =x[10005]

image = e.reshape(28,28)

plt.imshow(image, cmap = plt.cm.binary,

           interpolation = 'nearest')

plt.axis('off')

plt.show()

print(y[10005])
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.2, random_state = 123)



xtrain.shape, xtest.shape, ytrain.shape, ytest.shape
xtrain = xtrain.reshape(-1, 28, 28, 1)

xtest = xtest.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1,28,28,1)
#Normalizamos los datos

xtrain = xtrain.astype("float32")/255

xtest = xtest.astype("float32")/255

test = test.astype("float32")/255

ytrain = to_categorical(ytrain, num_classes=10)

ytest = to_categorical(ytest, num_classes=10)
#Modelo Convolutional Neural Network usando Keras

model = Sequential()



model.add(Conv2D(filters = 32,

    kernel_size = (3,3),padding = 'Same', activation ='relu', 

    input_shape = (28,28,1)))

model.add(BatchNormalization())



model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(BatchNormalization())



model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(BatchNormalization())



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same',  activation ='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(BatchNormalization())

model.add(Dropout(0.25))



model.add(Dense(10, activation = "softmax"))
# Optimizamos

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999 )

# Compilamos el modelo

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])



model.summary()
datagen = ImageDataGenerator(

    # establecer la media de entrada en 0 sobre el conjunto de datos

        featurewise_center=False, 

        samplewise_center=False,  

        featurewise_std_normalization=False, 

        samplewise_std_normalization=False, 

        zca_whitening=False, 

    # rotar aleatoriamente las imágenes en el rango de 0 a 180 grados

        rotation_range=10, 

    # Ampliar imagen al azar

        zoom_range = 0.1,

    # desplazar aleatoriamente las imágenes verticalmente y horizontalmente

        width_shift_range=0.1,  

        height_shift_range=0.1, 

    # voltear imágenes al azar

        horizontal_flip=False,  

        vertical_flip=False) 

datagen.fit(xtrain)
batch_size = 128

epochs = 50

reduce_lr = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)



history = model.fit_generator(datagen.flow(xtrain, ytrain, batch_size = batch_size), epochs = epochs, 

                              validation_data = (xtest, ytest), verbose=2, 

                              steps_per_epoch=xtrain.shape[0] // batch_size,

                              callbacks = [reduce_lr])
model.evaluate(xtest, ytest)
model.evaluate(xtrain, ytrain)
#Graficamos

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(['Train','Test'])

plt.show()
#Graficamos

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
#Cargamos los datos

testprediction=np.argmax(model.predict(test),axis=1)

testimage=[]

for i in range (len(testprediction)):

    testimage.append(i+1)

final={'ImageId':testimage,'Label':testprediction}

submission=pd.DataFrame(final)

submission.to_csv('submssion.csv',index=False)