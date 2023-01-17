# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
import keras
import glob
import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
import shutil

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!ls ../input/aptos-20152019-preprocessed
path='../input/aptos-20152019-preprocessed'
train = pd.read_csv(path+'/trainData.csv')

image = load_img(path+"/trainImgs/trainImgs/44e951e45dca.png")

plt.imshow(image)
image.mode
print(train.head())
train.shape
train.isnull().sum().sum()
train.hist(bins=50,figsize=(10,5))
plt.show()
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, GaussianNoise, GaussianDropout
from keras.layers import Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D, SeparableConv2D, AveragePooling2D
from keras.constraints import maxnorm
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras import regularizers, optimizers
from sklearn.model_selection import train_test_split
train['diagnosis'] =  train['diagnosis'].astype(str)
train['diagnosis'] =  train['diagnosis'].astype('string')
train['id_code'] =  train['id_code'].astype(str)+'.png'
X=train['id_code']
Y=train['diagnosis']
Y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


datagen=ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    vertical_flip=True,
    horizontal_flip=True)

image_size=100
batch_size=40
train_gen=datagen.flow_from_dataframe(
    dataframe=train,
    directory=path+"/trainImgs/trainImgs",
    x_col="id_code",
    y_col="diagnosis",
    batch_size=batch_size,
    shuffle=True,
    class_mode="categorical",
    target_size=(image_size,image_size),
    subset='training')

test_gen=datagen.flow_from_dataframe(
    dataframe=train,
    directory=path+"/trainImgs/trainImgs",
    x_col="id_code",
    y_col="diagnosis",
    batch_size=batch_size,
    shuffle=True,
    class_mode="categorical", 
    target_size=(image_size,image_size),
    subset='validation')
y_train = train['diagnosis']
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
num_classes = y_train.shape[1]
y_train
def build_model():
    # create model
    model = Sequential()
    #model.add(Reshape((x_train.shape[0],),))
    #model.add(GaussianDropout(0.3,input_shape=[96,96,3]))
    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (100,100,3)))
    model.add(GaussianDropout(0.3))
    model.add(Conv2D(64, (5, 5), activation='relu', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(96, (5, 5), activation='relu'))
    
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.0001)
                   ,activity_regularizer=regularizers.l1(0.01)))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.adam(lr=0.0001, amsgrad=True), metrics=['accuracy'])
    return model
def build_model2():
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',activation ='relu', input_shape = (image_size,image_size,3)))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu',kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    #kernel_constraint=maxnorm(3)
    model.add(Conv2D(filters =64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(filters = 92, kernel_size = (3,3),padding = 'Same',activation ='relu',kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2))) #avrage before the last layer
    
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(5, activation = "softmax",kernel_regularizer=regularizers.l2(0.0001),activity_regularizer=regularizers.l1(0.01)))
    #,kernel_regularizer=regularizers.l2(0.0001),activity_regularizer=regularizers.l1(0.005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.adam(lr=0.0001, amsgrad=True), metrics=['accuracy'])
    return model
def build_model3():
    # create model
    
    model = Sequential()

    model.add(Conv2D(filters = 30, kernel_size = (5, 5), input_shape = (100, 100, 3), activation = 'relu'))
    model.add(Conv2D(filters = 30, kernel_size = (3, 3), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters = 46, kernel_size = (3, 3), activation = 'relu',kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 46, kernel_size = (3, 3), activation = 'relu',kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2))) 
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))

    model.add(Dense(5, activation = 'softmax'))
    #kernel_regularizer=regularizers.l2(0.0001),activity_regularizer=regularizers.l1(0.01)
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.adam(lr=0.00006, amsgrad=True), metrics = ['categorical_accuracy'])

    return model
model = build_model()
model.summary()
from keras.callbacks import EarlyStopping, ModelCheckpoint
es= EarlyStopping(monitor='val_loss', mode ='min', verbose = 1, patience = 20)
mc = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only = True, mode ='min', verbose = 1)
model22=model.fit_generator(generator=train_gen,              
                                    steps_per_epoch=len(train_gen),
                                    validation_data=test_gen,                    
                                    validation_steps=len(test_gen),
                                    epochs=20,
                                    callbacks = [es,mc], 
                                    use_multiprocessing = True,
                                    verbose=1)
plt.plot(model22.history['accuracy'])
plt.plot(model22.history['val_accuracy'])
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

datagen=ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    vertical_flip=True,
    horizontal_flip=True)

example_df = train.sample(n=1).reset_index(drop=True)
image_size=100
example_generator = datagen.flow_from_dataframe(
    example_df, 
    path+"/trainImgs/trainImgs", 
    x_col='id_code',
    y_col='diagnosis',
     batch_size=batch_size,
    shuffle=True,
    target_size=(image_size,image_size),
    class_mode='categorical',
    
)
plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()