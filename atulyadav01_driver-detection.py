# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import keras

import matplotlib.pyplot as plt

from keras import Sequential

from keras.layers import Convolution2D, Conv2D , MaxPooling2D , Flatten , Dropout , BatchNormalization , Dense

from keras.activations import relu

from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
train_dir = os.listdir("../input/train/")

print(train_dir)

data_folder = ['c0' , 'c1' , 'c2' , 'c3' , 'c4' , 'c5' ,'c6' , 'c7' , 'c8' , 'c9']
training_data = []

index_label = 0

for folder in data_folder:

    path = os.path.join("../input/train/" + folder)

    img = os.listdir(path)

    index_label = data_folder.index(folder)

    for i in range(len(img)):

        read_img = cv2.imread(os.path.join(path + "/" + img[i]) , cv2.IMREAD_GRAYSCALE)

        #reshape_img = cv2.resize(read_img , (240*240 , 1))

        reshape_img = cv2.resize(read_img , (1 , 240*240))

        training_data.append([

                reshape_img,index_label])
X = []

y = []

for features , label in training_data:

    X.append(features)

    y.append(label)

X = np.array(X)

y = np.array(y)

y = y.reshape([X.shape[0] , 1])

print(X.shape)

print(y.shape)
# dimesnions of images are a row vector we need to convert it into 240 ,240 , 1 to pass into cnn

X = X.reshape(-1 , 240 , 240 ,1)

X.shape
# one hot encoded

y = to_categorical(y)



# split the data into train and test

X_train , X_val , y_train , y_val  = train_test_split(X , y , test_size = 0.25)

print("Train Data Shape {} , Train Label Shape {} ".format(X_train.shape , y_train.shape))

print("Test Data Shape {} , Test Label Shape {} ".format(X_val.shape , y_val.shape))

# define model

def cnn():

    model = Sequential([

        

        Convolution2D(32,kernel_size = (3,3), strides = (1,1), activation='relu' , input_shape=(240,240,1)),

        BatchNormalization(),

        Convolution2D(32 , kernel_size=(3,3) , strides = (1,1), activation='relu' , padding = 'SAME'),

        BatchNormalization(),

        MaxPooling2D(pool_size = (2,2) , strides = (2,2) , padding = 'SAME'),

        Dropout(0.3), 

        

        Convolution2D(64 , kernel_size=(3,3) , strides = (1,1), activation='relu' , padding = 'SAME'),

        BatchNormalization(),

        Convolution2D(64 , kernel_size = (3,3) , strides = (1,1) , activation = 'relu' , padding = 'SAME'),

        BatchNormalization(),

        MaxPooling2D(pool_size = (2,2) , strides = (2,2) , padding = 'SAME'),

        Dropout(0.3),

        

        Convolution2D(128 ,kernel_size=(3,3), strides = (1,1),activation='relu', padding = 'SAME'),

        BatchNormalization(),

        Convolution2D(128 , kernel_size = (3,3) , strides = (1,1) , activation = 'relu' , padding = 'SAME'),

        BatchNormalization(),

        MaxPooling2D(pool_size = (2,2) , strides = (2,2) , padding = 'SAME'),

        Dropout(0.5),

        

        Flatten(),

        

        Dense(512 , activation = 'relu'),

        BatchNormalization(),

        Dropout(0.5),

        Dense(128 ,activation = 'relu'),

        Dropout(0.25),

        Dense(10 , activation = 'softmax')

        

    ])

    

    model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

    

    return model
classifier = cnn()

print(classifier.summary())
history = classifier.fit(X_train , y_train , batch_size = 50 , epochs = 20 , validation_data = (X_val , y_val) , verbose = 1) 
plt.plot(history.history['loss'] , 'green' , label = 'Training loss')

plt.plot(history.history['val_loss'] , 'red' , label = "Validation loss")

plt.legend()

plt.plot(history.history['acc'] , 'blue' , label = "Training Accuracy")

plt.plot(history.history['val_acc'] , 'orange' , label = "Validation Accuracy")

plt.legend()