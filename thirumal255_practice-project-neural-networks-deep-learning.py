# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

import h5py

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
# Reading the .h5 File

h5f = h5py.File('../input/SVHN_single_grey1.h5', 'r')



# Load the training, test and validation set

x_train = h5f['X_train'][:]

y_train = h5f['y_train'][:]

x_test = h5f['X_test'][:]

y_test = h5f['y_test'][:]





# Close this file

h5f.close()

# To Visualise images in the Data set choosen

%matplotlib inline



import matplotlib.pyplot as plt

w=10

h=10

fig=plt.figure(figsize=(10, 10))

columns = 10

rows = 10

for i in range(1, columns*rows +1):

    img = x_test[i]

    fig.add_subplot(rows, columns, i)

    plt.imshow(img, cmap='gray')

plt.show()
print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)

print(x_train.dtype)
#Importing opencv module for the resizing function

import cv2



#Create a resized dataset for training and testing inputs with corresponding size. Here we are resizing it to 28X28 (same input size as MNIST)

x_train_res = np.zeros((x_train.shape[0],28,28), dtype=np.float32)

for i in range(x_train.shape[0]):

  #using cv2.resize to resize each train example to 28X28 size using Cubic interpolation

  x_train_res[i,:,:] = cv2.resize(x_train[i], dsize=(28, 28), interpolation=cv2.INTER_CUBIC)



x_test_res = np.zeros((x_test.shape[0],28,28), dtype=np.float32)

for i in range(x_test.shape[0]):

  #using cv2.resize to resize each test example to 28X28 size using Cubic interpolation

  x_test_res[i,:,:] = cv2.resize(x_test[i], dsize=(28, 28), interpolation=cv2.INTER_CUBIC)

  

#We don't need the original dataset anynmore so we can clear up memory consumed by original dataset

del x_train

del x_test



print(x_train_res.shape)

print(x_test_res.shape)
y_test
# To Visualise images in the Data set choosen after Re shape

%matplotlib inline



import matplotlib.pyplot as plt

w=10

h=10

fig=plt.figure(figsize=(10, 10))

columns = 10

rows = 10

for i in range(1, columns*rows +1):

    img = x_test_res[i]

    fig.add_subplot(rows, columns, i)

    plt.imshow(img, cmap='gray')

plt.show()
# Reshaping Input into 2 Dimensional. As the ML algorithms take only 2 Dimensional

x_train_ml = x_train_res.reshape(x_train_res.shape[0], 28*28).astype('float32')

x_test_ml = x_test_res.reshape(x_test_res.shape[0], 28*28).astype('float32')
x_test_ml[0]
#Normalizing the input

x_train_ml /= 255

x_test_ml /= 255
# Checking with K=5 and euclidean_distance 

from sklearn.neighbors import KNeighborsClassifier

NNH = KNeighborsClassifier(n_neighbors= 5 , weights = 'distance',n_jobs = -1 )

NNH.fit(x_train_ml, y_train)
# For every test data point, predict it's label based on 5 nearest neighbours in this model. The majority class will 

# be assigned to the test data point



predicted_labels = NNH.predict(x_test_ml)

NNH.score(x_test_ml, y_test)
#Checking Unique Lables of the Predicted output ones and Train Output

print('Number of Classes in Input Test set:',np.unique(y_train))

print('Number of Classes in Predicted Test Output:',np.unique(predicted_labels))
# Checking with K=5 and manhattan_distance 

NNH = KNeighborsClassifier(n_neighbors= 5 ,p=1, weights = 'distance',n_jobs = -1 )

NNH.fit(x_train_ml, y_train)

predicted_labels = NNH.predict(x_test_ml)

NNH.score(x_test_ml, y_test)
# euclidean_distance seems better prediction

#So Checking with K=7 and euclidean_distance 

NNH = KNeighborsClassifier(n_neighbors= 7 ,p=2, weights = 'distance',n_jobs = -1 )

NNH.fit(x_train_ml, y_train)

predicted_labels = NNH.predict(x_test_ml)

NNH.score(x_test_ml, y_test)
#So Checking with K=7 and euclidean_distance 

NNH = KNeighborsClassifier(n_neighbors= 9 ,p=2, weights = 'distance',n_jobs = -1 )

NNH.fit(x_train_ml, y_train)

predicted_labels = NNH.predict(x_test_ml)

NNH.score(x_test_ml, y_test)


print(NNH.score(x_test_ml, y_test))

from sklearn import metrics

print(metrics.confusion_matrix(y_test, predicted_labels))
import keras

batch_size = 128

num_classes = 10

epochs = 20

# convert class vectors to binary class matrices ( Output label Conversion)

y_train_dl = keras.utils.to_categorical(y_train, num_classes)

y_test_dl = keras.utils.to_categorical(y_test, num_classes)

print(y_train_dl[0])
# input image dimensions

img_rows, img_cols = 28, 28



#Keras expects data to be in the format (N_E.N_H,N_W,N_C)  N_E = Number of Examples, N_H = height, N_W = Width, N_C = Number of Channels.

x_train_dl = x_train_res.reshape(x_train_res.shape[0], img_rows, img_cols, 1)

x_test_dl = x_test_res.reshape(x_test_res.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)

x_train_dl.shape
#Normalize both the train and test image data from 0-255 to 0-1



x_train_dl=x_train_dl/255

x_test_dl=x_test_dl/255
#Initialize the model 

# Train the Model with Adam Optimizer



model = Sequential()

#Reshape data from 3D to 1D -> 28x28 to 784

model.add(keras.layers.Reshape((784,),input_shape=(28,28,1)))

#Hidden layer1

model.add(keras.layers.Dense(784, activation='relu', name='Layer_1'))

#Hidden layer2

model.add(keras.layers.Dense(500, activation='relu', name='Layer_2'))



#Hidden layer3

model.add(keras.layers.Dense(300, activation='relu', name='Layer_3'))



#Hidden layer4

model.add(keras.layers.Dense(100, activation='relu', name='Layer_4'))

#Output layer

model.add(keras.layers.Dense(10, activation='softmax', name='Output'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



#Train the model

model.fit(x_train_dl,y_train_dl,          

          validation_data=(x_test_dl,y_test_dl),

          epochs=epochs,

          batch_size=batch_size)

#Testing the model on test set

score = model.evaluate(x_test_dl, y_test_dl)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
# Create the Same above DL network with SGD Optimizer 

from keras import optimizers

model2 = Sequential()





#Reshape data from 3D to 1D -> 28x28 to 784

model2.add(keras.layers.Reshape((784,),input_shape=(28,28,1)))

#Hidden layer1

model2.add(keras.layers.Dense(784, activation='relu', name='Layer_1'))

#Hidden layer2

model2.add(keras.layers.Dense(500, activation='relu', name='Layer_2'))



#Hidden layer3

model2.add(keras.layers.Dense(300, activation='relu', name='Layer_3'))



#Hidden layer4

model2.add(keras.layers.Dense(100, activation='relu', name='Layer_4'))

#Output layer

model2.add(keras.layers.Dense(10, activation='softmax', name='Output'))



#To use SGD optimizer for learning weights with learning rate = 0.03

sgd = optimizers.SGD(lr=0.03, decay=1e-6, momentum=0.9)

#Set the loss function and optimizer for the model training

model2.compile(loss='categorical_crossentropy',

              optimizer=sgd,

              metrics=['accuracy'])

# Training the Model

model2.fit(x_train_dl, y_train_dl,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(x_test_dl, y_test_dl))
#Testing the model on test set

score = model2.evaluate(x_test_dl, y_test_dl)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
# Create the Same above DL network with Batch Normalisation and Adam Optimizer 

from keras.layers import BatchNormalization

from keras.optimizers import Adam

from keras.losses import categorical_crossentropy



model3 = Sequential()



#Reshape data from 3D to 1D -> 28x28 to 784

model3.add(keras.layers.Reshape((784,),input_shape=(28,28,1)))



#Normalize the data

model3.add(keras.layers.BatchNormalization())



#Hidden layer1

model3.add(keras.layers.Dense(784, activation='relu', name='Layer_1'))

#Hidden layer2

model3.add(keras.layers.Dense(500, activation='relu', name='Layer_2'))



#Apply Dropout with 0.5 probability 

model3.add(Dropout(0.5,name='drop_1'))



#Hidden layer3

model3.add(keras.layers.Dense(300, activation='relu', name='Layer_3'))



#Hidden layer4

model3.add(keras.layers.Dense(100, activation='relu', name='Layer_4'))

#Output layer

model3.add(keras.layers.Dense(10, activation='softmax', name='Output'))





#To use adam optimizer for learning weights with learning rate = 0.001

adamopt = Adam(lr=0.001)



#Set the loss function and optimizer for the model training

model3.compile(loss=categorical_crossentropy,

              optimizer=adamopt,

              metrics=['accuracy'])

# Training the Model

model3.fit(x_train_dl, y_train_dl,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(x_test_dl, y_test_dl))
#Testing the model on test set

score = model3.evaluate(x_test_dl, y_test_dl)

print('Test loss:', score[0])

print('Test accuracy:', score[1])