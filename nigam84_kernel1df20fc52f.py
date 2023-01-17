# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


import numpy as np

from keras.models import Sequential

from keras.layers import Conv2D,MaxPool2D,Dropout,Dense,Flatten

from keras.preprocessing import image

import cv2

from keras.utils import np_utils

from sklearn.utils import shuffle

import glob

import pandas as pd

import os

import csv

import matplotlib.pyplot as plt

# %matplotlib inline 



#importing dataset and dividing them in training and testing data

sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test_set = pd.read_csv("../input/digit-recognizer/test.csv")

train_set = pd.read_csv("../input/digit-recognizer/train.csv")



Y=train_set['label']

X=train_set.drop(columns='label')

X=np.array(X).reshape(-1,28,28)

Y=np.array(Y).reshape(-1,1)

X_test=np.array(test_set).reshape(-1,28,28)





image_index = 0 # You may select anything up to 42,000

print(Y[image_index]) # The label is 1

plt.imshow(X[image_index], cmap='Greys')





from sklearn.model_selection import train_test_split

X_train,X_cv,y_train,y_cv=train_test_split(X,Y,test_size=0.2,random_state=0)





y_train=np_utils.to_categorical(y_train,num_classes=10)

y_cv=np_utils.to_categorical(y_cv,num_classes=10)





# Reshaping the array to 4-dims so that it can work with the Keras API

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_cv = X_cv.reshape(X_cv.shape[0], 28, 28, 1)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

input_shape = (28, 28, 1)

# Making sure that the values are float so that we can get decimal points after division

X_train = X_train.astype('float32')

X_cv = X_cv.astype('float32')

X_test = X_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.

X_train /= 255

X_cv /= 255

X_test /= 255

print('x_train shape:', X_train.shape)

print('Number of images in x_train', X_train.shape[0])

print('Number of images in x_cv', X_cv.shape[0])

print('Number of images in x_test', X_test.shape[0])





# creating the model

model=Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3),activation="relu",input_shape=(28,28,1)))

model.add(Conv2D(filters=32,kernel_size=(3,3),activation="relu"))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=16,kernel_size=(3,3),activation="relu"))

model.add(Conv2D(filters=16,kernel_size=(3,3),activation="relu"))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters=64,kernel_size=(3,3),activation="relu"))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512,activation="relu"))

model.add(Dropout(0.25))

model.add(Dense(256,activation="relu"))

model.add(Dropout(0.25))

model.add(Dense(128,activation="relu"))

model.add(Dropout(0.25))

model.add(Dense(64,activation="relu"))

model.add(Dropout(0.25))

model.add(Dense(10,activation="softmax"))

model.summary()



#compiling above model

model.compile(optimizer="RMSprop",loss="categorical_crossentropy",metrics=["accuracy"])

#fitting the model

model.fit(x=X_train,y=y_train,batch_size=125,epochs=20,validation_data=(X_cv,y_cv))



#saving model into disk

from keras.models import model_from_json

model_json = model.to_json()

with open("mnist_model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("mnist_weights.h5")

print("Saved model to disk")



#list storing all the prediction for test dataset

all_predictions=[]

for i in range(0,28000):

  predictions=model.predict(X_test[i].reshape(1,28,28,1))

  prediction=predictions.argmax()

  all_predictions.append(prediction)



#storing all prediction in the form of csv file 



with open('submission_final.csv', 'w', newline='') as file:

    writer = csv.writer(file)

    writer.writerow(["ImageId", "Label"])

    for i in range(0,28000):

      writer.writerow([i+1,all_predictions[i]])

    print("saved csv file")