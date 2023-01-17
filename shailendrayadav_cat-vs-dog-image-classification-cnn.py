# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import cv2

import matplotlib.pyplot as plt

%matplotlib inline



# Any results you write to the current directory are saved as output.
#creating training data

files = os.listdir("../input/train")

img_size=80

training_data =[]



for i in files:

    path=os.path.join("../input/train",i)

    

    #for label as cat or dog

    name=i.split(".")[0]

    if name=="cat":

        label=0

    else:

        label=1

    #reading images and resize using imread and resize and gray scale is forblack n white images   

    images=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(img_size,img_size)) #resize the image

    training_data.append([np.array(images),np.array(label)])

    np.random.shuffle(training_data)

    np.save("cat_dog_traing_data.npy",training_data)

        
training_data=np.array(training_data)

training_data.shape
#dividing the data into images and label for the neural network

X= np.array([i[0] for i in training_data])

y=np.array([i[1] for i in training_data])

#scale the data

X=X/255

X=X.reshape(-1,80,80,1)

X.shape,y.shape
import tensorflow as tf

from keras.models import Sequential 

from keras import layers 

from keras.layers import Conv2D,MaxPool2D,Dense,Softmax,Flatten
X.shape
y.shape
from keras.layers import Dropout

from keras.optimizers import Adam

opt=Adam(lr=.0001)

model=Sequential()

model.add(Dense(16,activation="relu",input_shape=(80,80,1)))

model.add(Conv2D(32,(3,3)))#image size with color/bw

model.add(MaxPool2D(2,2))



model.add(Conv2D(32,(3,3)))#image size with color/bw

model.add(MaxPool2D(2,2))





model.add(Flatten())

model.add(Dense(50,activation="relu"))

model.add(Dropout(0.25))

model.add(Dense(50,activation="relu"))

model.add(Dropout(0.25))

model.add(Dense(1,activation="sigmoid"))



model.compile(optimizer=opt,loss="binary_crossentropy",metrics=["accuracy"])

model.summary()

                 
model.fit(X,y,epochs=30,validation_split=0.30)
#creating testing data

testing_data =[]

for imgs in os.listdir("../input/test"):

        path=os.path.join("../input/test",imgs)

        #for label as cat or dog

        name=i.split(".")[0]

        if name=="cat":

            label=0

        else:

            label=1

    

        images=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(img_size,img_size))

        testing_data.append([np.array(images),np.array(label)])

        np.random.shuffle(testing_data)

        np.save("cat_dog_testing_data.npy",testing_data)

        
#dividing the data into images and label for the neural network

X_test= np.array([i[0] for i in testing_data]).reshape(-1,80,80,1)

y_test=np.array([i[1] for i in testing_data])

X_test=X_test/255
X_test.shape,y_test.shape
y_pred=model.predict(X_test)
model.save("Imageclassfication.h5py")
X_image_11=X_test[11].reshape(80,80)

X_image_11.shape

plt.imshow(X_image_11)
y_test[11]  