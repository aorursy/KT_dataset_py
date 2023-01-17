



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

      

#import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt

import os

import cv2



#DATADIR = "/kaggle/input/datasetcats2"

DATADIR = "/kaggle/input/"

CATEGORIES=["datasetcats2", "datasetdogs2"]

print("inport ok")



IMG_SIZE=50

training_data=[]

def create_training_data():

    for category in CATEGORIES:

        path=os.path.join(DATADIR,category )

        class_num = CATEGORIES.index(category)

        print(path)

        for img in os.listdir(path):

            print(path)

            try:

                img_array=cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

                new_array=cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                training_data.append([new_array, class_num])

            except Exception as e:

                pass



create_training_data()

print("finished creating data")








print("hello")
# shuffle the data

# if all the cats are togtehr then all the dgs are togther wont work

import random

random.shuffle(training_data)

print("finshed shuffling")
X=[]

y=[]



for features, label in training_data:

    X.append(features)

    y.append(label)

    

X=np.array(X).reshape( -1, IMG_SIZE, IMG_SIZE, 1) 

print("ok")
X=[]

y=[]



for features, label in training_data:

    X.append(features)

    y.append(label)

    

X=np.array(X).reshape( -1, IMG_SIZE, IMG_SIZE, 1) 

print("ok")
X=[]

y=[]



for features, label in training_data:

    X.append(features)

    y.append(label)

    

X=np.array(X).reshape( -1, IMG_SIZE, IMG_SIZE, 1) 

print("ok")
import pickle



pickle_out=open("X.pickle","wb")

pickle.dump(X, pickle_out)

pickle_out.close()



pickle_out=open("y.pickle","wb")

pickle.dump(y, pickle_out)

pickle_out.close()

print("saved using pickle")
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import tensorflow as tf

from tensorflow.keras.datasets import cifar10

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D



import pickle



pickle_in = open("/kaggle/input/catordog/X.pickle","rb")

X = pickle.load(pickle_in)

pickle_in.close()



pickle_in = open("/kaggle/input/catordog/y.pickle","rb")

y = pickle.load(pickle_in)

pickle_in.close()

X = X/255.0



model = Sequential()



model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(256, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors



model.add(Dense(64))



model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3)

print("ok")