# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cv2

import time

import random 

import pickle 

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

from tensorflow.keras.callbacks import TensorBoard







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_dir ="/kaggle/input/cat-and-dog/training_set/training_set"

categories = ["dogs" ,"cats"]

for category in categories:

    path = os.path.join(data_dir,category)

    for img in os.listdir(path):

        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)

        plt.imshow(img_array, cmap="gray")

        break

        

    break
print(img_array.shape)
img_size = 50

new_array = cv2.resize(img_array, (img_size,img_size))

plt.imshow(new_array, cmap="gray")
training_data =[]



def create_training_data():

    for category in categories:

        path = os.path.join(data_dir,category)

        class_num = categories.index(category)

        for img in os.listdir(path):

            try:

                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)

                new_array = cv2.resize(img_array, (img_size,img_size))

                training_data.append([new_array, class_num])

            except Exception as e:

                pass

create_training_data()

       

print(len(training_data))
test_dir = "/kaggle/input/cat-and-dog/test_set/test_set"



def prepare(test_dir):

    for category in categories:

        path = os.path.join(test_dir,category)

        class_num = categories.index(category)

        for img in os.listdir(path):

            try:

                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)

                new_array = cv2.resize(img_array, (img_size,img_size))

                training_data.append([new_array, class_num])

            except Exception as e:

                pass

    return new_array.reshape(-1, img_size, img_size,1)

    
random.shuffle(training_data)
x =[]

y =[]
for features, label in training_data:

    x.append(features)

    y.append(label)

    

x =np.array(x).reshape(-1, img_size,img_size,1)
NAME ="Cats_vs_Dogs{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='log/{}'.format(NAME))

pickle_out = open("x.pickle","wb")

pickle.dump(x,pickle_out)

pickle_out.close()



pickle_out = open("y.pickle","wb")

pickle.dump(y,pickle_out)

pickle_out.close()
x = pickle.load(open("x.pickle","rb"))

y = pickle.load(open("y.pickle","rb"))



x=np.array(x/255.0)

y=np.array(y)


model = Sequential()



model.add( Conv2D(64, (3,3), input_shape = x.shape[1:]) )

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2,2) ) )



model.add(Conv2D(64, (3,3)))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())



model.add(Dense(64))

model.add(Activation("relu"))

model.add(Dropout(0.2))



model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss="binary_crossentropy",

         optimizer="adam",

         metrics=['accuracy'])



model.fit(x, y, batch_size=32, epochs=15, validation_split=0.1, callbacks=[tensorboard])
model.save('cat-and-dog_cnn')
pred = model.predict([prepare(test_dir)])
print(categories[int(pred[0][0])])