# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import cv2

import os



label=[]



for filename in os.listdir('/kaggle/input/dogs-vs-cats/train/train'):

    if filename.split('.')[0] == 'dog':

        label.append(0)

        

    elif filename.split('.')[0] == 'cat':

        label.append(1)

 



from tqdm import tqdm

Z=[]

  

for filename in tqdm(os.listdir('/kaggle/input/dogs-vs-cats/train/train')):

    img = cv2.imread('/kaggle/input/dogs-vs-cats/train/train/'+filename)

    nimg =cv2.resize(img, (100, 100))

    Z.append(nimg)









print('/kaggle/input/dogs-vs-cats/train/train/'+filename)



               

 # for filename in tqdm(os.listdir('/kaggle/input/dogs-vs-cats/train/train')):

           # ('/kaggle/input/dogs-vs-cats/train/train/'+filename)
C = [Z, label]

C = np.transpose(C)

train = np.save('train_data.npy',C)
from keras.models import Sequential

from keras.layers import Dense, Flatten, Dropout

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

from keras import optimizers

from keras.utils import to_categorical

from keras import models, layers

import keras

import numpy as np





train_data = np.load('train_data.npy', allow_pickle=True)



def split_data(data, IMG_SIZE):

    images = np.array([i[0] for i in data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

    labels = np.array([i[1] for i in data])

    images = images / 255



    return images, labels



train, labels = split_data(train_data, 100)



def simple_NN():

    model = Sequential()

    

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=(100,100,3)))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.2))

    

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.2)) 

    

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.2)) 

    

    

    #model.add(Flatten())

    

    #model.add(Dense(64, activation='relu'))

    #model.add(Dropout(0.5))

    

    #model.add(Dense(2, activation='sigmoid'))

    



    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1,1), activation='tanh', input_shape=(100,100,3), padding="same"))

    model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))





    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='tanh', padding='valid'))

    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))





    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='tanh', padding='valid'))

    

    model.add(Flatten())

    model.add(Dense(84, activation='tanh'))

    model.add(Dense(2, activation='softmax'))



    return model







model = simple_NN()

              

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])              



model.compile(loss="binary_crossentropy", optimizer='SGD', metrics=["accuracy"]) 

history = model.fit(train, to_categorical(labels),validation_split=0.2, epochs=10, batch_size=8, verbose=1)              



model.summary()

model.save('catsDogs.h5')
import cv2

import os



Z=[]

 

for filename in tqdm(os.listdir('/kaggle/input/dogs-vs-cats/test/test/')):

    img = cv2.imread('/kaggle/input/dogs-vs-cats/test/test/'+filename)

    nimg =cv2.resize(img, (100, 100))

    Z.append(nimg)

    

    


test = np.save('test_data.npy',Z)
from keras.models import load_model



model = load_model('catsDogs.h5')

model.summary()
Z = np.array([i for i in Z]).reshape(-1,100,100,3)

predictions = model.predict(Z, batch_size=1) 
print(predictions)
import matplotlib.pyplot as plt 



plt.imshow(Z[0])



print(predictions[0])