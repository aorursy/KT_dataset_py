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
import cv2

import os

from tqdm import tqdm as tqdm

import matplotlib.pyplot as plt

V1=[]

V2=[]

        

for filename in tqdm(os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL')):

    img = cv2.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/'+filename, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (800, 800))

    V1.append([img, 0])



for filename in tqdm(os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA')):

    img = cv2.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/'+filename, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (800, 800))

    V2.append([img, 1])

    

images = V1 + V2[0:1341]



from random import shuffle





shuffle(images)



train = np.save('train_data.npy',images)
from keras.models import Sequential

from keras.layers import Dense, Flatten, Dropout

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

from keras import optimizers

from keras.utils import to_categorical

from keras import models, layers

import keras

import numpy as np





train_data = np.load('/kaggle/working/train_data.npy', allow_pickle=True)



def split_data(data, IMG_SIZE):

    images = np.array([i[0] for i in data]).reshape(-1, IMG_SIZE, IMG_SIZE,1)

    labels = np.array([i[1] for i in data])



    return images, labels



train, labels = split_data(train_data, 800)



def simple_NN():

    model = Sequential()

    

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=(800,800,1)))

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

    



    #model.add(Conv2D(32, kernel_size=(3, 3), strides=(1,1), activation='tanh', input_shape=(200,200,1), padding="same"))

    #model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))





    #model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='tanh', padding='valid'))

    #model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))





    #model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='tanh', padding='valid'))

    

    model.add(Flatten())

    model.add(Dense(84, activation='relu'))

    model.add(Dense(2, activation='sigmoid'))



    return model







model = simple_NN()

              

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])              



model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"]) 

history = model.fit(train, to_categorical(labels),validation_split=0.2, epochs=10, batch_size=1, verbose=1)              



model.summary()

model.save('XRay.h5')
import cv2

import os

from tqdm import tqdm as tqdm

import matplotlib.pyplot as plt

V1=[]

V2=[]

        

for filename in tqdm(os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/val/NORMAL')):

    img = cv2.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/val/NORMAL/'+filename, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (800, 800))

    V1.append([img, 0])





for filename in tqdm(os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA')):

    img = cv2.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA/'+filename, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (800, 800))

    V2.append([img, 1])

    

imagestest = V1 + V2

from random import shuffle





shuffle(imagestest)

from keras.models import load_model





def split_data(data, IMG_SIZE):

    images = np.array([i[0] for i in data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    labels = np.array([i[1] for i in data])



    return images, labels



images, labels = split_data(imagestest, 800)







test = np.save('val_data.npy',images,labels)



model = load_model('XRay.h5')

model.summary()



#newpredict = []



#for i in range (0, len(images)):

    

#    img = images[i].reshape(-1,400,400,1)

#    predictions = model.predict(img, batch_size=1)

    

#    if predictions[0][0] > predictions[0][1]:

        

#        newpredict.append(0 == label[i])

        

        

#    else :

#        newpredict.append(1 == label[i])



#print(newpredict)

#np.load()

### import matplotlib.pyplot as plt 



newpredict = []

newimg=[]

result=[]



for i in range (0, len(images)):

    img = images[i].reshape(-1,800,800,1)

    predictions = model.predict(img, batch_size=1)

    

    # VOIR SI ELLES SONT BONNES

    

    

    if predictions[0][0] > predictions[0][1]:

        newpredict.append(0)

        

    else : 

        newpredict.append(1)

        

result = newpredict==labels

    

#for i in range (0, len(new)):

    

 #   img = images[i].reshape(-1,400,400,1)

  #  newimg.append(img)



#print(newimg[3])



#plt.imshow(newimg[3])





#print(predictions[0])



#for i in range (0,len(predictions)):

    

#    if predictions[i][0] < predictions[i][1]:

        

#        newpredict.append(1)

        

#    else :

#        newpredict.append(0)

            

        

#print(newpredict)
unique_elements, counts_elements = np.unique(result, return_counts=True)

print(unique_elements)

print(counts_elements)