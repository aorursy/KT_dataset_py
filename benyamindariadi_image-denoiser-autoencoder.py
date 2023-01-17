import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import os

import cv2

import os
data_dir= r'/kaggle/input/four-shapes/shapes' 
cat=['circle','square','star','triangle']
for category in cat: 

    path=os.path.join(data_dir,category)

    for img in os.listdir(path):

        img_array=cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) 

        plt.imshow(img_array) 

        break

    break 

    
print(img_array)
img_array.shape
new_array=cv2.resize(img_array,(20,20))

plt.imshow(new_array)

train=[] 

def create_training_data():

    for category in cat: 

        path=os.path.join(data_dir,category) 

        class_cat=cat.index(category)

        for img in os.listdir(path):

            try:

                img_array=cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) 

                new_array=cv2.resize(img_array,(20,20)) 

                train.append([new_array, class_cat])

            except Exception as e:

                pass

create_training_data() 
print(len(train))
x=[]

y=[]



for features, label in train: 

    x.append(features)

    y.append(label)

    
x[0] 
x=np.array(x).reshape(-1,20,20,1) 
x=x/255
x[0] 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=101)
#Example to make the noises in the images

from tensorflow.keras.layers import GaussianNoise

sample = GaussianNoise(0.2) 

noisey = sample(x_test[0].reshape(20,20),training=True) 

plt.imshow(noisey) 

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Flatten,Reshape

from tensorflow.keras.optimizers import SGD

import tensorflow as tf







#Encoder sequence

encoder = Sequential()

encoder.add(Flatten(input_shape=[20,20,1]))

# Add noise to images before going through autoencoder

encoder.add(GaussianNoise(0.2))   

encoder.add(Dense(200,activation="relu"))

encoder.add(Dense(100,activation="relu"))

encoder.add(Dense(50,activation="relu"))

encoder.add(Dense(25,activation="relu"))



#Decoder sequence

decoder = Sequential()

decoder.add(Dense(50,input_shape=[25],activation='relu'))

decoder.add(Dense(100,activation='relu'))

decoder.add(Dense(200,activation='relu'))

decoder.add(Dense(20 * 20, activation="sigmoid"))

decoder.add(Reshape([20, 20]))



noise_remover = Sequential([encoder, decoder])

noise_remover.compile(loss="binary_crossentropy", optimizer='adam',metrics=['accuracy'])

noise_remover.fit(x_train, x_train, epochs=10, validation_data=[x_test, x_test])
one_noisey_images = sample(x_test[0].reshape(-1,20,20),training=True) 

plt.imshow(one_noisey_images[0])  

#look noisy picture
denoised = noise_remover(x_test[:10]) 

plt.imshow(denoised[0])      

#look the output image after processed through denoiser