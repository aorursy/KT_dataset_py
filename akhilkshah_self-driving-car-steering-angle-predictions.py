import numpy as np

import pandas as pd

import cv2

import os

import matplotlib.pyplot as plt



from keras import layers



from keras import models



from keras.layers import (Input, Dense, Activation, ZeroPadding2D,

BatchNormalization, Flatten, Conv2D, concatenate, Lambda)



from keras.layers import (AveragePooling2D, MaxPooling2D, Dropout,

GlobalMaxPooling2D, GlobalAveragePooling2D)



from keras.models import Model, load_model

from keras import regularizers, optimizers



from sklearn.model_selection import train_test_split



from keras.utils import to_categorical



print(os.listdir("../input"))

print(os.listdir('../input/self driving car training data/data'))
path = '../input/self driving car training data/data'

path = os.path.join(path,'driving_log.csv')



data_frame = pd.read_csv(path)

center = data_frame[data_frame.columns[0]].values

left = data_frame[data_frame.columns[1]].values

right = data_frame[data_frame.columns[2]].values

steering = data_frame[data_frame.columns[3]].values



no_of_examples = len(steering)

print(no_of_examples)
def random_flip(image, steering_angle):

    

    image = cv2.flip(image, 1)

    steering_angle = -steering_angle

    

    return image, steering_angle
train_x = []

train_y = []



img_folder = '../input/self driving car training data/data/IMG'

stear_adjust_factor = 0.2

IMAGE_HEIGHT = 100 

IMAGE_WIDTH = 100



for i in range(no_of_examples):

    

    for choice in range(3):

        

        if choice == 0: #Center

            img = cv2.imread(os.path.join(img_folder,center[i].split('IMG/')[1]))

            steering_angle = steering[i]



        elif choice == 1: #Left

            img = cv2.imread(os.path.join(img_folder,left[i].split('IMG/')[1]))

            steering_angle = steering[i] + stear_adjust_factor



        elif choice == 2: #Right

            img = cv2.imread(os.path.join(img_folder,right[i].split('IMG/')[1]))

            steering_angle = steering[i] - stear_adjust_factor

        

        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,1]

        img = cv2.resize(img,(IMAGE_HEIGHT,IMAGE_WIDTH))

        

        train_x.append(img)

        train_y.append(steering_angle)

        

        flip_img,steering_angle = random_flip(img,steering_angle)

                

        train_x.append(flip_img)

        train_y.append(steering_angle)

        



train_x = np.array(train_x)

train_x = np.reshape(train_x,[train_x.shape[0],train_x.shape[1],train_x.shape[2],1])



train_y = np.array(train_y)

train_y = np.reshape(train_y,[train_y.shape[0],1])



print(train_x.shape)

print(train_y.shape)
x_train,x_test,y_train,y_test = train_test_split(train_x,train_y,random_state=42,test_size=.20)
def model(height,width):

        

    x_input = Input(shape=(height,width,1))

    

    x = Lambda(lambda x: x/127.5-1.0)(x_input)

    

    x = Conv2D(32,(3,3),activation='relu',padding='same')(x_input)

    

    x = Conv2D(32,(3,3),activation='relu',padding='same')(x)

    x = MaxPooling2D((2,2),padding='valid')(x)

    

    x = Conv2D(64,(3,3),activation='relu',padding='same')(x)

    

    x = Conv2D(64,(3,3),activation='relu',padding='same')(x)

    x = MaxPooling2D((2,2),padding='valid')(x)

    

    x = Conv2D(128,(3,3),activation='relu',padding='same')(x)

    x = MaxPooling2D((2,2),padding='valid')(x)

    

    x = Conv2D(128,(3,3),activation='relu',padding='same')(x)

    x = MaxPooling2D((2,2),padding='valid')(x)

    

    x = Flatten()(x)

    x = Dropout(0.5)(x)

       

    x = BatchNormalization()(x)

    x = Dense(512)(x)

    x = Dense(256)(x)

    x = Dense(64)(x)

    x = Dense(1)(x)

    

    model = Model(inputs=x_input,outputs=x,name='model')

    

    return model

    

model = model(IMAGE_HEIGHT,IMAGE_WIDTH)

print(model.summary())



opt = optimizers.Adam(lr=0.0001)

model.compile(loss='mse',

             optimizer=opt,

             metrics=['mse'])



hist = model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=32,epochs=10)
model.save('Saved_Model.h5')