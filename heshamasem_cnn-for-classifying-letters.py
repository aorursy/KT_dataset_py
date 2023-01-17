import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import tensorflow as tf

import keras

import os

import glob as gb

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten ,Conv2D, MaxPooling2D
all_letters = os.listdir('../input/notmnist/notMNIST_large/notMNIST_large')



print(f'We have {len(all_letters)} letters , which are : {all_letters}')
total_images = 0

for letter in all_letters : 

    available_images = gb.glob(pathname= f'../input/notmnist/notMNIST_large/notMNIST_large/{letter}/*.png')

    total_images+=len(available_images)

    print(f'for letter {letter} we have  {len(available_images)} available images')

print('-----------------------')    

print(f'Total Images are {total_images} images')
X = list(np.zeros(shape=(total_images , 28,28)))

y = list(np.zeros(shape=(total_images)))
i=0

y_value = 0

for letter in all_letters : 

    available_images = gb.glob(pathname= f'../input/notmnist/notMNIST_large/notMNIST_large/{letter}/*.png')

    for image in available_images : 

        try : 

            x = plt.imread(image)

            X[i] = x

            y[i] = y_value

            i+=1

        except : 

            pass

    y_value+=1
ohe  = OneHotEncoder()

y = np.array(y)

y = y.reshape(len(y), 1)

ohe.fit(y)

y = ohe.transform(y).toarray()
y[10000]
X = np.expand_dims(X, -1).astype('float32')/255.0
X.shape
X_part, X_cv, y_part, y_cv = train_test_split(X, y, test_size=0.15, random_state=44, shuffle =True)



print('X_train shape is ' , X_part.shape)

print('X_test shape is ' , X_cv.shape)

print('y_train shape is ' , y_part.shape)

print('y_test shape is ' , y_cv.shape)
X_train, X_test, y_train, y_test = train_test_split(X_part, y_part, test_size=0.25, random_state=44, shuffle =True)



print('X_train shape is ' , X_train.shape)

print('X_test shape is ' , X_test.shape)

print('y_train shape is ' , y_train.shape)

print('y_test shape is ' , y_test.shape)
KerasModel = keras.models.Sequential([

        keras.layers.Conv2D(filters = 32, kernel_size = 4,  activation = tf.nn.relu , padding = 'same'),

        keras.layers.MaxPool2D(pool_size=(3,3), strides=None, padding='valid'),

        keras.layers.BatchNormalization(),

        keras.layers.Conv2D(filters=32, kernel_size=4,activation = tf.nn.relu , padding='same'),

        keras.layers.MaxPool2D(),

        keras.layers.BatchNormalization(),

        keras.layers.Conv2D(filters=64, kernel_size=5,activation = tf.nn.relu , padding='same'),

        keras.layers.MaxPool2D(),

        keras.layers.Flatten(),    

        keras.layers.Dropout(0.5),        

        keras.layers.Dense(64),    

        keras.layers.Dropout(0.3),            

        keras.layers.Dense(units= 10,activation = tf.nn.softmax ),                



    ])

    



KerasModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
#Train

KerasModel.fit(X_train,y_train,validation_data=(X_cv, y_cv),epochs=3,batch_size=64,verbose=1)
KerasModel.summary()
y_pred = KerasModel.predict(X_test)



print('Prediction Shape is {}'.format(y_pred.shape))
Letters ={0:'A', 1:'B' , 2:'C' ,3:'D' ,4:'E' ,5:'F' ,6:'G' ,7:'H' ,8:'I' ,9:'J' }



for i in list(np.random.randint(0,len(X_test) ,size= 10)) : 

    print(f'for sample  {i}  the predicted value is   {Letters[np.argmax(y_pred[i])]}   , while the actual letter is {Letters[np.argmax(y_test[i])]}')

ModelLoss, ModelAccuracy = KerasModel.evaluate(X_test, y_test)



print('Test Loss is {}'.format(ModelLoss))

print('Test Accuracy is {}'.format(ModelAccuracy ))