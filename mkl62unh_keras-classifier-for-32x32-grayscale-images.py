#import modules

import pandas as pd

import numpy as np

from collections import Counter

import os





#visuals

import matplotlib.pyplot as plt



#determine class weights

from sklearn.utils import class_weight



#Image Preprocessing

from keras.preprocessing.image import ImageDataGenerator



#Keras

import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

from keras.optimizers import Adam

from keras.optimizers import SGD

from keras import initializers

from keras.callbacks import TensorBoard

from keras.utils import np_utils

from keras.constraints import maxnorm

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers.normalization import BatchNormalization



#Disable Warnings

import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv('..//input//train.csv')

print(train.shape)

print(train.head(10))
train.has_cactus.value_counts()


#set class weights

class_weights = class_weight.compute_class_weight('balanced',

                                                 np.unique(train['has_cactus']),

                                                 train['has_cactus'])

print(class_weights)
def create_training_model_data(dataframe, batch_size=32, mode='categorical'):

    

        #TypeError: If class_mode="binary", y_col="has_cactus" column values must be strings.

        dataframe['has_cactus'] = dataframe['has_cactus'].astype(str) #resolve error

        

        IDG = ImageDataGenerator(rescale=1./255.,      #rescale: make RGB values between 1 and 0

                                 horizontal_flip=True, #horizontal_flip: Boolean. Randomly flip inputs horizontally.

                                 vertical_flip=True)   #vertical_flip: Boolean. Randomly flip inputs vertically.

        

        #Create Train Data to Feed Into CNN

        train_data = IDG.flow_from_dataframe(dataframe=dataframe[:15925], #select first 90% of data

                                             directory='..//input//train//train', #path to the images

                                             x_col='id', #column in df that contains image names

                                             y_col='has_cactus', #column in df that contains labels

                                             class_mode='binary', #binary output

                                             batch_size=batch_size, #batch size

                                             color_mode='grayscale', #convert images to gray scale

                                             target_size=(32,32)) #input image size

        

        #Create Validation Data to Feed Into CNN

        validation_data = IDG.flow_from_dataframe(dataframe=dataframe[15925:], #select last 10% of data

                                                  directory='..//input//train//train//',

                                                  x_col='id',

                                                  y_col='has_cactus',

                                                  class_mode='binary',

                                                  batch_size=batch_size,

                                                  color_mode='grayscale',

                                                  target_size=(32,32))

        

        return train_data, validation_data
train_data, validation_data = create_training_model_data(train)
model_alexnet = Sequential()



# 1st Convolutional Layer

model_alexnet.add(Conv2D(32,(3,3),                

                 input_shape=(32, 32, 1), #dimensions = 32X32, color channel = B&W

                 padding='same',

                 activation='relu'))



#pooling

model_alexnet.add(MaxPooling2D(pool_size=(2,2), padding='same')) 

model_alexnet.add(BatchNormalization())



# 2nd Convolutional Layer

model_alexnet.add(Conv2D(64,(3,3),

                padding='same',

                activation='relu'))



#pooling

model_alexnet.add(MaxPooling2D(pool_size=(2,2), padding='same'))

model_alexnet.add(BatchNormalization())



# 3rd Convolutional Layer

model_alexnet.add(Conv2D(64,(3,3),

                padding='same',

                activation='relu'))

model_alexnet.add(BatchNormalization())



#4th Convolutional Layer

model_alexnet.add(Conv2D(128,(3,3),

                padding='same',

                activation='relu'))

model_alexnet.add(BatchNormalization())



#5th Convolutional Layer

model_alexnet.add(Conv2D(128,(3,3),

                padding='same',

                activation='relu'))



#pooling

model_alexnet.add(MaxPooling2D(pool_size=(3,3), padding='same'))

model_alexnet.add(BatchNormalization())





#Flatten

model_alexnet.add(Flatten())



#1st Dense Layer

model_alexnet.add(Dense(128,

               activation='relu', kernel_initializer='glorot_uniform'))

model_alexnet.add(Dropout(0.10))

model_alexnet.add(BatchNormalization())



#2nd Dense Layer

model_alexnet.add(Dense(256,

               activation='relu', kernel_initializer='glorot_uniform'))

model_alexnet.add(Dropout(0.20))

model_alexnet.add(BatchNormalization())



# # 3rd Dense Layer

model_alexnet.add(Dense(512,

               activation='relu', kernel_initializer='glorot_uniform'))

model_alexnet.add(Dropout(0.2))

model_alexnet.add(BatchNormalization())



#output layer

model_alexnet.add(Dense(1, activation='sigmoid'))



#Compile 

model_alexnet.compile(loss='binary_crossentropy', optimizer='adam',

 metrics=['accuracy'])



# Set callback functions to early stop training and save the best model so far

callbacks = [EarlyStopping(monitor='val_loss', patience=20), #stop if no improvment after 10 epochs

             

             ModelCheckpoint(filepath='best_model_alexnet.h5', monitor='val_loss', save_best_only=True)] #improvment val_loss



#summary of Model

model_alexnet.summary()
#fit model

history = model_alexnet.fit_generator(train_data,

          epochs=100,

          steps_per_epoch=(15925/32),

          callbacks=callbacks,

          validation_data = validation_data,

          validation_steps=(1575/32),

          class_weight=class_weights,

          verbose=2)
accuracy = history.history['acc']

val_accuracy = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')

plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')

plt.title('Training and Validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
#Create IDG for test images

test_IDG = ImageDataGenerator(rescale=1./255.) #rescale test image RGB values



#Perform IDG on images within the test directory

test_data = test_IDG.flow_from_directory(

    directory='..//input//test//',

    target_size=(32,32),

    color_mode='grayscale',

    class_mode='binary',

    batch_size=1,

    shuffle=False)
#get prediction probabilities for the 4000 test images

y_pred = model_alexnet.predict_generator(test_data,steps=4000)



#turn array into a single list

y_pred = np.hstack(y_pred).tolist()
#get a count of class predictions

has_cactus = [0 if proba<0.50 else 1 for proba in y_pred]

print(Counter(has_cactus).keys()) # equals to list(set(words))

print(Counter(has_cactus).values())
#get the name of the files in the directory

files=[]

files = [f for f in sorted(os.listdir('..//input//test//test'))]
submission = pd.DataFrame({'id':files,

                          'has_cactus':y_pred})
submission.head(10)
submission.to_csv('submission.csv', index=False)