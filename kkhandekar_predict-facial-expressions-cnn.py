#Generic Packages

import numpy as np

import os

import pandas as pd

import random



#Machine Learning Library

from sklearn.metrics import confusion_matrix

from sklearn.utils import shuffle           



#Plotting Libraries

import seaborn as sn; sn.set(font_scale=1.4)

import matplotlib.pyplot as plt             



#openCV

import cv2                                 



#Tensor Flow & Keras

import tensorflow as tf    

import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense , Activation , Dropout ,Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.metrics import categorical_accuracy

from keras.models import model_from_json

from keras.callbacks import ModelCheckpoint

from keras.optimizers import *

from keras.layers.normalization import BatchNormalization



#Train & Test Data Split

from sklearn.model_selection import train_test_split



#Garbage Collector

import gc
# FER File Location

fer_file = '../input/facialexpressionrecognition/fer2013.csv'



# Expression Labels

exp_label = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']



#Column Names

col_names=['emotion','pixels','usage']



#Dataset

data = pd.read_csv(fer_file,names=col_names, na_filter=False)



im=data['pixels']

data.head()
#Function to read the file

def getData(file):

    Y = []

    X = []

    first = True

    for line in open(file):

        if first:

            first = False

        else:

            row = line.split(',')

            Y.append(int(row[0]))

            X.append([int(p) for p in row[1].split()])



    X, Y = np.array(X) / 255.0, np.array(Y)

    return X, Y



# Load Data

X, Y = getData(fer_file)

num_class = len(set(Y))

#print(num_class)



# reshape X dataset

N, D = X.shape

X = X.reshape(N, 48, 48, 1)
#Train & Test Data Split



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)

y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)
#Build Model



def build_model():

    model = Sequential()

    input_shape = (48,48,1)

    model.add(Conv2D(64, (5, 5), input_shape=input_shape,activation='relu', padding='same'))

    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))

    model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))

    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Flatten())

    model.add(Dense(128))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(0.2))

    model.add(Dense(7))

    model.add(Activation('softmax'))

    

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')

    return model
#Instantiate the Model

model=build_model()



#Model Architecture Summary

model.summary()
path_model='model_filter.h5' # save model at this location after each epoch

K.tensorflow_backend.clear_session() # destroys the current graph and builds a new one

model = build_model ()

K.set_value(model.optimizer.lr,1e-3) # set the learning rate
#Train The Model

# fit the model

h=model.fit(x=X_train,y=y_train,batch_size=100,epochs=10,verbose=1,validation_data=(X_test,y_test),shuffle=True,callbacks=[ModelCheckpoint(filepath=path_model),])
gc.collect()
#Checking the Accuracy

test_loss = model.evaluate(X_test, y_test)
objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

y_pos = np.arange(len(objects))

#print(y_pos)
def emotion_analysis(emotions):

    objects = ['ang', 'dis', 'fear', 'hap', 'sad', 'sur', 'neu']

    y_pos = np.arange(len(objects))

    plt.bar(y_pos, emotions, align='center', alpha=0.9)

    plt.tick_params(axis='x', which='both', pad=10,width=20,length=15)

    plt.xticks(y_pos, objects)

    plt.ylabel('percentage')

    plt.title('emotion')

    plt.figure(figsize=(15,10))

    plt.show()

    

y_pred=model.predict(X_test)

#print(y_pred)

#y_test.shape
# Randomly select an unseen image and predict the expression



from skimage import io



dirc = '../input/random-facial-expressions/'  #unseen random images folder

random_img = random.choice(os.listdir(dirc))  # randomly select an image



#img = image.load_img('../input/random-facial-expressions/5.jpg', grayscale=True, target_size=(48, 48))

#show_img=image.load_img('../input/random-facial-expressions/5.jpg', grayscale=False, target_size=(400, 400))



img = image.load_img(dirc+random_img, grayscale=True, target_size=(48, 48))

show_img=image.load_img(dirc+random_img, grayscale=False, target_size=(400, 400))







x = image.img_to_array(img)

x = np.expand_dims(x, axis = 0)



x /= 255



custom = model.predict(x)

emotion_analysis(custom[0])



x = np.array(x, 'float32')

x = x.reshape([48, 48]);



plt.gray()

plt.imshow(show_img)

plt.show()



m=0.000000000000000000001

a=custom[0]

for i in range(0,len(a)):

    if a[i]>m:

        m=a[i]

        ind=i

        

print('Predicted Expression:',objects[ind])