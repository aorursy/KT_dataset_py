import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import csv
import cv2
import gc
import operator
import random
import warnings
from os.path import split

from sklearn.utils import class_weight
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from random import shuffle
from IPython.display import Image
from pathlib import Path
import matplotlib.image as mpimg
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.utils import np_utils
import keras.backend as K
from keras.models import Sequential
from keras import optimizers
from collections import Counter

trainDir = "../input/whale-categorization-playground/train/train"
testDir="../input/whale-categorization-playground/test/test/"
valuesFile= "../input/whale-categorization-playground/train.csv"

trainData = pd.read_csv(valuesFile)
def rgb2grey(rgb): 
    if len(rgb.shape)==3:
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]) 
    else:
        return rgb


def transform_image(img):
    resized = cv2.resize(img, (64, 64), cv2.INTER_LINEAR)
    
    normalized = cv2.normalize(resized, None, 0.0, 1.0, cv2.NORM_MINMAX)
                         
    
    normalized= np.expand_dims(normalized, axis=2)

    
    return normalized
def imageprep(dataset):
    
    if dataset=="train":
        namelist=os.listdir(trainDir) 
        filedir=trainDir
    elif dataset=="test":
        namelist=os.listdir(testDir)
        filedir=testDir
    
    X_train = np.zeros((len(namelist), 64, 64, 1))
    
    
    for i in range(len(namelist)):
      
        img = mpimg.imread(filedir+"/"+namelist[i])
        
        gs_img= rgb2grey(img)
        
        trans_img= transform_image(gs_img)
        
        X_train[i] = trans_img
    
            
    return X_train
def labelprep(Y):
    
    labels_encoder = LabelEncoder()
    
    onehot_encoder = OneHotEncoder(sparse=False)
    
    int_encoded = labels_encoder.fit_transform(Y)
    
    int_encoded = int_encoded.reshape(len(int_encoded), 1)
      
    onehot_encoded = onehot_encoder.fit_transform(int_encoded)
   
    return onehot_encoded, labels_encoder
X = imageprep("train")


print("Shape of train data: ", X.shape)
Y = trainData['Id']

y, label_encoder = labelprep(array(Y))
outputdim=len(np.unique(Y))
model = Sequential()

model.add(Conv2D(32, (5, 5), strides = (1, 1), input_shape = (64, 64, 1)))
model.add(BatchNormalization(axis = 3))
model.add(Activation('relu'))
model.add(MaxPooling2D((3, 3)))

model.add(Conv2D(64, (3, 3), strides = (1,1)))
model.add(Activation('relu'))
model.add(BatchNormalization(axis = 3))
model.add(AveragePooling2D((3, 3)))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.65))
model.add(Dense(outputdim, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(Y),
                                                 Y)
class_weights = {i : class_weights[i] for i in range(len(class_weights))}
ini_weights=model.get_weights()
history = model.fit(X, y, epochs=400, batch_size=100, verbose=1)
gc.collect()
weights0=model.get_weights()
plt.plot(history.history['accuracy'])
plt.title('Accuracy of CNN, training set, no AUG, no class weights')
plt.ylabel('Accuracy')
plt.xlabel('Training epoch')
plt.show()
model.set_weights(ini_weights)
history_weights = model.fit(X, y, epochs=400, batch_size=100, verbose=1, class_weight= class_weights)
gc.collect()
weights1=model.get_weights()
plt.plot(history_weights.history['accuracy'])
plt.title('Accuracy of CNN, training set, no AUG, with class weights')
plt.ylabel('Accuracy')
plt.xlabel('Training epoch')
plt.show()
datagen = image.ImageDataGenerator( 
    #rescale=1./255,
    #rotation_range=15,
    #width_shift_range=.15,
    #height_shift_range=.15,
    horizontal_flip=True)

datagen.fit(X)



gc.collect()
model.set_weights(ini_weights)
history_aug=model.fit_generator(datagen.flow(X, y, batch_size=100), epochs=400, verbose=1)
gc.collect()
weights2=model.get_weights()
plt.plot(history_aug.history['accuracy'])
plt.title('CNN, augmented flips, no class weights')
plt.ylabel('Accuracy on test set')
plt.xlabel('Learning epoch')
plt.show()
model.set_weights(ini_weights)
history1=model.fit_generator(datagen.flow(X, y, batch_size=100), epochs=400, verbose=1, class_weight= class_weights)
gc.collect()
weights3=model.get_weights()

plt.plot(history1.history['accuracy'])
plt.title('CNN, augmented flips,  class weights')
plt.ylabel('Accuracy on test set')
plt.xlabel('Learning epoch')
plt.show()
plt.plot(history1.history['accuracy'],'r-', label="AUG, CW")
plt.plot(history_aug.history['accuracy'], 'b-', label="AUG, NO CW")
plt.plot(history_weights.history['accuracy'], label= "NO AUG, CW")
plt.plot(history.history['accuracy'], label="NO AUG, NO CW")
plt.rcParams["figure.figsize"] = (20,20)
plt.legend()
plt.title('CNN traning set accuracy')
plt.ylabel('Accuracy on test set')
plt.xlabel('Learning epoch')
plt.show()
X_test = imageprep("test")


print("Shape of train data: ", X_test.shape)

filelist=os.listdir(testDir) 
model.set_weights(weights2)
with open("sample_submission.csv","w") as f:
    with warnings.catch_warnings():
        f.write("Image,Id\n")
        warnings.filterwarnings("ignore",category=DeprecationWarning)
      
        for i in range(len(filelist)):
            filename=filelist[i]
            y = model.predict_proba(X_test[i].reshape(1,64,64,1))

            predicted_args = np.argsort(y)[0][::-1][:5]

            inverted = label_encoder.inverse_transform(predicted_args)

            image = split(filename)[-1]

            predicted_args = " ".join( inverted)

         

            f.write("%s,%s\n" %(image, predicted_args))