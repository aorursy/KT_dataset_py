import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
print("No warnings!")
################################################################################
##-------------------------------Lot of imports-------------------------------##
################################################################################
print("---Import modules---")
import numpy as np 
import pandas as pd
import h5py

import matplotlib.pylab as plt
from matplotlib import cm
%matplotlib inline

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from keras.preprocessing import image as keras_image
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.metrics import top_k_categorical_accuracy, categorical_accuracy

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers import Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D

from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator

import os
#print(os.listdir("../input/"))
from os.path import join
print("---Succeded---")
#Initialize all values
ep_number = 10
train_path = "../input/flowers/flowers"
img_size = 128
batch_size_num =32
print("Sorts of flowers are:")
types_of_flowers = os.listdir(train_path)
print(types_of_flowers)
num_classes = len(types_of_flowers)
train_generator = ImageDataGenerator(
    rescale=1./255, horizontal_flip=True, shear_range=0.2, zoom_range=0.2,
    width_shift_range=0.2, height_shift_range=0.2, fill_mode='nearest',
    validation_split=0.2)

train=train_generator.flow_from_directory(
    train_path,target_size=(img_size,img_size),classes=types_of_flowers,
    batch_size=batch_size_num, class_mode="categorical",subset='training')
val = train_generator.flow_from_directory(
    train_path,target_size=(img_size,img_size),classes=types_of_flowers,
    batch_size=batch_size_num,class_mode="categorical",subset='validation')
###############################################################################
#--------------------------------Plot function--------------------------------#
###############################################################################
def history_plot(fit_history, n):
    plt.figure(figsize=(18, 12))
    
    plt.subplot(211)
    plt.plot(fit_history.history['loss'][n:], color='slategray', label = 'train')
    plt.plot(fit_history.history['val_loss'][n:], color='#4876ff', label = 'valid')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title('Loss Function');  
    
    plt.subplot(212)
    plt.plot(fit_history.history['categorical_accuracy'][n:], color='slategray', label = 'train')
    plt.plot(fit_history.history['val_categorical_accuracy'][n:], color='#4876ff',label = 'valid')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")    
    plt.legend()
    plt.title('Accuracy');
# Create callbacks
checkpointer = ModelCheckpoint(filepath='weights.best.model.hdf5', 
                               verbose=2, save_best_only=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss',
                                 patience=5, verbose=2, factor=0.75)
print("Checkpoints setted")
def model():
    model = Sequential()
    model.add(Conv2D(32, (2,2), activation='relu', input_shape=(img_size,img_size,3)))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())  
    model.add(Dense(512,activation='relu'))
    model.add(Dense(5,activation='softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy',categorical_accuracy])
    return model

print("Creating model")
model = model()
print("Model created")
model.summary()
print("start fitting")

#history = model.fit(train_X, train_data_temp_y, epochs = 10, steps_per_epoch=100)
#history = model.fit(train_X,train_y,epochs=25,batch_size=100,verbose=2,validation_data=(val_X,val_y))
history = model.fit_generator(train,validation_data=val,
                              validation_steps=861/20,steps_per_epoch=173,
                              epochs=ep_number,verbose=1,shuffle=False)
# Plot the training history
history_plot(history, 0)