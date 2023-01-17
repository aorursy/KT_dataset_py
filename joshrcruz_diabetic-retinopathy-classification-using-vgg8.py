#Lots of code was used from Huseyinefe's kernel, to help me with reading/processing data

#Import the necessary packages

import tensorflow as tf

import keras

import numpy as np

import matplotlib

import sklearn

import cv2

import os

import glob

from matplotlib import pyplot

from sklearn.model_selection import train_test_split, KFold

from keras import models

from keras.preprocessing.image import load_img,img_to_array,ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Flatten, Dropout

from keras.optimizers import Adam, SGD

from PIL import Image

from keras.utils import to_categorical



#-----Define important functions that will be used to process data-----



def read_images(path, number_of_images):

    arr = np.zeros((number_of_images, 224, 224, 3))

    i = 0

    for image in os.listdir(path): #image will be the name of the file

        image_path = path + "/" + image #creating a full path for the image 

        image = Image.open(image_path, mode='r')

        image_data = np.asarray(image, dtype='uint8')

        arr[i] = image_data

        i += 1

    return arr



#Read the images in a path for each of the different categories

#0: No DR, 1: Mild, 2: Moderate, 3: Proliferate, 4: Severe

def read_images_in_path(category):

    if category == 0: #No_DR

        path = r"/kaggle/input/diabetic-retinopathy-224x224-gaussian-filtered/gaussian_filtered_images/gaussian_filtered_images/No_DR"

    elif category == 1: #Mild

        path = r"/kaggle/input/diabetic-retinopathy-224x224-gaussian-filtered/gaussian_filtered_images/gaussian_filtered_images/Mild"

    elif category == 2: #Moderate

        path = r"/kaggle/input/diabetic-retinopathy-224x224-gaussian-filtered/gaussian_filtered_images/gaussian_filtered_images/Moderate"

    elif category == 3: #Proliferate_DR

        path = r"/kaggle/input/diabetic-retinopathy-224x224-gaussian-filtered/gaussian_filtered_images/gaussian_filtered_images/Proliferate_DR"

    elif category == 4: #Severe

        path = r"/kaggle/input/diabetic-retinopathy-224x224-gaussian-filtered/gaussian_filtered_images/gaussian_filtered_images/Severe"

    else:

        raise ValueError('Invalid category')

    end_path = path + '/*'

    num_in_path = len(glob.glob(end_path))

    images = read_images(path, num_in_path)

    images = images.astype('uint8')

    return num_in_path, images



#Normalizes pixels for faster training

def normalize_pixels(images):

    images = images.astype('float32')

    images = images/255

    return images



#Decreases square image to res x res

def decrease_res(images, num_images, res):

    new_images = np.zeros((num_images, res, res, 3))

    i = 0

    for image in images:

        new_image     = cv2.resize(image, (res,res))

        new_images[i] = new_image

        i += 1

    return new_images
No_DR_num, No_DR_images   = read_images_in_path(0) #1805

Mild_num, Mild_images     = read_images_in_path(1) #370

Mod_num, Mod_images       = read_images_in_path(2) #999

Prolif_num, Prolif_images = read_images_in_path(3) #295

Severe_num, Severe_images = read_images_in_path(4) #193
pyplot.imshow(Mild_images[29])

pyplot.axis("off")

pyplot.show()
No_DR_images  = normalize_pixels(No_DR_images)

Mild_images   = normalize_pixels(Mild_images)

Mod_images    = normalize_pixels(Mod_images)

Prolif_images = normalize_pixels(Prolif_images)

Severe_images = normalize_pixels(Severe_images)
no_DR  = np.zeros(No_DR_num)

mild   = np.ones(Mild_num)

mod    = np.full(Mod_num, 2)

prolif = np.full(Prolif_num, 3)

severe = np.full(Severe_num, 4)

labels = np.concatenate((no_DR, mild, mod, prolif, severe), axis=0)

labels = to_categorical(labels)
x = np.concatenate((No_DR_images, Mild_images, Mod_images, Prolif_images, Severe_images))

y = labels
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.2, random_state=42)
x_train  = decrease_res(x_train, np.shape(x_train)[0], res=64)

x_val    = decrease_res(x_val, np.shape(x_val)[0], res=64)

x_test   = decrease_res(x_test, np.shape(x_test)[0], res=64)
print(np.shape(x_train))

print(np.shape(x_val))

print(np.shape(x_test))
def define_VGG8():

    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', input_shape=(64,64,3)))

    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform'))

    model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform'))

    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform'))

    model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform'))

    model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform'))

    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())

    model.add(Dense(512, activation = 'relu', kernel_initializer = 'he_uniform'))

    model.add(Dropout(0.5))

    model.add(Dense(5, activation = 'softmax'))

    opt = SGD(lr=0.01, momentum=0.9)

    model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    return model
def evaluate_model(trainX, trainY, valX, valY, model, batch_size, epochs):

    hist = model.fit(trainX, trainY, batch_size, epochs, verbose=1, validation_data=(valX, valY))

    _, train_score = model.evaluate(trainX,trainY)

    _, val_score  = model.evaluate(valX,valY)

    return hist, train_score, val_score



#Summarizing results of a particular model

def results_summary(hist):

    pyplot.subplot(2,1,1)

    pyplot.title('Loss')

    pyplot.plot(hist.history['loss'], color='blue',label='Train')

    pyplot.plot(hist.history['val_loss'], color='orange', label='Validation')

    pyplot.subplot(2,1,2)

    pyplot.title('Accuracy')

    pyplot.plot(hist.history['accuracy'], color='blue', label='Train')

    pyplot.plot(hist.history['val_accuracy'], color='orange', label = 'Validation')

    pyplot.show()
model   = define_VGG8()

hist, train_score, val_score = evaluate_model(x_train, y_train, x_val, y_val, model, batch_size=32, epochs=75)
results_summary(hist)
model2 = define_VGG8()

model3 = define_VGG8()

model4 = define_VGG8()



hist2, train_score2, val_score2 = evaluate_model(x_train, y_train, x_val, y_val, model2, batch_size=32, epochs=30)

hist3, train_score3, val_score3 = evaluate_model(x_train, y_train, x_val, y_val, model3, batch_size=16, epochs=30)

hist4, train_score4, val_score4 = evaluate_model(x_train, y_train, x_val, y_val, model4, batch_size= 8, epochs=50)
results_summary(hist2)

results_summary(hist3)

results_summary(hist4)
def fit_model(x_train, y_train, model, batch_size, epochs):

    history = model.fit(x_train, y_train, batch_size, epochs, verbose=1)

    _, train_score = model.evaluate(x_train, y_train)

    model.save('Diabetic_Retinopathy_Model.h5')

    return history, train_score
final_model = define_VGG8()

hist, train_score = fit_model(x_train, y_train, final_model, batch_size=8, epochs=50)
final_model = models.load_model('Diabetic_Retinopathy_Model.h5')

_, score = final_model.evaluate(x_test, y_test)

print(score)