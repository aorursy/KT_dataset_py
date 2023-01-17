# Import Libraries

import os

import cv2

import scipy

import warnings

import numpy as np

from tqdm import tqdm

from random import shuffle

import matplotlib.pyplot as plt



from keras.optimizers import Adam, RMSprop

from keras.models import Sequential, Model

from keras.layers import  Conv2D, MaxPooling2D, Activation, Flatten,Dense, Dropout, GlobalAveragePooling2D, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
# Filter Warnings

warnings.filterwarnings("ignore")
# File Paths

input_path = "../input/covid19xray/Covid-19-X-Ray/"
# File Contents

for _set in ['Train', 'Validation']:

    normal = len(os.listdir(input_path + _set + '/healthy'))

    infected = len(os.listdir(input_path + _set + '/infected'))

    print('The {} folder contains {} Normal and {} Infected images.'.format(_set, normal, infected))
# Preprocesing Data Function

def preprocess_data(input_path, img_dims, batch_size):

    

    # Data Augmentation for Train & Test Images

    train_datagen = ImageDataGenerator(

        rescale = 1./255,

        zoom_range = 0.2,

        shear_range = 0.2,      

        rotation_range = 40,

        width_shift_range = 0.2,

        height_shift_range = 0.2,

        horizontal_flip = True,

        fill_mode='nearest')



    test_datagen = ImageDataGenerator(

        rescale = 1./255)

    

    train_images = train_datagen.flow_from_directory(

        directory = input_path + 'Train', 

        target_size = (img_dims, img_dims), 

        batch_size = batch_size, 

        class_mode = 'binary')



    test_images = test_datagen.flow_from_directory(

        directory = input_path + 'Validation', 

        target_size = (img_dims, img_dims), 

        batch_size = batch_size, 

        class_mode = 'binary')



    # I'm created these lists for make prediction on test image and showing confusion matrix.

    train_labels = []

    test_labels = []



    for file_name in ['/healthy/', '/infected/']:

        for img in (os.listdir(input_path + 'Validation' + file_name)):

            img = cv2.imread(input_path + 'Validation' + file_name + img, cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, (img_dims, img_dims))

            img = np.dstack([img, img, img])

            img = img.astype('float32') / 255

            if file_name == '/healthy/':

                label = 0

            elif file_name == '/infected/':

                label = 1

            train_labels.append(img)

            test_labels.append(label)

        

    train_labels = np.array(train_labels)

    test_labels = np.array(test_labels)

    

    return train_images, train_labels, test_images, test_labels
img_dims = 150

epochs = 30

batch_size = 2



# Set Images&Labels for Train,Test

train_images, train_labels, test_images, test_labels = preprocess_data(input_path, img_dims, batch_size)
# Create Model with KERAS library

model = Sequential()



model.add(Conv2D(32, (3,3), activation="relu", input_shape=(img_dims,img_dims,3)))

model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation="relu"))

model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation="relu"))

model.add(MaxPooling2D(2,2))

model.add(Conv2D(256, (3,3), activation="relu"))

model.add(MaxPooling2D(2,2))

model.add(Conv2D(256, (3,3), activation="relu"))

model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(512, activation="relu"))

model.add(Dropout(0.6))

model.add(Dense(1,activation="sigmoid"))

model.summary()
# Set Optimizer

optimizer = Adam(lr = 0.0001)





# Compile Model

model.compile(

    optimizer= optimizer,

    loss='binary_crossentropy',

    metrics=['acc'])
# Fit the Model

history = model.fit_generator(

            train_images,

            steps_per_epoch = train_images.samples // batch_size, 

            epochs = epochs, 

            validation_data = test_images,

            validation_steps = test_images.samples // batch_size)
# Visualize Loss and Accuracy Rates

fig, ax = plt.subplots(1, 2, figsize=(12, 3))

ax = ax.ravel()

plt.style.use("ggplot")



for i, met in enumerate(['acc', 'loss']):

    ax[i].plot(history.history[met])

    ax[i].plot(history.history['val_' + met])

    ax[i].set_title('Model {}'.format(met))

    ax[i].set_xlabel('Epochs')

    ax[i].set_ylabel(met)

    ax[i].legend(['Train', 'Val'])
# Predictions, Confusion Matrix & Performance Metrics



# Prediction on Model

Y_pred = model.predict(train_labels)

Y_pred = [ 1 if y >= 0.5 else 0 for y in Y_pred]



# Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, Y_pred)



from mlxtend.plotting import plot_confusion_matrix

fig, ax = plot_confusion_matrix(conf_mat=cm)

plt.show()



# Performance Metrics

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 

print('Confusion Matrix :')

print(cm) 

print('Accuracy Score :',accuracy_score(test_labels, Y_pred))

print('Report : ')

print(classification_report(test_labels, Y_pred))
from PIL import Image

from keras.preprocessing import image



# Image Classifer Script

def predict_image(model, img_path, img_dims = 150):

    img = image.load_img(img_path, target_size = (img_dims, img_dims))

    plt.imshow(img)

    plt.show()

    img = image.img_to_array(img)

    x = np.expand_dims(img, axis=0) * 1./255

    score = model.predict(x)

    print('Predictions: %', (float)(score*100), 'NORMAL' if score < 0.5 else 'INFECTED')

# Test on Validation Images

predict_image(model,(input_path + 'Test/healty1.png'))

predict_image(model,(input_path + 'Test/infected2.png'))
# Save Model

model.save("Covid-19-CNN-Model.h5")

model.save_weights("Covid19-CNN-Weights.h5")