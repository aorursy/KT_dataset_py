# Import Libraries

import os

import cv2

import scipy

import numpy as np

from tqdm import tqdm

from random import shuffle

import matplotlib.pyplot as plt



from keras.optimizers import Adam

from keras.applications import InceptionResNetV2

from keras.models import Sequential, Model

from keras.layers import  Dropout,Dense,GlobalAveragePooling2D

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
# File Paths

input_path = "../input/chest-xray-pneumonia/chest_xray/"
# File Contents

for _set in ['train', 'val', 'test']:

    normal = len(os.listdir(input_path + _set + '/NORMAL'))

    infected = len(os.listdir(input_path + _set + '/PNEUMONIA'))

    print('The {} folder contains {} Normal and {} Pneumonia images.'.format(_set, normal, infected))
# Preprocesing Data Function

def preprocess_data(input_path, img_dims, batch_size):

    

    # Data Augmentation for Train & Test Images

    train_datagen = ImageDataGenerator(

        rescale = 1./255,

        zoom_range = 0.2,

        shear_range = 0.2,      

        rotation_range = 20,

        width_shift_range = 0.2,

        height_shift_range = 0.2,

        horizontal_flip = True)



    test_datagen = ImageDataGenerator(

        rescale = 1./255)

    

    train_images = train_datagen.flow_from_directory(

        directory = input_path + 'train', 

        target_size = (img_dims, img_dims), 

        batch_size = batch_size, 

        class_mode = 'binary', 

        shuffle = True)



    test_images = test_datagen.flow_from_directory(

        directory = input_path + 'test', 

        target_size = (img_dims, img_dims), 

        batch_size = batch_size, 

        class_mode = 'binary', 

        shuffle = True)



    # I'm created these lists for make prediction on test image and showing confusion matrix.

    train_labels = []

    test_labels = []



    for file_name in ['/NORMAL/', '/PNEUMONIA/']:

        for img in (os.listdir(input_path + 'test' + file_name)):

            img = plt.imread(input_path + 'test' + file_name+img)

            img = cv2.resize(img, (img_dims, img_dims))

            img = np.dstack([img, img, img])

            img = img.astype('float32') / 255

            if file_name == '/NORMAL/':

                label = 0

            elif file_name == '/PNEUMONIA/':

                label = 1

            train_labels.append(img)

            test_labels.append(label)

        

    train_labels = np.array(train_labels)

    test_labels = np.array(test_labels)

    

    return train_images, train_labels, test_images, test_labels
img_dims = 150

epochs = 25

batch_size = 32



# Set Images&Labels for Train,Test

train_images, train_labels, test_images, test_labels = preprocess_data(input_path, img_dims, batch_size)
# Set Transfer Model

transfer_model = InceptionResNetV2(weights= 'imagenet', include_top = False, input_shape = (img_dims, img_dims, 3))
# Add new Layers

new_model = Sequential()

new_model.add(GlobalAveragePooling2D(input_shape = transfer_model.output_shape[1:], data_format=None)),  

new_model.add(Dense(512, activation='relu'))

new_model.add(Dropout(0.5))

new_model.add(Dense(1024, activation='relu'))

new_model.add(Dropout(0.5))

new_model.add(Dense(1, activation='sigmoid'))
# Combine Models

model = Model(inputs=transfer_model.input, outputs=new_model(transfer_model.output))

model.summary()
# Set Optimizer

optimizer = Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08,decay = 0.0)



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

fig, ax = plt.subplots(1, 2, figsize=(10, 3))

ax = ax.ravel()



for i, met in enumerate(['acc', 'loss']):

    plt.style.use("ggplot")

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
from keras.preprocessing.image import image



# Image Classifer Script

def predict_image(model, img_path, img_dims = 150):

    img = image.load_img(img_path, target_size = (img_dims, img_dims))

    plt.imshow(img)

    plt.show()

    img = image.img_to_array(img)

    x = np.expand_dims(img, axis=0) * 1./255

    score = model.predict(x)

    print('Predictions: ', score, 'NORMAL' if score < 0.5 else 'PNEUMONIA')
# Test on Validation Images

predict_image(model,(input_path+'val/NORMAL/NORMAL2-IM-1431-0001.jpeg'))

predict_image(model,(input_path+'val/PNEUMONIA/person1947_bacteria_4876.jpeg'))
# Save Model

model.save("InceptionResnetV2-Transfer-Model-Pneumonia.h5")