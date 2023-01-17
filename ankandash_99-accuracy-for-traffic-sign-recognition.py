import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image

import cv2

import tensorflow as tf



from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

from sklearn.model_selection import train_test_split
base_path = '/kaggle/input/gtsrb-german-traffic-sign/'

print(os.listdir(base_path))
len(os.listdir(os.path.join(base_path, 'Train')))
# Read in the image data and labels

data = []

labels = []

number_of_classes = 43



for i in range(number_of_classes):

    path = os.path.join(base_path, 'Train', str(i))

    images = os.listdir(path)

    

    for item in images:

        try:

            image = Image.open(path + '/' + item)

            image = image.resize((30,30))

            image = np.array(image)

            data.append(image)

            labels.append(i)

        except:

            print('Error loading the image file')

data = np.array(data)

labels = np.array(labels)
print(f'Shape of the data {data.shape}')

print(f'Shape of the labels {labels.shape}')
# Split into training and validation data 

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)



# normalize the input data

X_train = X_train/255.

X_test = X_test/255.



print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
# one hot encoding the labels for the training and validation labels

y_train = to_categorical(y_train, number_of_classes)

y_test = to_categorical(y_test, number_of_classes)
# CNN model defination

model = Sequential([

    Conv2D(32, (3,3), activation='relu', input_shape=X_train.shape[1:]),

    Conv2D(64, (3,3), activation='relu'),

    MaxPool2D(2,2),

    Dropout(0.3),

    Conv2D(64, (3,3), activation='relu'),

    MaxPool2D(2,2),

    Dropout(0.3),

    Conv2D(128, (3,3), activation='relu'),

    MaxPool2D(2,2),

    Dropout(0.3),

    Flatten(),

    Dense(128, activation='relu'),

    Dropout(0.3),

    Dense(number_of_classes, activation='softmax')

])



# Compiling the model

metrics = ['accuracy',

           tf.keras.metrics.Precision(name='precision'),

           tf.keras.metrics.Recall(name='recall')

          ]

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)
# Running the model

epochs = 30

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, verbose=2)
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs = axs.ravel()



for i, met in enumerate(['accuracy', 'loss', 'precision', 'recall']):

    axs[i].plot(history.history[met])

    axs[i].plot(history.history['val_' + met])

    axs[i].set_title(f'Model {met}')

    axs[i].set_xlabel('epochs')

    axs[i].set_ylabel(met)

    axs[i].legend(['training', 'validation'])

plt.tight_layout()
# read in the Test.csv file 

test_data = pd.read_csv(os.path.join(base_path, 'Test.csv'))

test_data.head()
# extract the 'ClassId' and image 'Path'

image_file_path = test_data['Path'].values

print(image_file_path[:5])



test_labels = test_data['ClassId'].values

print(test_labels[:5])
# Read in the image data and normalize it

data=[]



for img in image_file_path:

    image = Image.open(base_path + '/' + img)

    image = image.resize((30,30))

    data.append(np.array(image))

X_test = np.array(data)

X_test = X_test/255.
# making prediction

prediction = model.predict_classes(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



print(accuracy_score(test_labels, prediction))
print(classification_report(test_labels, prediction))