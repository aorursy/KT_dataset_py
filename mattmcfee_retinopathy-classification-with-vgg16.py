import numpy as np 

import pandas as pd

import cv2

import seaborn as sns

import matplotlib.pyplot as plt

import random

import os

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.image import img_to_array, load_img

from keras.optimizers import Adam

from keras.models import Sequential

from keras.layers.core import Activation

from keras.layers.core import Flatten

from keras.layers.core import Dense

from keras.applications.vgg16 import VGG16

from keras.utils.np_utils import to_categorical

from keras import backend as K
base_dir = '/kaggle/input/diabetic-retinopathy-224x224-gaussian-filtered/gaussian_filtered_images/gaussian_filtered_images/'



data = []

labels = []



# Walk through all the images and convert them to arrays to be fed into the network



for subdir, dirs, files in os.walk(base_dir):

    for file in files:

        if file.endswith('.pkl') is False:

            filepath = subdir + os.sep + file

            image = load_img(filepath, target_size=(224,224))

            # image = cv2.resize(image, (122,122))

            image = img_to_array(image)

            data.append(image)

        

            label = filepath.split(os.path.sep)[-2]

            labels.append(label)

        

        else:

            continue



data = np.stack(data)

data /= 255.0

labels = np.array(labels)

print(np.unique(labels))



# Shuffle the image data and labels in unison 

X = data

y = labels



le = LabelEncoder()



y = le.fit_transform(y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)



y_train = to_categorical(y_train)

y_test = to_categorical(y_test)

# Generate the trained model and set all layers to be trainable

trained_model = VGG16(input_shape=(224,224,3), include_top=False)



for layer in trained_model.layers:

    layer.trainable = True



# Construct the model and compile

mod1 = Flatten()

mod_final = Dense(5, activation='softmax')



model = Sequential([trained_model, mod1, mod_final])



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# Fit the model to the data and validate

history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=15)



# Plot the model results using seaborn and matplotlib

sns.set(style='darkgrid')



plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()