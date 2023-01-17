import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import sys

import os

from keras.applications.vgg16 import VGG16

import keras

from numpy import load

from matplotlib import pyplot

from sklearn.model_selection import train_test_split

from keras import backend

from keras.layers import Dense

from keras.layers import Flatten

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D

from keras.optimizers import SGD

from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.layers import Dropout

from keras.layers.normalization import BatchNormalization
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

traindir = "../input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train"

validdir = "../input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/valid"

testdir = "../input/new-plant-diseases-dataset/test/test"



train_datagen = ImageDataGenerator(rescale=1./255,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   fill_mode='nearest')



valid_datagen = ImageDataGenerator(rescale=1./255)



batch_size = 128

training_set = train_datagen.flow_from_directory(traindir,

                                                 target_size=(224, 224),

                                                 batch_size=batch_size,

                                                 class_mode='categorical')



valid_set = valid_datagen.flow_from_directory(validdir,

                                            target_size=(224, 224),

                                            batch_size=batch_size,

                                            class_mode='categorical')

class_dict = training_set.class_indices

print(class_dict)
li = list(class_dict.keys())

print(li)
train_num = training_set.samples

valid_num = valid_set.samples

print("train_num is:",train_num)

print("valid_num is:",valid_num)
# base_model=VGG16(include_top=False,input_shape=(224,224,3))

# base_model.trainable=False
# classifier=keras.models.Sequential()

# classifier.add(base_model)

# classifier.add(Flatten())

# classifier.add(Dense(38,activation='softmax'))

# classifier.summary()
# classifier.compile(optimizer='adam',

#               loss='categorical_crossentropy',

#               metrics=['accuracy'])
# #fitting images to CNN

# history = classifier.fit(training_set,

#                          steps_per_epoch=train_num//batch_size,

#                          validation_data=valid_set,

#                          epochs=5,

#                          validation_steps=valid_num//batch_size,

#                          )
# #Saving our model

# filepath="Mymodel.h5"

# classifier.save(filepath)
# import matplotlib.pyplot as plt

# import seaborn as sns

# sns.set()



# acc = history.history['accuracy']

# val_acc = history.history['val_accuracy']

# loss = history.history['loss']

# val_loss = history.history['val_loss']

# epochs = range(1, len(loss) + 1)



# #accuracy plot

# plt.plot(epochs, acc, color='green', label='Training Accuracy')

# plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')

# plt.title('Training and Validation Accuracy')

# plt.ylabel('Accuracy')

# plt.xlabel('Epoch')

# plt.legend()



# plt.figure()

# #loss plot

# plt.plot(epochs, loss, color='pink', label='Training Loss')

# plt.plot(epochs, val_loss, color='red', label='Validation Loss')

# plt.title('Training and Validation Loss')

# plt.xlabel('Epoch')

# plt.ylabel('Loss')

# plt.legend()



# plt.show()
from tensorflow import keras

classifier = keras.models.load_model("../input/modelvgg16/Mymodel.h5")
# predicting an image

import matplotlib.pyplot as plt

from keras.preprocessing import image

import numpy as np

image_path = "../input/new-plant-diseases-dataset/test/test/TomatoEarlyBlight1.JPG"

new_img = image.load_img(image_path, target_size=(224, 224))

img = image.img_to_array(new_img)

img = np.expand_dims(img, axis=0)

img = img/255



print("Following is our prediction:")

prediction = classifier.predict(img)

# decode the results into a list of tuples (class, description, probability)

# (one such list for each sample in the batch)

d = prediction.flatten()

j = d.max()

for index,item in enumerate(d):

    if item == j:

        class_name = li[index]



     

        

##Another way

img_class = classifier.predict_classes(img)

img_prob = classifier.predict_proba(img)

#ploting image with predicted class name        

plt.figure(figsize = (4,4))

plt.imshow(new_img)

plt.axis('off')

plt.title(class_name)

plt.show()
#Confution Matrix and Classification Report

from sklearn.metrics import classification_report, confusion_matrix

Y_pred = classifier.predict_generator(valid_set, valid_num//batch_size+1)
class_dict = valid_set.class_indices

li = list(class_dict.keys())

print(li)
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')

print(confusion_matrix(valid_set.classes, y_pred))

print('Classification Report')

target_names =li ## ['Peach___Bacterial_spot', 'Grape___Black_rot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

print(classification_report(valid_set.classes, y_pred, target_names=target_names))