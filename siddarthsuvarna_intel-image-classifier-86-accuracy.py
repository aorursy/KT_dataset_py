!pip install -q kaggle
from google.colab import files

files.upload()
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle
! chmod 600 ~/.kaggle/kaggle.json
! kaggle datasets list
!kaggle datasets download 'puneet6060/intel-image-classification'
!unzip intel-image-classification.zip -d sample_data
import tensorflow as tf

import cv2 

import matplotlib.pyplot as plt

import os

import numpy as np 

from sklearn.utils import shuffle

from sklearn.preprocessing import OneHotEncoder

from tensorflow import keras

import matplotlib.gridspec as gridspec

from random import randint

from keras.utils import to_categorical

from tensorflow.keras import layers, models
def get_images(directory):

    images = []

    labels = []

    

    for dir in os.listdir(directory):

        label = -1

        if(dir == "buildings"):

            label = 0

        if(dir == "forest"):

            label = 1

        if(dir == "glacier"):

            label = 2

        if(dir == "mountain"):

            label = 3

        if(dir == "sea"):

            label = 4

        if(dir == "street"):

            label = 5

        for image_dir in os.listdir(directory +"//"+ dir):

            image = cv2.imread(directory +"//"+ dir + "//"+ image_dir)

            image = cv2.resize(image, (150,150))

            images.append(image)

            labels.append(label)

    return shuffle(images, labels)
images, labels = get_images("/content/sample_data/seg_train/seg_train")

images = np.array(images)

labels = np.array(labels)
print(images.shape)
print(labels[0])

labels = to_categorical(labels)

print(labels[0])
images = images.astype('float32')

images /= 255.0
model = models.Sequential()

model.add(layers.Conv2D(filters = 64, kernel_size=(7,7), strides=(2,2), activation='relu', input_shape=(150,150,3)))

model.add(layers.BatchNormalization())

model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))

model.add(layers.Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))

model.add(layers.BatchNormalization())

model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))

model.add(layers.Conv2D(filters=128,kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))

model.add(layers.BatchNormalization())

model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))

model.add(layers.Conv2D(filters=156, kernel_size=(2,2), strides=(1,1), activation='relu', padding='same'))

model.add(layers.BatchNormalization())

model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(72, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(6, activation='softmax'))
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

model.summary()
trained = model.fit(images, labels, epochs =64 ,validation_split = 0.1, batch_size=32)
plt.plot(trained.history['accuracy'])

plt.plot(trained.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel("epochs")

plt.legend(['train', 'test'], loc = 'upper left')

plt.show()



plt.plot(trained.history['loss'])

plt.plot(trained.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('epochs')

plt.legend(['train','test'], loc = 'upper left')

plt.show()
trained = model.fit(images, labels, epochs =1 ,validation_split = 0.1, batch_size=32)
test_images, test_labels = get_images('/content/sample_data/seg_test/seg_test')
test_images = np.array(test_images)

test_labels = np.array(test_labels)

test_labels = to_categorical(test_labels)
test_images = test_images.astype('float32')

test_images /= 255.0
model.evaluate(test_images, test_labels, verbose = 1)
pred_images, pred_labels = get_images('/content/sample_data/seg_pred')

pred_images = np.array(pred_images)

pred_images.shape
pred_image = pred_images[1:30]

pred_image = pred_image.astype('float32')

pred_image /= 255.0
pred_class = model.predict_classes(np.array([pred_images[3]]))

plt.imshow(pred_images[3])

prediction = "none"

if(pred_class == 0):

    prediction = "Building"

elif(pred_class == 1):

    prediction = "Forest"  

elif(pred_class == 2):

    prediction = "Glacier"    

elif(pred_class == 3):

    prediction = "Mountain"

elif(pred_class == 4):

    prediction = 'Sea'

elif(pred_class == 5):

    prediction = "Street"

print("Model Prediction: " + prediction)
pred_class = model.predict_classes(np.array([pred_images[7]]))

plt.imshow(pred_images[7])

prediction = "none"

if(pred_class == 0):

    prediction = "Building"

elif(pred_class == 1):

    prediction = "Forest"  

elif(pred_class == 2):

    prediction = "Glacier"    

elif(pred_class == 3):

    prediction = "Mountain"

elif(pred_class == 4):

    prediction = 'Sea'

elif(pred_class == 5):

    prediction = "Street"

print("Model Prediction: " + prediction)