import numpy as np

import pandas as pd

import pydicom

import matplotlib.pyplot as plt

from PIL import Image

import tensorflow as tf

from random import shuffle

from sklearn.model_selection import train_test_split



import keras

from keras.models import Sequential

from keras.utils import plot_model

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Convolution2D

from keras.optimizers import SGD

from keras.layers. normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

import numpy as np
def data_loader_csv(path,columns=None):

    data = pd.read_csv(path)

    if not columns==None:

        data = data.filter(columns)

    return data



def image_reader(image_path, show='False'):

    image_arr = pydicom.read_file(image_path)

    image_arr = image_arr.pixel_array

    if show:

        plt.imshow(image_arr,cmap='gray')

        plt.show()

    return image_arr



def label_one_shot(label_value):

    if label_value == 0:

        return np.array([0, 1])

    elif label_value == 1:

        return np.array([1, 0])

    



def load_data(dataset, resize_size):

    data = []

    for i in range(len(dataset)):

        array_img = image_reader('../input/rsna-pneumonia-detection-challenge/stage_2_train_images/'+dataset[i][0]+'.dcm', show=False)

        img = Image.fromarray(array_img)

        img = img.resize(resize_size)

        array_img = np.array(img) / 255

        data.append([array_img, label_one_shot(dataset[i][1])])

        

    shuffle(data)

    

    return data

 

data_rsna = data_loader_csv('../input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv', ['patientId','Target'])

#data_rsna.head(10)

dataset = data_rsna.values



arr = image_reader('../input/rsna-pneumonia-detection-challenge/stage_2_train_images/'+dataset[0][0]+'.dcm', show=True) #/ test



image_height = int(arr.shape[0] / 8)  # Image size checking -> 128 x 128

image_width = int(arr.shape[1] / 8)



data_r = load_data(dataset, (image_width, image_height)) # Image size 128x128

print('Complete load_data!')
Images = np.array([i[0] for i in data_r]).reshape(-1, image_height, image_width, 1).astype('float32') # trainImages = Image size 128x128

Labels = np.array([i[1] for i in data_r]).astype('float32') # trainLabels

X_train, X_test, y_train, y_test = train_test_split(Images, Labels , test_size=0.1)
keras.optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
model = Sequential()

model.add(Conv2D(8, kernel_size=(4, 4), strides=(1, 1), padding='same',activation='relu', input_shape= X_train.shape[1:]))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(BatchNormalization())

model.add(Conv2D(16, (2, 2), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(BatchNormalization())

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1000, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.summary()
from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

%matplotlib inline

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
history = model.fit(X_train, y_train, validation_split = 0.1, batch_size = 50, epochs = 20, verbose = 1)
# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test Accuracy:', test_acc)

print('Test Loss:', test_loss)