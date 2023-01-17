import os

import cv2

import tensorflow as tf

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from tensorflow.keras.datasets import cifar10

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D



from tensorflow import keras

from tensorflow.keras import Model, Input, optimizers



from PIL import Image



from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split



import os

print(os.listdir("../input/cell_images/cell_images"))
infected = os.listdir('../input/cell_images/cell_images/Parasitized/') 

uninfected = os.listdir('../input/cell_images/cell_images/Uninfected/')
#retrieving the images and storing them in the arrays

data = []

labels = []



for i in infected:

    try:

    

        image = cv2.imread("../input/cell_images/cell_images/Parasitized/"+i)

        image_array = Image.fromarray(image , 'RGB')

        resize_img = image_array.resize((64 , 64))

        

        data.append(np.array(resize_img))

        

        labels.append(1)

        

        

    except AttributeError:

        print('')

    

for u in uninfected:

    try:

        

        image = cv2.imread("../input/cell_images/cell_images/Uninfected/"+u)

        image_array = Image.fromarray(image , 'RGB')

        resize_img = image_array.resize((64 , 64))

        

        data.append(np.array(resize_img))

        

        labels.append(0)

        

    except AttributeError:

        print('')
cells = np.array(data)

labels = np.array(labels)

cells.shape,labels.shape
#Shuffle the data

cells,labels = shuffle(cells,labels)

cells = cells.astype("float32")/255

labels = tf.keras.utils.to_categorical(labels)

x_train,x_test,y_train,y_test = train_test_split(cells,labels,test_size=0.33,random_state=45)

x_train.shape,x_test.shape,y_train.shape,y_test.shape




model = Sequential()



model.add(Conv2D(64, (3, 3), input_shape=x_train.shape[1:]))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors



model.add(Dense(32))



model.add(Dense(2))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



model.summary()
history = model.fit(x_train, y_train, batch_size=32, epochs=8, validation_split=0.15)
plt.plot(history.history['loss'],label='Loss')

plt.plot(history.history['val_loss'],label="Val Loss")

plt.legend()
accuracy  = model.evaluate(x_test,y_test)

print("Test Accuracy:-",accuracy)