# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


from __future__ import absolute_import, division, print_function

import pandas as pd

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

import cv2

import seaborn as sns

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import load_img

from keras.utils import np_utils



from PIL import Image

import os
print(os.listdir("../input/cell_images/cell_images"))
infected_cells = os.listdir('../input/cell_images/cell_images/Parasitized/')

uninfected_cells = os.listdir('../input/cell_images/cell_images/Uninfected/')
# visualising the dataset

# shows 5 images from specified

def show_images(file_location, infected):

    plt.figure(figsize=(12,12))

    

    for i in range(5):

        plt.subplot(1, 5, i+1)

        if infected:

            image = cv2.imread(file_location + '/' + infected_cells[i])

        else :

            image = cv2.imread(file_location + '/' + uninfected_cells[i])

        plt.imshow(image)

        plt.tight_layout()

    plt.show()
infected_filename = '../input/cell_images/cell_images/Parasitized'

uninfected_filename= '../input/cell_images/cell_images/Uninfected'
show_images(infected_filename, True)
show_images(uninfected_filename, False)
dataset=[]

labels=[]



for img in infected_cells:

    try:

        img_read = plt.imread(infected_filename + "/" + img)

        img_resize = cv2.resize(img_read, (50, 50))

        img_array = img_to_array(img_resize)

        img_aray=img_array/255

        dataset.append(img_array)

        labels.append(1)

    except:

        None

        

for img in uninfected_cells:

    try:

        img_read = plt.imread(uninfected_filename + "/" + img)

        img_resize = cv2.resize(img_read, (50, 50))

        img_array = img_to_array(img_resize)

        img_array= img_array/255

        dataset.append(img_array)

        labels.append(0)

    except:

        None
plt.imshow(dataset[0])

plt.show()
images = np.array(dataset)

label_arr = np.array(labels)



#arange the indices

index = np.arange(images.shape[0])



#shuffle the indices

np.random.shuffle(index)

images = images[index]

label_arr = label_arr[index]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(images, label_arr, test_size = 0.2, random_state = 42)
y_train = np_utils.to_categorical(y_train, num_classes=2)

y_test = np_utils.to_categorical(y_test, num_classes=2)
from keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense, Activation

from keras import optimizers

from keras.models import Sequential
inputs = (50, 50, 3)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=inputs))

model.add(MaxPooling2D(pool_size=2))

model.add(BatchNormalization(axis=-1))

model.add(Dropout(0.3))



model.add(Conv2D(filters=16, kernel_size=1, activation='relu'))

model.add(Dropout(0.3))



model.add(Conv2D(filters=16, kernel_size=3, activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(BatchNormalization(axis=-1))

model.add(Dropout(0.3))



model.add(Conv2D(filters=32, kernel_size=1, activation='relu'))

model.add(Dropout(0.3))

                           

model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(BatchNormalization(axis=-1))

model.add(Dropout(0.3))

                           

model.add(Flatten())

model.add(Dense(512, activation = 'relu'))

model.add(BatchNormalization(axis = -1))

model.add(Dropout(0.5))

model.add(Dense(2, activation = 'softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20)
# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train','test'], loc='upper right')

plt.show()
# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train','test'], loc='upper right')

plt.show()
pred = model.predict(x_test)
from sklearn.metrics import accuracy_score, classification_report

score = accuracy_score(y_test.argmax(axis=1), pred.argmax(axis=1))

print(score)
report = classification_report(y_test.argmax(axis=1), pred.argmax(axis=1))

print(report)