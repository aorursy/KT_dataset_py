# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!pip install tensorflow-gpu==2.0.0-alpha0;
import tensorflow as tf

print(tf.__version__)

tf.test.is_gpu_available(

    cuda_only=False,

    min_cuda_compute_capability=None

)
# defining the path and classes.

directory = '../input/train'

test_directory = '../input/test/'

classes = ['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
# defining a shape to be used for our models.

img_size = 240
import cv2

import matplotlib.pyplot as plt
# Train class image for display.

for i in classes:

    path = os.path.join(directory,i)

    for img in os.listdir(path):

        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_COLOR)

        RGB_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        plt.imshow(RGB_img)

        plt.show()

        break

    break
# Test class image for display.

test_array = []

for img in os.listdir(test_directory):

    img_array = cv2.imread(os.path.join(test_directory,img),cv2.IMREAD_COLOR)

    RGB_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    test_array = RGB_img

    plt.imshow(RGB_img)

    plt.show()

    break
# checkking image size using shape.

print(img_array.shape)
# trying out the resize image functionality

new_img = cv2.resize(test_array,(img_size,img_size))

print(new_img.shape)

plt.imshow(new_img)

plt.show()

# creating a training dataset.

training_data = []

i = 0

def create_training_data():

    for category in classes:

        path = os.path.join(directory,category)

        class_num = classes.index(category)

        

        for img in os.listdir(path):

            img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_COLOR)

            RGB_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

            new_img = cv2.resize(RGB_img,(img_size,img_size))

            training_data.append([new_img,class_num])
create_training_data()
print(len(training_data))
random.shuffle(training_data)

x = []

y = []
for features, label in training_data:

    x.append(features)

    y.append(label)
len(x),len(y)
x[0].shape
len(x)
x = np.array(x).reshape(-1,img_size,img_size,3)

x[0].shape
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=96)
del x,y
import gc

gc.collect()
from tensorflow.keras import utils
Y_train = utils.to_categorical(y_train,num_classes=10)

Y_test = utils.to_categorical(y_test,num_classes=10)
from tensorflow.keras import datasets, layers, models, callbacks

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation, BatchNormalization
model = models.Sequential()

## CNN 1

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(240,240,3)))

model.add(BatchNormalization())

model.add(Conv2D(32,(3,3),activation='relu',padding='same'))

model.add(BatchNormalization(axis = 3))

model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

model.add(Dropout(0.3))

## CNN 2

model.add(Conv2D(64,(3,3),activation='relu',padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(64,(3,3),activation='relu',padding='same'))

model.add(BatchNormalization(axis = 3))

model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

model.add(Dropout(0.3))

## CNN 3

model.add(Conv2D(128,(3,3),activation='relu',padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(128,(3,3),activation='relu',padding='same'))

model.add(BatchNormalization(axis = 3))

model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

model.add(Dropout(0.5))

## Dense & Output

model.add(Flatten())

model.add(Dense(units = 512,activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(units = 128,activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(10,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

callback = [callbacks.EarlyStopping(monitor='val_accuracy',patience=5)]

batch_size = 70

n_epochs = 15
results = model.fit(x_train,Y_train,

                    batch_size=batch_size,epochs=n_epochs,

                    verbose=1,

                    validation_data=(x_test,Y_test),

                    callbacks=callback)
# Plot training & validation accuracy values

plt.plot(results.history['accuracy'])

plt.plot(results.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='lower right')

plt.show()



# Plot training & validation loss values

plt.plot(results.history['loss'])

plt.plot(results.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper right')

plt.show()
pred= model.predict_classes(x_test)
import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(y_test, pred)

plt.show()
from sklearn.metrics import classification_report 

print(classification_report(y_test, pred))
random_filename = random.choice([

    x for x in os.listdir(test_directory)

    if os.path.isfile(os.path.join(test_directory, x))])

random_filename=test_directory+random_filename

print(random_filename)

img_array = cv2.imread(random_filename,cv2.IMREAD_COLOR)

RGB_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

new_img = cv2.resize(RGB_img,(img_size,img_size))

print('Prediction : {}'.format(model.predict_classes(new_img.reshape(-1,img_size,img_size,3))))

plt.imshow(new_img)

plt.show()

model.save_weights('./driverdistraction_lr_weights.h5', overwrite=True)
model.save('./driverdistraction.h5')