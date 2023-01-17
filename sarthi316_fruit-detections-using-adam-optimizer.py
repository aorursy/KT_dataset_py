# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.datasets import load_files

import numpy as np





def load_dataset(path):

    data = load_files(path)

    files = np.array(data['filenames'])

    targets = np.array(data['target'])

    target_labels = np.array(data['target_names'])

    return files,targets,target_labels

    

x_train, y_train,target_labels = load_dataset('/kaggle/input/fruits/fruits-360_dataset/fruits-360/Training/')

x_test, y_test,_ = load_dataset('/kaggle/input/fruits/fruits-360_dataset/fruits-360/Test/')

print('Loading complete!')



print('Training set size : ' , x_train.shape[0])

print('Testing set size : ', x_test.shape[0])
no_of_classes = len(np.unique(y_train))

print(no_of_classes)

print(y_train[0:10])
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train,no_of_classes)

y_test = np_utils.to_categorical(y_test,no_of_classes)

y_train[0]
x_test,x_valid = x_test[7000:],x_test[:7000]

y_test,y_vaild = y_test[7000:],y_test[:7000]

print('Vaildation X : ',x_valid.shape)

print('Vaildation y :',y_vaild.shape)

print('Test X : ',x_test.shape)

print('Test y : ',y_test.shape)
from keras.preprocessing.image import array_to_img, img_to_array, load_img



def convert_image_to_array(files):

    images_as_array=[]

    for file in files:

        # Convert to Numpy Array

        images_as_array.append(img_to_array(load_img(file)))

    return images_as_array



x_train = np.array(convert_image_to_array(x_train),dtype='float16')/255

print('Training set shape : ',x_train.shape)



x_valid = np.array(convert_image_to_array(x_valid),dtype='float16')/255

print('Validation set shape : ',x_valid.shape)



x_test = np.array(convert_image_to_array(x_test),dtype='float16')/255

print('Test set shape : ',x_test.shape)



print('1st training image shape ',x_train[0].shape)
x_train[0]
import matplotlib.pyplot as plt

%matplotlib inline



fig = plt.figure(figsize =(30,5))

for i in range(10):

    ax = fig.add_subplot(2,5,i+1,xticks=[],yticks=[])

    ax.imshow(np.squeeze(x_train[i]))
from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D

from keras.layers import Activation, Dense, Flatten, Dropout

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint

from keras import backend as K



model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = 3,input_shape=(100,100,3),padding='same'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=2,strides=2))



model.add(Conv2D(filters = 32,kernel_size = 3,activation= 'relu',padding='same'))

model.add(MaxPooling2D(pool_size=2,strides=2))



model.add(Conv2D(filters = 64,kernel_size = 3,activation= 'relu',padding='same'))

model.add(MaxPooling2D(pool_size=2,strides=2))



model.add(Conv2D(filters = 128,kernel_size = 3,activation= 'relu',padding='same'))

model.add(MaxPooling2D(pool_size=2,strides=2))



model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(150))

model.add(Activation('relu'))

model.add(Dropout(0.4))

model.add(Dense(120,activation = 'softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',

              optimizer='Adam',

              metrics=['accuracy'])

print('Compiled!')
batch_size = 32



checkpointer = ModelCheckpoint(filepath = 'own_cnn_with_adam.hdf5', verbose = 1, save_best_only = True)



history = model.fit(x_train,y_train,

        batch_size = 32,

        epochs=30,

        validation_data=(x_valid, y_vaild),

        callbacks = [checkpointer],

        verbose=2, shuffle=True)
model.load_weights('own_cnn_with_adam.hdf5')
score = model.evaluate(x_test, y_test, verbose=0)

print('\n', 'Test accuracy:', score[1])
y_pred = model.predict(x_test)

print(y_pred)

print(x_test.shape)

# plot a random sample of test images, their predicted labels, and ground truth

fig = plt.figure(figsize=(16, 9))

for i, idx in enumerate(np.random.choice(x_test.shape[0], size=16, replace=False)):

    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])

    ax.imshow(np.squeeze(x_test[idx]))

    pred_idx = np.argmax(y_pred[idx])

    true_idx = np.argmax(y_test[idx])

    ax.set_title("{} ({})".format(target_labels[pred_idx], target_labels[true_idx]),

                 color=("green" if pred_idx == true_idx else "red"))
import matplotlib.pyplot as plt 

plt.figure(1)  

   

 # summarize history for accuracy  

   

plt.subplot(211)  

plt.plot(history.history['accuracy'])  

plt.plot(history.history['val_accuracy'])  

plt.title('model accuracy')  

plt.ylabel('accuracy')  

plt.xlabel('epoch')  

plt.legend(['train', 'test'], loc='upper left')  

   

 # summarize history for loss  

   

plt.subplot(212)  

plt.plot(history.history['loss'])  

plt.plot(history.history['val_loss'])  

plt.title('model loss')  

plt.ylabel('loss')  

plt.xlabel('epoch')  

plt.legend(['train', 'test'], loc='upper left')  

plt.show()