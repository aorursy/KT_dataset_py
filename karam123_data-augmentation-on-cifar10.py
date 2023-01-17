# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import shutil
from PIL import Image as PImage
from matplotlib import pyplot as plt
import keras
import tensorflow as tf
from keras.datasets import cifar10
data = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = data.load_data()
size = x_train.shape
size
x_train[0]
plt.imshow(x_train[1])
plt.show()
x_train[0]
samples  = x_train[0]
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(width_shift_range=[-0.2,0.2],height_shift_range= [-0.2,0.2])
# prepare iterator
it = datagen.flow(x_train[0:1,:] ,batch_size=1)
# generate samples and plot
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # generate batch of images
    batch = it.next()
    # convert to unsigned integers for viewing
    image = batch[0].astype('uint8')
    # plot raw pixel data
    plt.imshow(image)
data_lst =  []
label_lst = []
# prepare iterator
it = datagen.flow(x_train,y_train ,batch_size=1)
i = 0
for x_batch,y_batch in it:
    #plt.subplot(330 + 1 + i)
    image = x_batch[0].astype('uint8')
    data_lst.append(image)
    label_lst.append(y_batch[0])
    #print(image)
    #print(y_batch)
    i= i+1
    #plt.imshow(image)
    if(i>999):
        break
label_lst
final = []
for i in range(1000):
    final.extend(label_lst[i])
final
from collections import Counter
final = Counter(final)
final
data_augmented  = np.stack(data_lst,axis=0)
data_augmented.shape
final_data  = np.vstack((x_train,data_augmented))
final_data.shape
label_augmented = np.array(label_lst)
label_augmented.shape
y_final = np.vstack((y_train,label_lst))
y_final.shape
x_train = final_data.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
num_classes = 10

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_final, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
input_shape = (32,32,3)
batch_size =100
epochs =10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
model  = Sequential()

model.add(Conv2D(6,(5,5),activation = 'relu',input_shape = input_shape))
model.add((MaxPooling2D(pool_size = (2,2))))

model.add(Conv2D(16,(5,5),activation = 'relu'))
model.add(MaxPooling2D(pool_size= (2,2)))

model.add(Flatten())

model.add(Dense(120,activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(84,activation = 'relu'))

model.add(Dense(10,activation = 'softmax'))
model.summary()
model.compile(loss= 'categorical_crossentropy',optimizer = 'adam' , metrics = ['accuracy'])
history=model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
%matplotlib notebook
%matplotlib inline

import time
# https://gist.github.com/greydanus/f6eee59eaf1d90fcb3b534a25362cea4
# https://stackoverflow.com/a/14434334
# this function is used to update the plots for each epoch and error
def plt_dynamic(x, vy, ty, ax, colors=['b']):
    ax.plot(x, vy, 'b', label="Validation Loss")
    ax.plot(x, ty, 'r', label="Train Loss")
    #plt.legend()
    plt.grid()
    plt.show()
    fig.canvas.draw()
fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,epochs+1))

# print(history.history.keys())
# dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
# history = model_drop.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

# we will get val_loss and val_acc only when you pass the paramter validation_data
# val_loss : validation loss
# val_acc : validation accuracy

# loss : training loss
# acc : train accuracy
# for each key in histrory.histrory we will have a list of length equal to number of epochs

vy = history.history['val_accuracy']
ty = history.history['accuracy']
plt_dynamic(x, vy, ty, ax)