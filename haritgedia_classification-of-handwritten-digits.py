# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# Any results you write to the current directory are saved as output.
import h5py
import cv2
from tensorflow.python.keras.applications.resnet import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow import keras
import matplotlib.pylab as plt

img_paths=[]
letters2 = pd.read_csv("../input/classification-of-handwritten-letters/letters2.csv")
num_ex=len(letters2)
for k in range(0,len(letters2)):
    img_paths.append('../input/classification-of-handwritten-letters/letters2/'+ letters2['file'][k])
image_size = 32
def read_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs=[load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    output=output/255
    return(output)
test_data=read_prep_images(img_paths)
x=(np.dot(test_data,[0.299, 0.587, 0.114]))
y=letters2['label']
plt.imshow(x[1000], cmap=plt.cm.bone)

#print(np.shape(y))
y = keras.utils.to_categorical(y-1,num_classes=None)
x=x.reshape(-1,32,32,1)
print(np.shape(y))
def history_plot(fit_history, n):
    plt.plot(fit_history.history['loss'], color='slategrey', label='train')
    plt.plot(fit_history.history['val_loss'], color='blue', label='valid')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size = 0.2, 
                                                    random_state = 1)
n = int(len(x_test)/2)
x_valid, y_valid = x_test[:n], y_test[:n]
x_test, y_test = x_test[n:], y_test[n:]
# Print the shape
print ("Training tensor's shape:", x_train.shape)
print ("Training target's shape", y_train.shape)
print ("Validating tensor's shape:", x_valid.shape)
print ("Validating target's shape", y_valid.shape)
print ("Testing tensor's shape:", x_test.shape)
print ("Testing target's shape", y_test.shape)
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, Activation, GlobalMaxPooling2D, MaxPooling2D

def model():
    model = Sequential()

    model.add(Conv2D(32, (5, 5), padding='same', input_shape=(32, 32,1)))
    model.add(Activation('relu'))    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(196, (5, 5)))
    model.add(Activation('relu'))    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(GlobalMaxPooling2D()) 
    
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5)) 
    
    model.add(Dense(33))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

model2 = model()
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout

model=Sequential()
model.add(Conv2D(32,
                 activation='relu',
                 kernel_size=5,
                 input_shape=(32, 32,1),
                 padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(20,
                 kernel_size=5,
                 activation='relu'))
model.add(Dropout(0.25))
model.add(Conv2D(20,activation='relu',
                        kernel_size=3))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(33, activation='softplus'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy'])
history=model.fit(x_train, y_train, 
                    epochs=50, batch_size=64, verbose=2,
                    validation_data=(x_valid, y_valid))

history_plot(history,0)
