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
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout
from keras.datasets import cifar10

import matplotlib.pyplot as plt 
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print('Shape of training data and labels are - ',X_train.shape,y_train.shape)
print('Shape of test data and labels are - ',X_test.shape,y_test.shape)
# lets look at training data into much more detail 
print('Nummber of images - ',X_train.shape[0])
print('Dimensions of an image - ',X_train.shape[1:3])
print('Number of channels - ',X_train.shape[-1])
def show_channels(img):
    plt.imshow(img)
    plt.title('Original image')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    red_channel = img[:,:,0]
    green_channel = img[:,:,1]
    blue_channel = img[:,:,2]
    
    fit,ax = plt.subplots(1,3,figsize = (12,6))
    ax[0].imshow(red_channel,cmap = 'Reds')
    ax[0].set_title('Red Channel')
    ax[1].imshow(green_channel,cmap = 'Greens')
    ax[1].set_title('Green Channel')
    ax[2].imshow(blue_channel,cmap = 'Blues')
    ax[2].set_title('Blue channel')
    for i in range(3):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
idx = np.random.randint(50000)
show_channels(X_train[idx])
labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
sample = np.random.choice(np.arange(50000),10) #to get random indices
 

fig, axes = plt.subplots(2, 5, figsize=(12,4))
axes = axes.ravel()

for i in range(10):
    idx = sample[i]
    axes[i].imshow(X_train[idx])
    axes[i].set_title(labels[y_train[idx][0]])
    axes[i].axis('off')
    plt.subplots_adjust(wspace=1)


# normalizing the data
X_train = X_train / 255
X_test = X_test / 255 
#one hot encoding
num_classes = 10
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)
len_flatten = np.product(X_train.shape[1:])
X_train_flatten = X_train.reshape(X_train.shape[0],len_flatten)
X_test_flatten = X_test.reshape(X_test.shape[0],len_flatten)
print(X_train_flatten.shape,y_train.shape)
model = Sequential()

model.add(Dense(units=1024, activation='relu', input_shape=(len_flatten,)))
model.add(Dropout(0.2))
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_flatten, y_train, epochs=50, validation_split=.3)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()


_,test_acc = model.evaluate(X_test_flatten, y_test)
print('Test accuracy is ',test_acc)
# now lets apply this function to get the 
X_train_g1 = np.mean(X_train, -1)
X_test_g1 = np.mean(X_test,-1)
def show_org_and_modified(org,mod):
    fig,ax = plt.subplots(1,2,figsize = (8,16))
    ax[0].imshow(org)
    ax[0].set_title('original image')
    ax[1].imshow(mod,cmap='Greys')
    ax[1].set_title('grey scaled')

    ax[0].axis('off')
    ax[1].axis('off')
idx = np.random.randint(50000)
show_org_and_modified(X_train[idx],X_train_g1[idx])
len_flatten = np.product(X_train_g1.shape[1:])
X_train_g1 = X_train_g1.reshape(X_train_g1.shape[0],len_flatten)
X_test_g1 = X_test_g1.reshape(X_test_g1.shape[0],len_flatten)

print('Shape of data is - ')
print(X_train_g1.shape,y_train.shape)

model = Sequential()

model.add(Dense(units=512, activation='relu', input_shape=(len_flatten,)))
model.add(Dropout(0.2))

model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=64, activation='relu'))

model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_g1, y_train, epochs=50, validation_split=.3)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()


_,test_acc = model.evaluate(X_test_g1, y_test)
print('Test accuracy is ',test_acc)
X_train_g2 = np.dot(X_train[...,:3], [0.299, 0.587, 0.114])
X_test_g2 = np.dot(X_test[...,:3], [0.299, 0.587, 0.114])
X_train_g2.shape
idx = np.random.randint(50000)
show_org_and_modified(X_train[idx],X_train_g2[idx])
len_flatten = np.product(X_train_g2.shape[1:])
X_train_g2 = X_train_g2.reshape(X_train_g2.shape[0],len_flatten)
X_test_g2 = X_test_g2.reshape(X_test_g2.shape[0],len_flatten)

print('Shape of data is - ')
print(X_train_g2.shape,y_train.shape)


model = Sequential()

model.add(Dense(units=512, activation='relu', input_shape=(len_flatten,)))
model.add(Dropout(0.2))

model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=64, activation='relu'))

model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))

model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_g2, y_train, epochs=50, validation_split=.3)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()



_,test_acc = model.evaluate(X_test_g2, y_test)
print('Test accuracy is ',test_acc)
