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
#loading the libraries 
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
%matplotlib inline
import random


#for neural network
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout
path_training_images = '../input/mnist-dataset/train-images.idx3-ubyte'
path_training_labels = '../input/mnist-dataset/train-labels.idx1-ubyte'

path_test_images = '../input/mnist-dataset/t10k-images.idx3-ubyte'
path_test_labels = '../input/mnist-dataset/t10k-labels.idx1-ubyte'
#for loading the files
X_train, y_train = loadlocal_mnist(path_training_images,path_training_labels)
X_test, y_test = loadlocal_mnist(path_test_images, path_test_labels)
print(type(X_train),type(y_train))
# we have the data in numpy arrays
print('Dim of X train is ',X_train.shape)
print('Dim of y train is ',y_train.shape)
print('Dim of X test is ',X_test.shape)
print('Dim of y test is ',y_test.shape)
ex_image = X_train[0]
ex_label = y_train[0]

print('Shape of ex is ',ex_image.shape)
print('label of ex is ',ex_label)
print(ex_image)
# pixel values are from 0 to 255
#0 - black, 255 - white
#but the above one is not that a good visualization..Lets plot the pixels of this example
plt.imshow(ex_image.reshape(28,28),cmap=plt.cm.gray)
plt.title('This image represents {}'.format(ex_label))
plt.xticks([])
plt.yticks([])
def show(image, title):
    index = 1 
    plt.figure(figsize=(10,5))

    for x in zip(image, title):        
        image = x[0]        
        title = x[1]
        plt.subplot(2, 5, index)        
        plt.imshow(image.reshape(28,28), cmap=plt.cm.gray)  
        plt.title(x[1], fontsize = 9)
        plt.xticks([])
        plt.yticks([])
        index += 1
image = []
title = []
for i in range(0, 5):
    r = random.randint(1, len(X_train))
    image.append(X_train[r])
    title.append('training image:' + str(y_train[r]))       

for i in range(0, 5):
    r = random.randint(1, len(X_test))
    image.append(X_test[r])
    title.append('testing image:' + str(y_test[r]))
    
show(image, title)
#before model creation, we need to one hot encode labels
num_classes = 10 # from 0 to 0
y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)
print(X_train.shape,y_train.shape)
print(y_train[0])
model = Sequential()

model.add(Dense(units=128, activation='relu', input_shape=(784,)))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))
model.summary()
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

#fitting the model
history = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=.1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()
_,test_acc = model.evaluate(X_test, y_test)
print('Test accuracy is ',test_acc)


#normalize the data 
X_train = X_train / 255 
X_test = X_test / 255 

print(X_train[0])
# now the values are in between 0 to 1


model = Sequential()

model.add(Dense(units=128, activation='relu', input_shape=(784,)))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))

model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=.1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

_,test_acc = model.evaluate(X_test, y_test)
print('Test accuracy is ',test_acc)
model = Sequential()

model.add(Dense(units=256, activation='relu', input_shape=(784,)))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))

model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=.1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

_,test_acc = model.evaluate(X_test, y_test)
print('Test accuracy is ',test_acc)
model = Sequential()

model.add(Dense(units=256, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))

model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=.1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

_,test_acc = model.evaluate(X_test, y_test)
print('Test accuracy is ',test_acc)
sample = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
sample.head()
test = pd.read_csv('../input/digit-recognizer/test.csv')
print(test.shape)
test.head()
test = np.array(test)
test.shape
# normalize the data 
test = test / 255
#predictions 
preds = model.predict(test)
preds[0]
pred_classes = np.argmax(preds,axis=1)
pred_classes.shape
# lets look at some predictions made
show(test[:5],pred_classes[:5])
submission = pd.DataFrame({'ImageId':np.arange(1,28001),'Label':pred_classes})
submission.head()
submission.to_csv('sub.csv',index = False)
