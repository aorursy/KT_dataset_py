# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop 
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
test.head()
trainLabels = train["label"]
train = train.drop(labels = ["label"],axis=1)
train = train.astype('float32')/255
trainLabels = trainLabels.astype('int32')

test= test.astype('float32')/255
trainLabels
train = train.values.reshape(42000,28,28,1)
test = test.values.reshape(28000,28,28,1)
meantr = train.mean().astype(np.float32)
stdtr = train.std().astype(np.float32)

def standardize(x):
    return (x-meantr)/stdtr
trainLabels = to_categorical(trainLabels)
train,validation,trainLabels,validationLabels = train_test_split(train,trainLabels,test_size=0.1)
model = Sequential()
model.add(layers.Conv2D(32, kernel_size = (3,3), activation = 'relu', input_shape = (28,28,1)))
model.add(layers.Conv2D(32,kernel_size = (3,3), activation = 'relu'))
model.add(MaxPool2D((2,2)))
model.add(layers.Conv2D(64,kernel_size = (3,3), activation = 'relu'))
model.add(layers.Conv2D(64,kernel_size = (3,3), activation = 'relu'))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

model.compile(optimizer=RMSprop(lr=0.001),loss="categorical_crossentropy",metrics = ["accuracy"])
reducingLearningRate = ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.5,min_lr=0.00001)
history = model.fit(train,trainLabels,batch_size=86,epochs=1,validation_data=(validation,validationLabels))
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
predicted_labels = model.predict(test)
predicted_labels = np.argmax(predicted_labels,axis=1)
predicted_labels=pd.Series(predicted_labels,name='Label')
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predicted_labels],axis = 1)
submission.to_csv("cnn_mnist_without_augmentation.csv",index=False)
