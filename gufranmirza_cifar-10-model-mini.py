# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy as sp

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import keras 

import keras.backend as k

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D

from keras.layers import Dense,Dropout,Flatten

from keras.layers.normalization import BatchNormalization

from keras import losses

from keras.optimizers import Adam,RMSprop,Adadelta

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer

from keras.applications.vgg16 import preprocess_input,decode_predictions

import matplotlib.pyplot as plt

num_classes=10

#load the image data

train_data=np.load('../input/train_images.npy')

train_data=np.reshape(train_data,(50000,32,32,3))

train_data.shape

#Train labels

labels=pd.read_csv('../input/train_labels.csv')

print(labels.shape)

labels.head()

label_data=pd.Series(labels['Category'])

label_data
test_data=np.load('../input/test_images.npy')

test_data=np.reshape(test_data,(200000,32,32,3))

print(test_data.shape)

print(test_data)

#Normalization

lb=LabelBinarizer()

label_data=lb.fit_transform(label_data)

train_data=train_data/255

test_data=test_data/255

#Split the train dataset

X_train,X_val,Y_train,Y_val=train_test_split(train_data,label_data,test_size=0.2,random_state=431)

print(X_train.shape,X_val.shape)

print(Y_train.shape,Y_val.shape)
#CNN model

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(32,32,3)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))
#Visualization of CNN arcitecture

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model,show_shapes=True,show_layer_names=True,rankdir='TB').create(prog='dot',format='svg'))
#Compile the model

model.compile(loss=losses.categorical_crossentropy,optimizer=Adadelta(),metrics=['accuracy'])
#Training the model

cnn=model.fit(X_train,Y_train,batch_size=128,epochs=50,verbose=1,validation_data=(X_val,Y_val),shuffle=True)
#Plots fro training and validation process:loss and accuracy

plt.figure(figsize=(10,6))

plt.plot(cnn.history['acc'],'g')

plt.plot(cnn.history['val_acc'],'b')

plt.xticks(np.arange(1,60,2))

plt.title('training accuracy vs validation accuracy')

plt.xlabel('No. of Epochs')

plt.ylabel('Accuracy')

plt.legend(['train','validation'])

plt.show()



plt.figure(figsize=(10,6))

plt.plot(cnn.history['loss'],'g')

plt.plot(cnn.history['val_loss'],'b')

plt.xticks(np.arange(1,60,2))

plt.title('training loss vs validation loss')

plt.xlabel('No. of Epochs')

plt.ylabel('loss')

plt.legend(['train','validation'])

plt.show()



# Predict Against Training Data

index = 0

prediction =model.predict(X_train[:100],verbose=1) 

for p in prediction:

    print("Actual class", Y_train[index])

    print("Predicted class:", p ,"\n")

    index += 1