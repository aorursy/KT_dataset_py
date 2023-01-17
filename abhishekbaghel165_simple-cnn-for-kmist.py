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
from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout,BatchNormalization
#Load dataset

import pandas as pd



X = np.load('../input/kmnist-train-imgs.npz')['arr_0']

X = X.reshape((X.shape[0],X.shape[1],X.shape[2],1))

y = np.load('../input/kmnist-train-labels.npz')['arr_0']

Y = to_categorical(y,num_classes = 10)

      

X_test = np.load('../input/kmnist-test-imgs.npz')['arr_0']

X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))

y_test = np.load('../input/kmnist-test-labels.npz')['arr_0']

Y_test = to_categorical(y_test,num_classes = 10)



labels = pd.read_csv('../input/kmnist_classmap.csv')

labels.head()

#Preprocess Dataset



from sklearn.model_selection import train_test_split



X_train,X_val,Y_train,Y_val = train_test_split(X,Y, test_size = 0.1)



print("Training dataset shape : " + str(X_train.shape) + " Labels : " + str(Y_train.shape))

print("Validation dataset shape : " + str(X_val.shape) + " Labels : " + str(Y_val.shape))

print("Test dataset shape : " + str(X_test.shape) + " Labels : " + str(Y_test.shape))

X_test = X_test.astype('float32')

X_train = X_train.astype('float32')

X_val = X_val.astype('float32')

X_test /= 255

X_train /= 255

X_val /= 255

import matplotlib.pyplot as plt

import random



ind = random.randint(0,X_train.shape[0])



plt.imshow(X_train[ind,:,:,0],cmap = 'gray')

plt.show()
#Define the model



def model():

    model = Sequential()

    model.add(Conv2D(16,(3,3), strides = (1,1), padding = 'same', activation = 'relu', input_shape = (28,28,1)))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size = (2,2)))

    model.add(Conv2D(32,(3,3), strides = (1,1), padding = 'same', activation = 'relu'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size = (2,2)))

    model.add(Conv2D(48,(3,3), strides = (1,1), padding = 'same', activation = 'relu'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size = (2,2)))

    model.add(Flatten())

    model.add(Dense(32,activation = "relu"))

    model.add(Dropout(rate = 0.45))

    model.add(Dense(10,activation = "softmax"))

    

    return model
model = model()

model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])
hist = model.fit(X_train,Y_train,verbose = 1,epochs = 100,validation_data=(X_val,Y_val))
import matplotlib.pyplot as plt



plt.title("Loss Graph")

plt.plot(hist.history['val_loss'],label = 'val_loss')

plt.plot(hist.history['loss'],label = 'loss')

plt.legend()

plt.show()
acc = model.evaluate(X_test,Y_test)

print("Accuracy is {}".format(acc[1]*100))

print("Loss is {}".format(acc[0]))