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
data_train=pd.read_csv('/kaggle/input/train.csv')

data_test=pd.read_csv('/kaggle/input/test.csv')

sample=pd.read_csv('/kaggle/input/sample_submission.csv')
data_train.head()
data_train.shape
y=data_train['label']

x=data_train.drop('label',axis=1)
x/=255

data_test/=255
x=x.values.reshape(-1,28,28,1)

data_test=data_test.values.reshape(-1,28,28,1)
# one hot encoding 

from sklearn.preprocessing import OneHotEncoder
y=pd.get_dummies(y)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=123)
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop, SGD, Adam,TFOptimizer

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.wrappers.scikit_learn import KerasClassifier

from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D

from keras.models import Model, Sequential
model=Sequential()

nb_classes=10



# 1 - Convolution

model.add(Conv2D(64,(3,3), padding='same', input_shape=(28, 28,1)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# 2- Convolution



model.add(Conv2D(128,(5,5),padding='same'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))





# 3- Convolution



model.add(Conv2D(512,(3,3),padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))





# Flattening

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))

model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

epochs = 10 # Turn epochs to 30 to get 0.9967 accuracy

batch_size = 86
# Without data augmentation i obtained an accuracy of 0.98114



history = model.fit(xtrain, ytrain, batch_size = batch_size, epochs = epochs, 

          validation_data = (xtest, ytest), verbose = 2)

ypred=model.predict(data_test)
from sklearn import metrics
# select the indix with the maximum probability

ypred = np.argmax(ypred,axis = 1)



results = pd.Series(ypred,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen1.csv",index=False)