



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



train = pd.read_csv("../input/train.csv")

test  = pd.read_csv("../input/test.csv")



#print(train.head())



Y_train = train["label"]

X_train = train.drop(labels = ["label"], axis = 1)



#print(X_train.head())



Y_train.value_counts()



X_train.isnull().any().describe()





test.isnull().any().describe()



X_train = X_train / 255.0

test = test / 255.0



X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)



Y_train = to_categorical(Y_train, num_classes = 10)



random_seed = 2

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)



model = Sequential()

print(X_train.shape, Y_train.shape)



model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'same', activation = 'relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'same', activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))



optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)



model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])



history = model.fit(X_train, Y_train,epochs= 1, validation_data=(X_val, Y_val))



results = model.predict(test)



results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)

# Any results you write to the current directory are saved as output.