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
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
Y_train = train_data["label"]

X_train = train_data.drop(labels = ["label"], axis = 1)
X_train = X_train/255.0

test_data = test_data/255.0
X_train = X_train.values.reshape(-1, 28, 28, 1)
from keras.utils.np_utils import to_categorical

Y_train = to_categorical(Y_train, num_classes = 10)
Y_train.shape
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.2)
import matplotlib.image as mpimg

import matplotlib.pyplot as plt

plt.imshow(X_train[1][:,:,0])
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = "Same",

                activation = "relu", input_shape = (28, 28, 1)))

model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = "Same",

                activation = "relu", input_shape = (28, 28, 1)))



model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

#model.add(Dropout(0.25))



model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = "Same",

                activation = "relu"))

model.add(MaxPool2D(pool_size= (2, 2)))



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
from keras import optimizers

optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, Y_train, epochs=5, batch_size = 50, validation_split=0.2 )

#Y_train
test_data = test_data.values.reshape(-1, 28, 28, 1)

results = model.predict(test_data)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("results.csv",index=False)