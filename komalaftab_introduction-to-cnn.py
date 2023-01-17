import numpy as np

import pandas as pd

import tensorflow as tf

import seaborn as sns

np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

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



# Any results you write to the current directory are saved 
# loading data 

train_data = pd.read_csv("/kaggle/input/minist/train.csv")

test_data = pd.read_csv("/kaggle/input/minist/test.csv")

# display first five rows of train_data

train_data.head()
test_data.head()
# checking shape of train_data

train_data.shape # 

# checking shape of test_data



test_data.shape
# check the data

train_data.describe()
# check missing and null values

test_data.isnull().sum()
train_data.isnull().sum()
Y_train = train_data["label"]



# Drop 'label' column

X_train = train_data.drop(labels = ["label"],axis = 1) 



# free some space

del train_data 



g = sns.countplot(Y_train)



Y_train.value_counts()
# Normalize the data

X_train= X_train / 255.0

test_data= test_data / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , channel = 1)

X_train = X_train.values.reshape((-1,28,28,1))

test_data = test_data.values.reshape((-1,28,28,1))
test_data.shape
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

Y_train = to_categorical(Y_train, num_classes = 10)

# Set the random seed

random_seed = 2
# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
# Some examples

import matplotlib.pyplot as plt



h = plt.imshow(X_train[0][:,:,0])
k = plt.imshow(X_train[10][:,:,0])
from tensorflow.keras import layers

from tensorflow.keras import models



model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()
model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))
model.summary()
# Define the optimizer

#optimizer = rmsprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=30, batch_size=40)
test_loss, test_acc = model.evaluate(X_val, Y_val)

test_acc
results = model.predict(test_data)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_submission5.csv",index=False)