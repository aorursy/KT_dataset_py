# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





from keras.layers import Dense, Dropout,Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.models import Sequential

from keras import layers

from keras import backend as K

from keras.datasets import mnist

from keras.utils import np_utils

import keras

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10,10) # Make the figures a bit bigger



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data_train_file = "../input/fashion-mnist_train.csv"

data_test_file = "../input/fashion-mnist_test.csv"



df_train = pd.read_csv(data_train_file)

df_test = pd.read_csv(data_test_file)
import matplotlib.pyplot as plt

%matplotlib inline



for i in range(5000, 5005):

    sample = np.reshape(df_test[df_test.columns[1:]].iloc[i].values/255, (28,28))

    plt.figure()

    plt.title("labeled class {}".format(df_test["label"].iloc[i]))

    plt.imshow(sample, 'gray')
X_train = df_train.drop(['label'], axis = 1).values

X_test = df_test.drop(['label'], axis = 1).values



Y_train =  df_train['label'].values

Y_test = df_test['label'].values
X_train = X_train.reshape(60000, 28, 28, 1)

X_test = X_test.reshape(10000, 28, 28, 1)
# Set numeric type to float32 from uint8

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')



# Normalize value to [0, 1]

X_train /= 255

X_test /= 255



# Transform lables to one-hot encoding

Y_train = np_utils.to_categorical(Y_train, 10)

Y_test = np_utils.to_categorical(Y_test, 10)
from keras.layers import Dropout



#Instantiate an empty model

model = Sequential()



# C1 Convolutional Layer

model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(28,28,1), padding="same"))



# S2 Pooling Layer

model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))



# C3 Convolutional Layer

model.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))



# S4 Pooling Layer

model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))



# C5 Fully Connected Convolutional Layer

model.add(layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))

#Flatten the CNN output so that we can connect it with fully connected layers

model.add(layers.Flatten())



# FC6 Fully Connected Layer

model.add(layers.Dense(84, activation='relu'))

model.add(Dropout(0.2))

#Output Layer with softmax activation

model.add(layers.Dense(10, activation='softmax'))



# Compile the model

model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])
hist = model.fit(x=X_train,y=Y_train, epochs=10, batch_size=128, validation_data=(X_test, Y_test), verbose=1)
result = model.evaluate(X_test, Y_test, verbose=0)

print('OUR  ACCURACY: ',result[1])
Xtrain = df_train.drop(['label'], axis = 1).values

Xtest = df_test.drop(['label'], axis = 1).values



Ytrain =  df_train['label'].values

Ytest = df_test['label'].values



predicted_classes = model.predict_classes(X_test)



# Check which items we got right / wrong

correct_indices = np.nonzero(predicted_classes == Ytest)[0]

incorrect_indices = np.nonzero(predicted_classes != Ytest)[0]



plt.figure()

for i, correct in enumerate(correct_indices[:9]):

    plt.subplot(3,3,i+1)

    plt.imshow(Xtest[correct].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], Ytest[correct]))

    

plt.figure()

for i, incorrect in enumerate(incorrect_indices[:9]):

    plt.subplot(3,3,i+1)

    plt.imshow(Xtest[incorrect].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], Ytest[incorrect]))