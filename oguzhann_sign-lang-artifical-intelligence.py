# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# standart data tools

import numpy as np

import pandas as pd



# common visualizing tools

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings("ignore")



# Any results you write to the current directory are saved as output.
X_train = np.load("../input/Sign-language-digits-dataset/X.npy")

Y_train = np.load("../input/Sign-language-digits-dataset/Y.npy")
# CNN layers and the Deep Learning model

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense , Flatten, Dropout

from keras.optimizers import Adam

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score



# splitting tool for the validation set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_train,Y_train,test_size=0.2,random_state=42)

x_train = x_train.reshape(-1,64,64,1)

x_test = x_test.reshape(-1,64,64,1)
model_sign_lang = Sequential()



model_sign_lang.add(Conv2D(filters=16,kernel_size=(5,5),activation="relu",padding="same",input_shape=(64,64,1)))

model_sign_lang.add(MaxPooling2D(pool_size=(2,2)))



model_sign_lang.add(Conv2D(filters=32,kernel_size=(4,4),activation="relu",padding="same"))

model_sign_lang.add(MaxPooling2D(pool_size=(2,2)))

model_sign_lang.add(Dropout(0.25))



model_sign_lang.add(Conv2D(filters=32,kernel_size=(3,3),activation="relu",padding="same"))

model_sign_lang.add(MaxPooling2D(pool_size=(2,2)))

model_sign_lang.add(Dropout(0.25))



model_sign_lang.add(Conv2D(filters=32,kernel_size=(2,2),activation="relu",padding="same"))

model_sign_lang.add(MaxPooling2D(pool_size=(2,2)))

model_sign_lang.add(Dropout(0.25))



model_sign_lang.add(Flatten())



model_sign_lang.add(Dense(128,activation="relu"))

model_sign_lang.add(Dense(64,activation="relu"))

model_sign_lang.add(Dense(30,activation="relu"))

model_sign_lang.add(Dense(10,activation="softmax"))



model_sign_lang.compile(optimizer=Adam(lr=0.0002),loss=keras.losses.categorical_crossentropy,metrics=["accuracy"])

    

model_sign_lang.summary()
model_sign_lang.compile(optimizer=Adam(lr=0.0002),loss=keras.losses.categorical_crossentropy,metrics=["accuracy"])
results = model_sign_lang.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test))
plt.figure(figsize=(12,7))



plt.suptitle("ACCURACY",fontsize=18)



plt.plot(results.history["val_acc"],label="validation_accuracy",c="red",linewidth=3)

plt.plot(results.history["acc"],label="training_accuracy",c="green",linewidth=3)

plt.legend()

plt.grid(True)



plt.show()



plt.figure(figsize=(12,7))

plt.plot(results.history["val_loss"],label="validation_loss",c="red",linewidth=3)

plt.plot(results.history["loss"],label="training_loss",c="green",linewidth=3)

plt.legend()

plt.grid(True)



plt.suptitle("LOSS",fontsize=18)

plt.show()
