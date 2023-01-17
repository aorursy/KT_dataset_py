import io

import csv

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn import datasets

from sklearn.decomposition import PCA
import pandas as pd

df = pd.read_csv("../input/2d_array_bin.csv")
x = np.array(df.drop('class', axis=1))

y = np.array(df['class'])



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)
print(x_train.shape, y_train.shape, x_test.shape)
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Reshape, Conv2D, AveragePooling2D, Flatten

from keras.layers import MaxPooling2D

from keras.optimizers import adam

model_name = "CNN"

model = Sequential()



model.add(Reshape(target_shape=(1, 28, 28), input_shape=(784,)))

model.add(Conv2D(kernel_size=(3, 3), filters=32, padding="same", data_format="channels_first", kernel_initializer="uniform", use_bias=False))

model.add(AveragePooling2D(pool_size=(2, 2), data_format="channels_first"))

model.add(Conv2D(kernel_size=(3, 3), filters=32, padding="same", data_format="channels_first", kernel_initializer="uniform", use_bias=False))

model.add(AveragePooling2D(pool_size=(2, 2), data_format="channels_first"))

model.add(Conv2D(kernel_size=(3, 3), filters=64, padding="same", data_format="channels_first", kernel_initializer="uniform", use_bias=False))

model.add(AveragePooling2D(pool_size=(2, 2), data_format="channels_first"))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(output_dim=256, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(output_dim=256, activation='relu'))

model.add(Dense(output_dim=10, activation='softmax'))



adam = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])



model.fit(x_train, y_train, epochs=50, batch_size=128)

y_pred = model.predict_classes(x_test)

output_prediction(y_pred, model_name)



y_all_pred[1] = y_pred