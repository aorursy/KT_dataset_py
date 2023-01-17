import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import tensorflow

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense , Flatten

from tensorflow.keras.datasets import mnist

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split
df_train = pd.read_csv('../input/digit-recognizer/train.csv')

df_test = pd.read_csv('../input/digit-recognizer/test.csv')
print( "Test =",df_test.shape , "\nTrain =",df_train.shape)
df_test.head()
df_train.head()
x_train = df_train.iloc[:, 1:785]

y_train = df_train.iloc[:, 0]

x_test = df_test.iloc[:, 0:784]
scaller = StandardScaler()

x_test = scaller.fit_transform(x_test)

x_train = scaller.fit_transform(x_train)
model = Sequential()

# model.add(Flatten())

model.add(Dense(784, activation = "relu", input_shape = (784,)))

model.add(Dense(100, activation = "relu"))

model.add(Dense(300, activation = "relu"))

model.add(Dense(300, activation = "relu"))

model.add(Dense(10, activation = "softmax"))

model.compile(loss = "sparse_categorical_crossentropy", optimizer= "adam", metrics = ["accuracy"])

history = model.fit(x_train, y_train, validation_split = 0.30, epochs=30, batch_size=len(x_train))
acc = history.history["accuracy"]

val_acc= history.history["val_accuracy"]

loss = history.history["loss"]

val_loss = history.history["val_loss"]

print("Accuracy = ", acc[-1], "\nValidation Accuracy = ", val_acc[-1])
tva = pd.DataFrame(

        {

            "Ta":acc,

            "Va":val_acc

        }

)

tva.boxplot()
prediction = model.predict_classes(x_test)

print("first digit is ",prediction[0])
import matplotlib.pyplot as plt

image = x_test[0]

image = np.array(image, dtype='float')

pixels = image.reshape((28, 28))

plt.imshow(pixels, cmap='gray')

plt.show()