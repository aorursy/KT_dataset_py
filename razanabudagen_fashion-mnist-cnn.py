import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

test_data = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
print("Fashion MNIST train -  rows:",train_data.shape[0]," columns:", train_data.shape[1])

print("Fashion MNIST test -  rows:",test_data.shape[0]," columns:", test_data.shape[1])
labels = {0 : "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",

          5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}
train_data
def data_preprocessing(raw):

    out_y = keras.utils.to_categorical(raw.label, 10)

    num_images = raw.shape[0]

    x_as_array = raw.values[:,1:]

    x_shaped_array = x_as_array.reshape(num_images, 28, 28, 1)

    out_x = x_shaped_array / 255

    return out_x, out_y
X, y = data_preprocessing(train_data)

X_test, y_test = data_preprocessing(test_data)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2018)
print("Fashion MNIST train -  rows:",X_train.shape[0]," columns:", X_train.shape[1:4])

print("Fashion MNIST valid -  rows:",X_val.shape[0]," columns:", X_val.shape[1:4])

print("Fashion MNIST test -  rows:",X_test.shape[0]," columns:", X_test.shape[1:4])
model = keras.Sequential()



model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=(28, 28, 1)))

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(10, activation='softmax'))





model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])

model.summary()
train_model = model.fit(X_train, y_train,batch_size=128,epochs=50,verbose=1,

                        validation_data=(X_val, y_val))
score = model.evaluate(X_test, y_test, verbose=0)

print('Test accuracy:', score[1])
predicted_classes = model.predict_classes(X_test)



y_true = test_data.iloc[:, 0]
target_names = ["Class {} ({}) :".format(i,labels[i]) for i in range(10)]

print(classification_report(y_true, predicted_classes, target_names=target_names))