import os

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import RMSprop
training = pd.read_csv("../input/train.csv")

testing = pd.read_csv("../input/test.csv")

training.head(1)
a = training.iloc[42][1:].values.reshape(-1,28,28,1)

number = plt.imshow(a[0][:,:,0], cmap='Greys')
y_train = training["label"].values

X_train = training.drop("label", axis=1).values

X_val = testing.values
training["label"].value_counts()
rows = 28

columns = 28

input_shape = (rows, columns, 1) 

classes = len(set(y_train))



X_train = X_train.reshape(X_train.shape[0], rows, columns, 1).astype("float32")/255

X_val = X_val.reshape(X_val.shape[0], rows, columns, 1).astype("float32")/255



y_train = keras.utils.to_categorical(y_train, classes)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, 

                                                          test_size=0.1, 

                                                          random_state=123, 

                                                          stratify=y_train, 

                                                          shuffle=True

                                                         )
BATCH_SIZE = 64

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)



digit_model = Sequential()

digit_model.add(Conv2D(20,

                 kernel_size=(3,3),

                 activation='relu',

                 input_shape=(28,28,1)))

digit_model.add(Conv2D(20, kernel_size=(3,3), activation='relu', strides=2))

digit_model.add(Conv2D(20, kernel_size=(3,3), activation='relu'))

digit_model.add(Flatten())

digit_model.add(Dense(128, activation='relu'))

digit_model.add(Dense(classes, activation='softmax'))

digit_model.compile(loss=keras.losses.categorical_crossentropy,

                    optimizer=optimizer,

                    metrics=['accuracy'])
datagen = ImageDataGenerator(

        rotation_range=10, 

        zoom_range = 0.1,

        width_shift_range=0.1, 

        height_shift_range=0.1, 

        horizontal_flip=False,  

        vertical_flip=False 

        )  





datagen.fit(X_train)
history = digit_model.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),

                    steps_per_epoch=len(X_train) / 32, epochs=5, validation_data=(X_test, y_test))



predictions = digit_model.predict_classes(X_val, verbose=1)



predictions = np.column_stack((np.arange(1, 28001), predictions))

np.savetxt("predict.csv", predictions, fmt="%i", delimiter=",", header="ImageId,Label", comments="")
plt.style.use('ggplot')



def plot_history(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    x = range(1, len(acc) + 1)



    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(x, acc, 'b', label='Training acc')

    plt.plot(x, val_acc, 'r', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(x, loss, 'b', label='Training loss')

    plt.plot(x, val_loss, 'r', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()

    

plot_history(history)
a = testing.iloc[55].values.reshape(-1,28,28,1)

number = plt.imshow(a[0][:,:,0], cmap='Greys')
predictions[55][1]