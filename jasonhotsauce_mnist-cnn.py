from math import ceil

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_csv(path)
    Y = df['label']
    X = df.drop(labels=['label'], axis=1)
    X /= 255.
    X = X.values.reshape(-1, 28, 28, 1)
    return X, Y


def prepare_training_data(X, Y, validation_split=0.0):
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=validation_split, random_state=5)

    return X_train, X_val, Y_train, Y_val


def build_model():
    model = Sequential()
    model.add(Conv2D(32, 3, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Conv2D(64, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=1e-3, decay=1e-4), metrics=['accuracy'])
    return model


def fit(input_path, data_augmentation=False, batch_size=32, validation_split=0.0):
    X, Y = load_data(input_path)
    model = build_model()
    if data_augmentation:
        X_train, X_val, Y_train, Y_val = prepare_training_data(X, Y, validation_split=validation_split)
        data_gen = ImageDataGenerator(rotation_range=0.8, 
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      zoom_range=0.25,
                                      fill_mode='nearest')
        data_gen.fit(X_train)
        train_data_generator = data_gen.flow(X_train, Y_train, batch_size)
        return model, model.fit_generator(train_data_generator,
                            steps_per_epoch=int(ceil(len(X_train)/float(batch_size))),
                            epochs=4,
                            validation_data=(X_val, Y_val))
    else:
        return model, model.fit(X, Y, batch_size=32, epochs=4, validation_split=validation_split)

model, history = fit('../input/train.csv', data_augmentation=True)
fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes =ax[0])
legend = ax[0].legend(loc='best')

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
legend = ax[1].legend(loc='best')
test_df = pd.read_csv('../input/test.csv')
X_test = test_df.values.reshape(-1, 28, 28, 1)
results = model.predict(X_test)
results = np.argmax(results, axis = 1)
results = pd.Series(results, name='label')
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"), results], axis = 1)

submission.to_csv("mnist.csv", index=False)
