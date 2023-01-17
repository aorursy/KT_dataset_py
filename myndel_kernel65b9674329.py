import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import cv2

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

from keras.layers.normalization import BatchNormalization

from keras.optimizers import SGD
TRAIN_PATH = '../input/fruits-fresh-and-rotten-for-classification/dataset/train/'

TEST_PATH = '../input/fruits-fresh-and-rotten-for-classification/dataset/test/'

SIZE = (112, 112)
CLASSES = []

for class_ in os.listdir(TRAIN_PATH):

    CLASSES.append(class_)

NUM_CLASSES = len(CLASSES)



print(CLASSES)
def load_data(path):

    X, y = [], []

    

    for label in CLASSES:

        for img in os.listdir(os.path.join(path, label))[:1466]:

            full_path = os.path.join(path, label, img)

            image = cv2.imread(full_path)

            image = cv2.resize(image, SIZE)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = np.asarray(image).astype(np.float32) / 255.0

            X.append(image)

            y.append(CLASSES.index(label))

            

    y = to_categorical(y, num_classes=NUM_CLASSES)

            

    return np.array(X), np.array(y)
X_train, y_train = load_data(TRAIN_PATH)
X_train.shape
y_train.shape
plt.figure(figsize=(25, 25)) # specifying the overall grid size



for i in range(25):

    plt.subplot(5,5,i+1)    # the number of images in the grid is 5*5 (25)

    plt.imshow(X_train[i*200])



plt.show()
X_test, y_test = load_data(TEST_PATH)
X_test.shape
model = Sequential()



model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu', input_shape=X_train.shape[1:]))

model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.2))



model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.2))



model.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.2))



model.add(Flatten())



model.add(Dense(2048, activation='relu', kernel_initializer='he_uniform'))

model.add(Dropout(0.5))

model.add(Dense(2048, activation='relu', kernel_initializer='he_uniform'))

model.add(Dropout(0.5))



model.add(Dense(NUM_CLASSES, activation='softmax'))



opt = SGD(lr=1e-3, momentum=0.9)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
# Vizualize history

fig, ax = plt.subplots(2,1)



ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="Validation loss")

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)