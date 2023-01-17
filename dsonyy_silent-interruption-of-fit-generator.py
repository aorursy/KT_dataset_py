from keras import layers, models

from keras.datasets import mnist

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

import pandas as pd



train = pd.read_csv("../input/digit-recognizer/train.csv")

y_train = train["label"]

X_train = train.drop(labels=["label"], axis=1)



X_train /= 255

X_train = X_train.to_numpy().reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train)



model = models.Sequential()

model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu", padding="same", input_shape=(28, 28, 1)))

model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu", padding="same"))

model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), activation="relu", padding="same"))

model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), activation="relu", padding="same"))

model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(layers.Dropout(0.25))

model.add(layers.Flatten())

model.add(layers.Dense(256, activation="relu"))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(10, activation="softmax"))



datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)



model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["acc"])
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=32), 

                              epochs=30, 

                              steps_per_epoch=len(X_train))