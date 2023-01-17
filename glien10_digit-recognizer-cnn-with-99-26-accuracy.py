import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPool2D

from tensorflow.keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split
path = "/kaggle/input/digit-recognizer/"

df = pd.read_csv(path + "train.csv")
df.head(5)
y = np.array(df['label'])

X = df.drop(columns=['label'])

X = X.values.reshape(-1, 28, 28, 1)

X = X / 255.0
y_counts = {}

for number in y:

    if number not in y_counts:

        y_counts[number] = 1

    else:

        y_counts[number] += 1

y_counts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)
model = Sequential()



model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Flatten())

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(10, activation="softmax"))



model.compile(

    optimizer=RMSprop(lr=0.0001),

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy'],

)
epochs = 25

batch_size = 50
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

history
model.evaluate(X_test, y_test, verbose=2)
losses = history.history['loss']

accuracies = history.history['accuracy']

epochs = [ i for i in range(len(losses))]
plt.title("Loss")

plt.plot(epochs, losses, color="blue")

plt.show()
plt.title("Accuracy")

plt.plot(epochs, accuracies, color="blue")

plt.show()