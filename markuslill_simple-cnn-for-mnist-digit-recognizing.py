import keras
import pandas as pd

# Load the data
df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
import numpy as np
from keras.utils import to_categorical

y_train = df["label"]
y_train = to_categorical(y_train)

X_train = df.drop("label", axis=1)
X_train = np.array(X_train).reshape(-1, 28, 28)

print(y_train.shape)
print(X_train.shape)
%matplotlib inline

import matplotlib.pyplot as plt

i = 6541

plt.imshow(X_train[i])
plt.show()
print(y_train[i])
from keras.preprocessing.image import ImageDataGenerator

gen = ImageDataGenerator(width_shift_range=3, height_shift_range=3)
X_train_reshaped = X_train.reshape(-1, 28, 28,1)
gen.fit(X_train_reshaped)

# Print one shifted digit
for batch in gen.flow(X_train_reshaped, y_train, shuffle=True):
    first_image = batch[0][0]
    plt.imshow(first_image.reshape(-1, 28, 28)[0])
    plt.show()
    break

X_train_shifted = gen.flow(X_train.reshape(-1, 28, 28,1), y_train, shuffle=True)
import math

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping

early_stopping_monitor = EarlyStopping(patience=10, monitor="loss")

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation="relu", padding="same"))
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))

model.add(Dense(10, activation="sigmoid"))

model.compile(optimizer=RMSprop(lr=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(
    X_train.reshape(-1, 28, 28, 1),
    y_train,
    epochs=9999,
    batch_size=1000,
    callbacks=early_stopping_monitor
)
res_train = model.evaluate(X_train.reshape(-1, 28, 28, 1), y_train)
for i in range(0, len(res_train)):
    print(model.metrics_names[i] + ": " + str(res_train[i]))
df = pd.read_csv("../input/digit-recognizer/test.csv")
df.shape
X_test = np.array(df).reshape(-1, 28, 28)
res = model.predict_classes(X_test.reshape(-1, 28, 28, 1))
res.shape
res_df = pd.DataFrame(res, index=range(1, 28001), columns=["Label"])
res_df.index.name="ImageId"
print(res_df)
res_df.to_csv("result.csv")