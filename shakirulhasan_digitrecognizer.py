# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, BatchNormalization, Dropout

from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.models import Sequential

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt



%matplotlib inline
train_df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_df = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
# Data Preprocessing

X_train = train_df.drop(["label"], axis=1)

Y_train = train_df["label"]



X_train = np.array(X_train).reshape(-1, 28, 28, 1)

Y_train = np.array(Y_train).reshape(-1, 1)

X_test = np.array(test_df).reshape(-1, 28, 28, 1)



X_train = X_train / 255

X_test = X_test / 255



Y_train = to_categorical(Y_train, num_classes=10)
n_train = X_train.shape[0]

n_test = X_test.shape[0]

n_classes = Y_train.shape[1]

image_size = X_train.shape[1:]



print(f"There are {n_train} training examples")

print(f"There are {n_test} testing examples")

print(f"There are {n_classes} classes")

print(f"Image shape: {image_size}")
datagen = ImageDataGenerator(

        rotation_range=10,  

        zoom_range = 0.10,  

        width_shift_range=0.1, 

        height_shift_range=0.1)

train_gen = datagen.flow(X_train, Y_train, batch_size=64)
index = 5

printing_digit = X_train[index].reshape(28, 28)

plt.imshow(printing_digit, cmap="gray")
model = Sequential([

    Input(shape=(28,28,1), name='digits'),

    Conv2D(32, (3,3), activation="relu"),

    BatchNormalization(),

    Conv2D(32, (3,3), activation="relu"),

    BatchNormalization(),

    Conv2D(32, (5,5), strides=2, padding="same", activation="relu"),

    BatchNormalization(),

    Dropout(0.4),

    

    Conv2D(64, (3,3), activation="relu"),

    BatchNormalization(),

    Conv2D(64, (3,3), activation="relu"),

    BatchNormalization(),

    Conv2D(64, (5,5), strides=2, padding="same", activation="relu"),

    BatchNormalization(),

    Dropout(0.4),

    

    Conv2D(128, (4,4), activation="relu"),

    BatchNormalization(),

    Flatten(),

    Dropout(0.4),

    Dense(10, activation="softmax")

])



model.compile(optimizer="Adam", loss=categorical_crossentropy, metrics=['accuracy'])



model.summary()
model.fit(train_gen, epochs=50)
Y_pred = model.predict(X_test)

Y_pred = np.argmax(Y_pred, axis=1)
index = 98

printing_digit = X_test[index].reshape(28, 28)

print(Y_pred[index])

plt.imshow(printing_digit, cmap="gray")
submission = pd.DataFrame({

    "ImageId": range(1, 28001),

    "Label": Y_pred.tolist()

})

submission.to_csv("/kaggle/working/submission.csv", index=False)