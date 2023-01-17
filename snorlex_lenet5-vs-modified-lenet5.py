# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_path = "../input/train.csv"

test_path = "../input/test.csv"

submission_path = "../input/sample_submission.csv"
train_df = pd.read_csv(train_path)

print(train_df.shape)
target = train_df["label"]

train_df.drop("label", axis=1, inplace=True)

train = train_df.values
import matplotlib.pyplot as plt

plt.imshow(train[1000].reshape(28, 28), cmap="gray")
# scaling

train = train.astype("float32") / 255

print(train[:5])
import tensorflow as tf

from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, AveragePooling2D, Flatten

from keras.layers import Dense, Dropout, BatchNormalization

from keras.optimizers import Adamax
y = to_categorical(target)

print(y[:5])
from sklearn.model_selection import train_test_split

train_x, val_x, train_y, val_y = train_test_split(train, y, test_size=0.2, random_state=42)
train_x = train_x.reshape(-1, 28, 28, 1)

val_x = val_x.reshape(-1, 28, 28, 1)
train_datagen = ImageDataGenerator()

val_datagen = ImageDataGenerator()
train_gen = train_datagen.flow(train_x, train_y, batch_size=100)

val_gen = val_datagen.flow(val_x, val_y, batch_size=100)
batch_size = 100

ss_train = int(train_x.shape[0] / batch_size)

ss_val = int(val_x.shape[0] / batch_size)
# LeNet5

model = Sequential()



model.add(Conv2D(6, kernel_size=(5,5), input_shape=(28, 28, 1), activation="tanh", padding="same"))

model.add(AveragePooling2D(2))



model.add(Conv2D(16, kernel_size=(5,5), activation="tanh"))

model.add(AveragePooling2D(2))



model.add(Conv2D(120, kernel_size=(5, 5), activation="tanh"))

model.add(Flatten())



model.add(Dense(84, activation="tanh"))

model.add(Dense(10, activation="softmax"))

adamax = Adamax(lr=0.001, decay=1e-6)

model.compile(loss="categorical_crossentropy", optimizer=adamax, metrics=['accuracy'])



model.summary()
history = model.fit_generator(train_gen, epochs=50, validation_data=val_gen, steps_per_epoch=ss_train, validation_steps=ss_val)
import matplotlib.pyplot as plt



plt.plot(history.history["acc"], c="b")

plt.plot(history.history["val_acc"], c="r")

plt.title("Accuracy Score")

plt.legend(["train_acc", "val_acc"])

plt.show()
# pad image

train_x1 = np.pad(train_x, ((0,0), (2,2), (2,2), (0,0)), 'constant')

val_x1 = np.pad(val_x, ((0,0), (2,2), (2,2), (0,0)), 'constant')

# reinitialise data generators

train_gen = train_datagen.flow(train_x1, train_y, batch_size=100)

val_gen = val_datagen.flow(val_x1, val_y, batch_size=100)
print(train_x1.shape)
# Modified LeNet5

model1 = Sequential()



model1.add(Conv2D(6, kernel_size=(3,3), input_shape=(32, 32, 1), activation="tanh"))

model1.add(Conv2D(6, kernel_size=(3, 3), activation="tanh"))

model1.add(AveragePooling2D(2))



model1.add(Conv2D(16, kernel_size=(3,3), activation="tanh"))

model1.add(Conv2D(16, kernel_size=(3,3), activation="tanh"))

model1.add(AveragePooling2D(2))



model1.add(Conv2D(120, kernel_size=(3, 3), activation="tanh"))

model1.add(Conv2D(120, kernel_size=(3, 3), activation="tanh"))

model1.add(Flatten())



model1.add(Dense(84, activation="tanh"))

model1.add(Dense(10, activation="softmax"))

adamax = Adamax(lr=0.001, decay=1e-6)

model1.compile(loss="categorical_crossentropy", optimizer=adamax, metrics=['accuracy'])



model1.summary()
history1 = model1.fit_generator(train_gen, epochs=50, validation_data=val_gen, steps_per_epoch=ss_train, validation_steps=ss_val)
plt.plot(history1.history["acc"], c="b")

plt.plot(history1.history["val_acc"], c="r")

plt.title("Accuracy Score")

plt.legend(["train_acc", "val_acc"])

plt.show()
test_df = pd.read_csv(test_path)

print(test_df.shape)
test = test_df.values.astype('float32') / 255

test = test.reshape(-1, 28, 28, 1)
pred = model.predict(test)
test1 = np.pad(test, ((0,0), (2,2), (2,2), (0,0)), 'constant')

pred1 = model1.predict(test1)
pred = np.argmax(pred, axis=1)

print(pred.shape)
pred1 = np.argmax(pred1, axis=1)

print(pred1.shape)
sub = pd.read_csv(submission_path)

sub["Label"] = pred.reshape((test.shape[0], 1))

sub.to_csv("submission.csv", index=False)
sub["Label"] = pred1.reshape((test.shape[0], 1))

sub.to_csv("submission1.csv", index=False)