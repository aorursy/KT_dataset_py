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
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

sns.set()
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission.head()
train_data.head()
X_train = train_data.drop("label", axis=1)
y_train = train_data[["label"]]
X_test = test_data.copy()
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255
X_train = X_train.values.reshape((len(X_train), 28, 28, 1))
X_test = X_test.values.reshape((len(X_test), 28, 28, 1))

y_train = pd.get_dummies(y_train, columns=["label"])
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), padding="valid", input_shape=X_train.shape[1:]))
model.add(Activation("relu"))
model.add(Conv2D(64, kernel_size=(3, 3), padding="valid"))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=(3, 3), padding="valid"))
model.add(Activation("relu"))
model.add(Conv2D(128, kernel_size=(3, 3), padding="valid"))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.4))
model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(optimizer="adadelta", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
epochs = 50
batch_size = 256
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
hist = model.history.history
f = plt.figure(figsize=(13, 8))
plt.plot(hist["loss"], c="blue", label="train")
plt.plot(hist["val_loss"], c="orange", label="val")
plt.legend()
plt.title("Loss", fontsize=15)
plt.xlabel("epoch", fontsize=13)
plt.ylabel("categorical crossentropy", fontsize=13)
plt.show()
hist = model.history.history
f = plt.figure(figsize=(13, 8))
plt.plot(hist["acc"], c="blue", label="train")
plt.plot(hist["val_acc"], c="orange", label="val")
plt.legend()
plt.title("Accuracy", fontsize=15)
plt.xlabel("epoch", fontsize=13)
plt.ylabel("accuracy", fontsize=13)
plt.show()
y_pred = np.argmax(model.predict(X_test), axis=1)
result = pd.Series(y_pred, name="Label").to_frame().reset_index().rename(columns={"index": "ImageId"})
result["ImageId"] += 1
result.head()
result.to_csv("out.csv", index=False)
