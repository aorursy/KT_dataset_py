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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import normalize, to_categorical
from sklearn.preprocessing import OneHotEncoder
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test.shape
X_train = normalize(x_train)
X_test = normalize(x_test)
Y_train = to_categorical(y_train)
Y_test = to_categorical(y_test)
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(784, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(X_train, Y_train, epochs=10, batch_size=50, validation_split=0.2)
loss = history.history["loss"]
val_loss = history.history["val_loss"]
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
n = np.arange(1, 11)
plt.plot(n, loss, label = "Training Loss")
plt.plot(n, val_loss, label = "Testing Loss")
plt.legend()
plt.show()
plt.plot(n, accuracy, label = "Training Accuracy")
plt.plot(n, val_accuracy, label = "Testing Accuracy")
plt.legend()
plt.show()
model_d = Sequential()
model_d.add(Flatten(input_shape = (28,28)))
model_d.add(Dense(784, activation="relu"))
model_d.add(Dropout(0.5))
model_d.add(Dense(130, activation="relu"))
model_d.add(Dropout(0.5))
model_d.add(Dense(10, activation="softmax"))
model_d.compile(loss = "categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model_d.fit(X_train, Y_train, epochs=10, batch_size=50, validation_split=0.2)
loss = history.history["loss"]
val_loss = history.history["val_loss"]
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
n = np.arange(1, 11)
plt.plot(n, loss, label = "Training Loss")
plt.plot(n, val_loss, label = "Testing Loss")
plt.legend()
plt.show()
plt.plot(n, accuracy, label = "Training Accuracy")
plt.plot(n, val_accuracy, label = "Testing Accuracy")
plt.legend()
plt.show()
x_predict = model.predict(X_test)
print(np.argmax(x_predict[704]))
plt.imshow(X_test[704], cmap="gray")
plt.show()
model.predict_classes(X_test)[704]