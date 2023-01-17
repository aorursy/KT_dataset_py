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
fasion_minst = keras.datasets.fashion_mnist
fasion_minst.load_data()
(X_train_full,y_train_full),(X_test,y_test) = fasion_minst.load_data()
import matplotlib.pyplot as plt

plt.imshow(X_train_full[900])
class_name = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
class_name[y_train_full[900]]
X_train_n = X_train_full/255.0
X_test_n = X_test/255.0
X_valid, X_train = X_train_n[:5000],X_train_n[5000:]
y_valid,y_train = y_train_full[:5000],y_train_full[5000:]
X_test = X_test_n
X_valid[900]

np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300,activation="relu"))
model.add(keras.layers.Dense(100,activation="relu"))
model.add(keras.layers.Dense(10,activation='softmax'))
model.summary()
keras.utils.plot_model(model)
weight,bias = model.layers[1].get_weights()
weight
model.compile(loss="sparse_categorical_crossentropy",
             optimizer="sgd",
             metrics=["accuracy"])
model_history = model.fit(X_train,y_train,epochs=30,validation_data=(X_valid,y_valid))
model_history.params
model_history.history
pd.DataFrame(model_history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
model_history.history
model.evaluate(X_test,y_test)
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)
Y_pred = model.predict_classes(X_new)
Y_pred
np.array(class_name)[Y_pred]
print(plt.imshow(X_new[0]))
print(plt.imshow(X_new[1]))
print(plt.imshow(X_new[2]))
