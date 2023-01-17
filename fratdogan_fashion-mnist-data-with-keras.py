# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Data visualtion
import pandas as pd # Data manipulating
from tensorflow import keras # High-level api
import tensorflow as tf
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/fashion-mnist_train.csv")
test = pd.read_csv("../input/fashion-mnist_test.csv")
train_labels = train["label"]
train_images = train.drop("label", axis = 1)
test_labels = test["label"]
test_images = test.drop("label", axis = 1)
train_images = train_images / 255.0
test_images = test_images / 255.0
image_tag = ["T-shirt / Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
train_images.shape
img = train.loc[0, "pixel1":].values
img = img.reshape([28, 28])
plt.imshow(img, cmap = plt.cm.binary)
plt.show()
plt.figure(figsize = (18, 18))
for i in range(25):
    img = train_images.loc[i, :].values.reshape(28, 28)
    plt.subplot(5, 5, i+1)
    plt.imshow(img, cmap = "hot")
    plt.xlabel(image_tag[train.label[i]])
plt.show()
"""[
    keras.layers.Dense(128, input_shape =(784,)),
    keras.layers.Activation("relu"),
    keras.layers.Dense(10),
    keras.layers.Activation('softmax')
]"""
model = keras.Sequential()
model.add(keras.layers.Dense(64, input_dim = 784))
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.Dense(64))
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.Dense(10))
model.add(keras.layers.Activation("softmax"))
optimizer = keras.optimizers.Adam(lr = 0.0001)
model.compile(optimizer = optimizer, loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,write_graph=True, write_images=True)
model.fit(train_images, train_labels, epochs = 5, batch_size = 32, callbacks=[tbCallBack])
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("\n\nAccuracy:", test_accuracy)
print("Loss:", test_loss)
model.save("model.h5")