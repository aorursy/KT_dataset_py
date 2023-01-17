# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.utils import to_categorical
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# hyperparameters
img_side = 28
num_classes = 10
batch_size = 32
epochs = 1
train_data = pd.read_csv("../input/train.csv")
train_data.head()
test_data = pd.read_csv("../input/test.csv")
test_data.head()
train_data.describe()
sparse_labels = train_data.loc[:, "label"].values
print(sparse_labels)
labels = to_categorical(sparse_labels, num_classes)
print(labels)
print(labels.shape)
x_data = train_data.iloc[:, 1:].values
x_data = x_data / 255.0
print(x_data.shape)
x_data = x_data.reshape(-1, img_side, img_side, 1)
print(x_data.shape)

x_test = test_data.values
x_test = x_test / 255.0
x_test = x_test.reshape(-1, img_side, img_side, 1)
print(x_test.shape)
x_train = x_data[:len(x_data) * 9 // 10]
y_train = labels[:len(x_data) * 9 // 10]
x_val = x_data[-len(x_data) // 10:]
y_val = labels[-len(x_data) // 10:]

print(x_train.shape)
print(x_val.shape)
# build the model
mlp = Sequential()
mlp.add(Reshape((-1,), input_shape = (img_side, img_side, 1)))
mlp.add(Dense(img_side * img_side, activation = "relu"))
mlp.add(Dense(500, activation = "relu"))
mlp.add(Dense(10, activation = "softmax"))

print(mlp.input_shape)
print(mlp.output_shape)

mlp.compile(optimizer = Adam(),
           loss = "categorical_crossentropy",
           metrics = ["accuracy"])

data_gen = ImageDataGenerator()
data_gen.fit(x_train)
train_batches = data_gen.flow(x_train, y_train, batch_size = batch_size)
val_batches = data_gen.flow(x_val, y_val, batch_size = batch_size)

train_res = mlp.fit_generator(generator = train_batches, steps_per_epoch = train_batches.n,
                  epochs = epochs, validation_data = val_batches, validation_steps = val_batches.n)
print("Training loss: %.3f" % train_res.history["acc"][-1])
print("Validation loss: %.3f" % train_res.history["val_acc"][-1])
test_batches = data_gen.flow(x_test, batch_size = len(x_test), shuffle = False)
probs = mlp.predict_generator(test_batches, steps = 1)
sparse_preds = np.argmax(probs, axis = 1)
print(sparse_preds)

res = pd.DataFrame(sparse_preds, index = np.arange(len(sparse_preds)) + 1, columns = ["Label"])
res.to_csv("submission.csv", index_label = "ImageId")

test = pd.read_csv("submission.csv")
test.head()