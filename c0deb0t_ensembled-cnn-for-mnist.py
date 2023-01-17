# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.utils import *
from keras.models import *
from keras.callbacks import *
from keras.layers.normalization import *
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

%matplotlib inline
# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")
train_data.head()
test_data = pd.read_csv("../input/test.csv")
test_data.head()
y_train_sparse = train_data.loc[:, "label"].values
y_train = to_categorical(y_train_sparse)
print(y_train.shape)
x_train = train_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
print(x_train.shape)

x_val = x_train[-len(x_train) // 10:]
y_val = y_train[-len(y_train) // 10:]
y_val_sparse = y_train_sparse[-len(y_train_sparse) // 10:]
x_train = x_train[:len(x_train) * 9 // 10]
y_train = y_train[:len(y_train) * 9 // 10]
print(x_val.shape)
print(x_train.shape)

x_test = test_data.values.reshape(-1, 28, 28, 1) / 255.0
x = Input(shape = (28, 28, 1))
train_data_gen = ImageDataGenerator(width_shift_range = 0.1,
                                    height_shift_range = 0.1,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    rotation_range = 30)
test_data_gen = ImageDataGenerator()
train_data_gen.fit(x_train)
test_data_gen.fit(x_train)
train_batches = train_data_gen.flow(x_train, y_train, batch_size = 64)
val_batches = test_data_gen.flow(x_val, y_val, shuffle = False, batch_size = 64)
test_batches = test_data_gen.flow(x_test, shuffle = False, batch_size = 64)
pred = Conv2D(32, (5, 5), padding = "same", activation = "relu")(x)
pred = BatchNormalization()(pred)
pred = MaxPooling2D(pool_size = (2, 2))(pred)
pred = Dropout(0.25)(pred)

pred = Conv2D(64, (5, 5), padding = "same", activation = "relu")(pred)
pred = BatchNormalization()(pred)
pred = MaxPooling2D(pool_size = (2, 2))(pred)
pred = Dropout(0.25)(pred)

pred = Flatten()(pred)
pred = Dense(1024, activation = "relu")(pred)
pred = BatchNormalization()(pred)
pred = Dropout(0.5)(pred)
pred = Dense(10, activation = "softmax")(pred)

model1 = Model(inputs = x, outputs = pred)
model1.compile(optimizer = "adam",
               loss = "categorical_crossentropy",
               metrics = ["accuracy"])
model1.summary()
model1_history = model1.fit_generator(train_batches,
                                      epochs = 30,
                                      validation_data = val_batches,
                                      callbacks = [EarlyStopping(patience = 5),
                                                   ReduceLROnPlateau(factor = 0.3, patience = 3, min_lr = 1e-4)])
plt.plot(model1_history.history["acc"], label = "train", color = "blue")
plt.plot(model1_history.history["val_acc"], label = "validation", color = "orange")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
plt.plot(model1_history.history["loss"], label = "train", color = "blue")
plt.plot(model1_history.history["val_loss"], label = "validation", color = "orange")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
pred = Conv2D(32, (5, 5), padding = "same", activation = "relu")(x)
pred = BatchNormalization()(pred)
pred = Conv2D(32, (5, 5), strides = 2, padding = "valid", activation = "relu")(pred)
pred = Dropout(0.25)(pred)

pred = Conv2D(64, (5, 5), padding = "same", activation = "relu")(pred)
pred = BatchNormalization()(pred)
pred = Conv2D(64, (5, 5), strides = 2, padding = "valid", activation = "relu")(pred)
pred = Dropout(0.25)(pred)

pred = Conv2D(128, (1, 1), padding = "valid", activation = "relu")(pred)
pred = BatchNormalization()(pred)
pred = Dropout(0.5)(pred)
pred = Conv2D(10, (1, 1), padding = "valid")(pred)
pred = GlobalAveragePooling2D()(pred)
pred = Activation("softmax")(pred)

model2 = Model(inputs = x, outputs = pred)
model2.compile(optimizer = "adam",
               loss = "categorical_crossentropy",
               metrics = ["accuracy"])
model2.summary()
model2_history = model2.fit_generator(train_batches,
                                      epochs = 30,
                                      validation_data = val_batches,
                                      callbacks = [EarlyStopping(patience = 5),
                                                   ReduceLROnPlateau(factor = 0.3, patience = 3, min_lr = 1e-4)])
plt.plot(model2_history.history["acc"], label = "train", color = "blue")
plt.plot(model2_history.history["val_acc"], label = "validation", color = "orange")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
plt.plot(model2_history.history["loss"], label = "train", color = "blue")
plt.plot(model2_history.history["val_loss"], label = "validation", color = "orange")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
models = [model1, model2]
pred = Average()([model.outputs[0] for model in models])
ensemble = Model(inputs = x, outputs = pred)

val_preds = ensemble.predict_generator(val_batches)
print(val_preds.shape)
val_preds = np.argmax(val_preds, axis = 1)
val_acc = np.sum(val_preds == y_val_sparse) / len(y_val_sparse)
print("Ensemble validation accuracy: %s" % val_acc)
test_preds = np.argmax(ensemble.predict_generator(test_batches), axis = 1)
res = pd.DataFrame(test_preds, index = np.arange(len(test_preds)) + 1, columns = ["Label"])
res.to_csv("submission.csv", index_label = "ImageId")

test = pd.read_csv("submission.csv")
test.head()