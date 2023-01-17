# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import adam, Adadelta

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Shape of train data", train_df.shape)
print("Shape of test data", test_df.shape)
X = train_df.drop("label", axis = 1)
y = train_df["label"]
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, shuffle = True)
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
# Model parameter which we will use in training
#input_dim = 784
output_dim = 10
batch_size = 128
epoch = 13
n_row, n_col = 28, 28
# Normalization
x_train = x_train/255
x_val = x_val/255
print("X train shape", x_train.shape)
print("X test shape", x_val.shape)
from keras import backend as k
if k.image_data_format() == "channels_first":
    x_train = x_train.values.reshape(x_train.shape[0], 1, n_row, n_col)
    x_val = x_val.values.reshape(x_val.shape[0], 1, n_row, n_col)
    test_df = test_df.values.reshape(test_df.shape[0], 1, n_row, n_col)
    input_shape = (1, n_row, n_col)
else:
    x_train = x_train.values.reshape(x_train.shape[0], n_row, n_col, 1)
    x_val = x_val.values.reshape(x_val.shape[0], n_row, n_col, 1)
    test_df = test_df.values.reshape(test_df.shape[0], n_row, n_col, 1)
    input_shape = (n_row, n_col, 1)
# Converts into binary class problem
from keras.utils import to_categorical
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)
# Conv2D Model
model = Sequential()
model.add(Conv2D(16, kernel_size = (2, 2), input_shape = input_shape, activation = "relu"))
model.add(Conv2D(32, kernel_size = (3, 3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (3, 3)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (3, 3)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(234, activation = "relu"))
model.add(Dense(output_dim, activation = "softmax"))
model.compile(loss = categorical_crossentropy, metrics = ["accuracy"], optimizer = "adam")
history = model.fit(x_train, y_train, batch_size, epochs = epoch, verbose = 1, validation_data = (x_val, y_val))
# Accuracy on validation data
score = model.evaluate(x_val, y_val, verbose = 0)
print("Validation Score:", score[0])
print("Validation Accuracy", score[1])
# Plot validataion and train loss
x = range(1, epoch + 1)
val_loss = history.history["val_loss"]
train_loss = history.history["loss"]
plt.plot(x, val_loss, "b", label = "Validation loss")
plt.plot(x, train_loss, "r", label = "Train loss")
plt.xlabel("Epoch")
plt.ylabel("Categorical Crossentropy loss")
plt.legend()
plt.show()
# predict results
pred = model.predict(test_df)
idx_max = np.argmax(pred,axis = 1)
idx_max = pd.Series(idx_max, name = "Label")
sample_submission = pd.concat([pd.Series(list(range(1,len(pred) + 1)),name = "ImageId"),idx_max],axis = 1)
sample_submission.to_csv("sample_submission.csv", index = False)