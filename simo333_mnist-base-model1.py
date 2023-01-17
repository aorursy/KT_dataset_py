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
from tensorflow.keras import models, layers

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
train_df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test_df = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

train_df.shape
train_X, test_X = train_test_split(train_df, test_size=0.2, shuffle=False)
train_X.shape
train_y = train_X.pop("label")
test_y = test_X.pop("label")
train_y.shape
train_X = train_X.values
test_X = test_X.values
train_X = train_X.reshape(33600, 28, 28)
test_X = test_X.reshape(8400, 28, 28)

train_X, test_X = train_X / 255.0, test_X / 255.0

train_X.shape
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation="relu",
                          kernel_initializer="he_uniform",
                          kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dense(256, activation="relu", 
                          kernel_initializer="he_uniform",
                          kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10)
])


predictions = model(train_X[1:]).numpy()
predictions

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(train_y[1:], predictions).numpy()


model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
              loss=loss_fn, metrics=["accuracy"])

history = model.fit(train_X, train_y, epochs=7,
                    batch_size=128, validation_split=0.2)

val_test = model.evaluate(test_X, test_y)
hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch


def plot_history(history):
    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Train, Validation Accuracy")
    plt.plot(hist["epoch"], hist["accuracy"], label="Train Acc")
    plt.plot(hist["epoch"], hist["val_accuracy"], label="Val Acc")
    plt.legend()
    
    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Train, Validation Loss")
    plt.plot(hist["epoch"], hist["loss"], label="Train Loss")
    plt.plot(hist["epoch"], hist["val_loss"], label="Val Loss")
    plt.legend()
    plt.show()
    
    
plot_history(hist)
plot_history(val_test)
test_df.shape
test_df = test_df.values
test_df =test_df.reshape(28000, 28, 28)
test_df = test_df / 255.0
test_df.shape
pred = model.predict(test_df, verbose=1)
pred = [np.argmax(y, axis=None, out=None) for y in pred]
pred = pd.DataFrame(pred)
sub = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

sub["Label"] = pred
sub.head()
sub.to_csv("My_base_submission", index=False)