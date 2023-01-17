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
train_df = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv")
test_df = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_test.csv")
train_df.shape, test_df.shape
train_labels = train_df.pop("label")
true_labels = test_df.pop("label")


label_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
train_X = train_df
test_X = test_df

train_X = train_X.values.reshape(60000, 28, 28)
test_X = test_X.values.reshape(10000, 28, 28)

train_X = train_X / 255.0
test_X = test_X / 255.0
plt.figure()
plt.imshow(train_X[0])
plt.colorbar()
plt.grid=False
plt.show()
plt.figure(figsize=(11, 11))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid=(False)
    plt.imshow(train_X[i])
    plt.xlabel(label_names[train_labels[i]])
    plt.colorbar()

plt.show()
train_X = train_X.reshape((train_X.shape[0], 28, 28, 1))
test_X = test_X.reshape((test_X.shape[0], 28, 28, 1))

train_X.shape, test_X.shape
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (5, 5), activation="relu",
                           kernel_initializer="uniform",
                           padding="same",
                           kernel_regularizer=tf.keras.regularizers.l2(0.001),
                           input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (5, 5), padding="same",
                           kernel_initializer="uniform",
                           kernel_regularizer=tf.keras.regularizers.l2(0.001),
                           activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu",
                           padding="same",
                           kernel_initializer="uniform",
                           kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu",
                           padding="same",
                           kernel_initializer="uniform",
                           kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation="relu",
                         kernel_initializer="uniform",
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dense(10, activation="softmax"),
    ])

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-03, decay=0.0),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

history = model.fit(train_X, train_labels, epochs=7, batch_size=84,
                    validation_split=0.2, verbose=1)

score = model.evaluate(test_X, true_labels, verbose=0)

print(f"test loss: {score[0]} / test accuracy: {score[1]*100}")
hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch

plt.figure(figsize=(5, 5))
plt.plot(hist["epoch"], hist["accuracy"], label="Train Acc:")
plt.plot(hist["epoch"], hist["val_accuracy"], label="Val Acc:")
plt.xlabel("Epochs")
plt.ylabel("Train, Val Acc")
plt.legend()
plt.show()

plt.figure(figsize=(5, 5))
plt.plot(hist["epoch"], hist["loss"], label="train loss")
plt.plot(hist["epoch"], hist["val_loss"], label="Val loss")
plt.xlabel=("Epochs")
plt.ylabel=("Train Val Loss")
plt.legend()
plt.show()
predictions = model.predict(test_X)

predictions[0]
true_labels[0]
test_X = test_X.reshape(10000, 28, 28)
test_X.shape
def plot_image(i, predictions_array, true_labels, img):
    predictions_array, true_labels, img = predictions_array, true_labels[i], img[i]
    plt.grid=False
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_labels:
        color = 'blue'
    else:
        color = 'red'
        
    

def plot_value_array(i, predictions_array, true_labels):
    true_labels = true_labels[i]
    plt.grid=False
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_labels].set_color('blue')
i = 0
plt.figure(figsize=(5, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], true_labels, test_X)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], true_labels)

plt.show()
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], true_labels, test_X)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], true_labels)

plt.tight_layout()
plt.show
predictions = [np.argmax(y, axis=None, out=None) for y in predictions]
predictions[:7]