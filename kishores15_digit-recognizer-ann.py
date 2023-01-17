# Importing the required Libraries

import pandas as pd

import numpy as np

import tensorflow as tf

import os

import random

import matplotlib.pyplot as plt

%matplotlib inline
# Loading the directory of the datasets

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Global Variable

RANDOM_STATE = 123
# Looking into the sample submission

sample_sub = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

sample_sub.head()
# Loading the training data

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
train.shape
train.head()
# Spliting the features and labels

label = train["label"]

data = train.drop(labels = "label", axis = 1)
# Loading the testing dataset

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
test.shape
test.head()
# Reshaping the dimensions of the training data

train_data = data.values.reshape(-1, 784)

train_data.shape
# Reshaping the dimensions of the testing data

test_data = test.values.reshape(-1, 784)

test_data.shape
# Visualizing the training and testing data

idx = random.randrange(1, train_data.shape[0])



plt.imshow(train_data[idx].reshape(28, 28), cmap = "gray")

plt.title(f"Label = {label[idx]}")

plt.axis("off")

plt.show()
# Normalizing the inputs

train_data = train_data / 255.0

test_data = test_data / 255.0
# One - Hot Encoding

labels = tf.keras.utils.to_categorical(label)

print(labels.shape)
# Training and validation split

from sklearn.model_selection import train_test_split



x_train, x_val, y_train, y_val = train_test_split(train_data, labels, test_size = 0.1, random_state = RANDOM_STATE)
print("X_Train shape: ", x_train.shape)

print("Y_Train shape: ", y_train.shape)

print("X_Val shape: ", x_val.shape)

print("Y_Val shape: ", y_val.shape)
# Model

model = tf.keras.models.Sequential()



model.add(tf.keras.layers.Dense(32, activation = "relu", input_shape = (784, )))

model.add(tf.keras.layers.Dense(32, activation = "relu"))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dropout(0.3))



model.add(tf.keras.layers.Dense(64, activation = "relu"))

model.add(tf.keras.layers.Dense(128, activation = "relu"))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dropout(0.3))



model.add(tf.keras.layers.Dense(10, activation = "softmax"))
# Optimizer

optimizer = tf.keras.optimizers.Adam(lr = 0.001)



# Compiling the model

model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])
model.summary()
model_history = model.fit(x_train, y_train, batch_size = 32, epochs = 20, validation_data = (x_val, y_val))
# Visualizing Model Loss

plt.plot(range(1, 21), model_history.history["loss"], label = "Loss")

plt.plot(range(1, 21), model_history.history["val_loss"], label = "Validation Loss")

plt.title("Model Loss")

plt.xlim([1, 20])

plt.xlabel("Epochs")

plt.ylim([0, 1])

plt.ylabel("Loss")

plt.legend()

plt.show()
# Visualizing Model Accuracy

plt.plot(range(1, 21), model_history.history["accuracy"], label = "Accuracy")

plt.plot(range(1, 21), model_history.history["val_accuracy"], label = "Validation Accuracy")

plt.title("Model Accuracy")

plt.xlim([1, 20])

plt.xlabel("Epochs")

plt.ylim([0.75, 1])

plt.ylabel("Accuracy")

plt.legend()

plt.show()
x_train.shape
y_train.shape
# Predictions

preds = model.predict(x_val)
# Convert One-Hot Encoded labels into class labels

preds_class = np.argmax(preds, axis = 1)



y_true = np.argmax(y_val, axis = 1)
# Accuracy_Score

from sklearn.metrics import accuracy_score



print("Accuracy = {:.3f} %".format(accuracy_score(y_true, preds_class) * 100))
predictions = model.predict(test_data)

predictions = np.argmax(predictions, axis = 1)

predictions = pd.Series(predictions, name = "Label")
img_id = pd.Series(range(1, test_data.shape[0] + 1), name = "ImageId")
submission = pd.concat([img_id, predictions], axis = 1)
submission.to_csv("Digit_Recognizer_ANN.csv", index = False)