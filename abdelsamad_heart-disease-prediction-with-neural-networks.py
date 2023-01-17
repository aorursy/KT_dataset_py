# Import packages that we will be working with.

import os

import numpy as np

import pandas as pd

from keras.layers import Dense

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense

from tensorflow.python.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score



print(os.listdir("../input"))

np.random.seed(10)
# Load the dataset, and view couple of the first rows.

data = pd.read_csv("../input/heart.csv")

print(data.head(3))



# Check the datatypes

print(data.dtypes)
# At this moment we have a dataframe that contains all of the heart.csv data. However we need to

# Separate them to [X, Y]. Where our target labels are 'Y', and 'X' is our training data.

Y = data.target.values

X = data.drop(['target'], axis=1)



# Now split to train/test with 80% training data, and 20% test data.

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



# Check dimensions of both sets.

print("Train Features Size:", X_train.shape)

print("Test Features Size:", X_test.shape)

print("Train Labels Size:", Y_train.shape)

print("Test Labels Size:", Y_test.shape)
# Define a Neural Network Model



def NN_model(learning_rate):

    model = Sequential()

    model.add(Dense(32, input_dim=13, kernel_initializer='normal', activation='relu'))

    model.add(Dense(16, kernel_initializer='normal', activation='relu'))

    model.add(Dense(2, activation='softmax'))

    Adam(lr=learning_rate)

    model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    return model
# Build a NN-model, and start training

learning_rate = 0.01

model = NN_model(learning_rate)

print(model.summary())
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=16, verbose=2)

# Plot the model accuracy vs. number of Epochs

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend(['Train', 'Test'])

plt.show()



# Plot the Loss function vs. number of Epochs

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['Train', 'Test'])

plt.show()
predictions = np.argmax(model.predict(X_test), axis=1)

model_accuracy = accuracy_score(Y_test, predictions)*100

print("Model Accracy:", model_accuracy,"%")

print(classification_report(Y_test, predictions))