import numpy as np

import pandas as pd 
# import the necessary packages

from sklearn.preprocessing import LabelBinarizer

from sklearn.preprocessing import MinMaxScaler



import glob

import cv2

import os

 

test = pd.read_csv("/kaggle/input/mldl-competition-1/test.csv")

train = pd.read_csv("/kaggle/input/mldl-competition-1/train.csv")

sampleSabmission = pd.read_csv("/kaggle/input/mldl-competition-1/sampleSubmission.csv")
train.head(3)

test.head(3)
sampleSabmission.head(3)
import tensorflow as tf



from tensorflow import keras

from tensorflow.keras import layers



print(tf.__version__)

column_names = ['Id' 'X','Y','Z','Time']

X_train_orig = train[["X", "Y", "Z"]]

X_test_orig  = test[["X", "Y", "Z"]]

Y_train_orig = train["Time"]

print(X_train_orig.shape)

print(X_test_orig.shape)

print(Y_train_orig.shape)

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler(feature_range=(-1, 1), copy=True)

scaler.fit(X_train_orig)

print("Maximum values of X_train (X, Y, Z): " + str(scaler.data_max_))

print("Minimum values of X_train (X, Y, Z): " + str(scaler.data_min_))



# Use the same transform for both train and test data:

X_train_norm = scaler.transform(X_train_orig)

X_test_norm = scaler.transform(X_test_orig)



# The inverse transform is also possible

#X_train_inv_orig = scaler.inverse_transform(X_train_norm)



#print(X_train)

#print(X_train_norm)

#print(X_train_orig - X_train)

from sklearn.model_selection import train_test_split

# create training and testing vars



X_train, X_val, y_train, y_val = train_test_split(X_train_norm, Y_train_orig, test_size=0.2)

print(X_train.shape)

print(y_train.shape)

print(X_val.shape)

print(y_val.shape)
model = keras.Sequential([

    keras.layers.Dense(128, input_dim=3, activation='relu'),

    keras.layers.Dense(6, activation='relu'),

    keras.layers.Dense(1, activation="linear")

])



print(model.summary())

 

model.compile(optimizer='adam',

              loss='MSE',

              metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5)
test_loss, test_acc = model.evaluate(X_val, y_val, verbose=0)

print('\nTest loss:', test_loss)

print('\nTest accuracy:', test_acc)
print('\nhistory dict:', history.history.keys())
import matplotlib.pyplot as plt



plt.figure(figsize=(7, 4))

plt.subplot(2, 1, 1)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.grid(True)

plt.show()

plt.subplot(2, 1, 2)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.grid(True)

plt.show()
# Generate predictions

predictions = model.predict(X_test_norm)
mySubmission = sampleSabmission

mySubmission["Predicted"] = predictions



mySubmission.head()



filename = 'IvanPredictions_1.csv'



mySubmission.to_csv(filename,index=False)



print('Saved file: ' + filename)