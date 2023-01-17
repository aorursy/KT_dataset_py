##importing required libraries

import math

import numpy as np

import h5py

import matplotlib.pyplot as plt

import scipy

from PIL import Image

from scipy import ndimage

import tensorflow as tf

from tensorflow.python.framework import ops

get_ipython().magic('matplotlib inline')

np.random.seed(1)

import cv2

import numpy as np

import os

import pandas as pd

from sklearn.model_selection import train_test_split

##printing version of tensorflow

print(tf.__version__)
train_path="../input/mnist-in-csv/mnist_train.csv"

test_path="../input/mnist-in-csv/mnist_test.csv"

train = pd.read_csv(train_path)

print(train.shape)

print(train.head())

test= pd.read_csv(test_path)

print(test.shape)

print(test.head())

# put labels into y_train variable

y_orig1 = train["label"]

y_orig1_test = test["label"]

# Drop 'label' column

X_orig2 = train.drop(labels = ["label"],axis = 1,inplace=False)

X_orig2_test = test.drop(labels = ["label"],axis = 1,inplace=False)

# visualize number of digits classes

import seaborn as sns

plt.figure(figsize=(15,7))

g = sns.countplot(y_orig1, palette="icefire")

plt.title("Number of digit classes")

y_orig1.value_counts()



# visualize number of digits classes

import seaborn as sns

plt.figure(figsize=(15,7))

g = sns.countplot(y_orig1_test, palette="icefire")

plt.title("Number of digit classes")

y_orig1_test.value_counts()

# Example of a picture

index = 1000

img = X_orig2.iloc[index].values

img = img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(y_orig1[index])

plt.axis("off")

plt.show()



# Example of a picture

index = 1000

img = X_orig2_test.iloc[index].values

img = img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(y_orig1_test[index])

plt.axis("off")

plt.show()

# Normalize the data

X_orig1 = X_orig2 / 255.0

X_orig1_test = X_orig2_test / 255.0

print("x_orig1 shape: ",X_orig1.shape)

print("x_orig1_test shape: ",X_orig1_test.shape)



# Reshape

X_orig = X_orig1.values.reshape(-1,28,28,1)

X_orig_test = X_orig1_test.values.reshape(-1,28,28,1)

print("x_orig shape: ",X_orig.shape)

print("x_orig_test shape: ",X_orig_test.shape)



# Label Encoding 

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

print("old y_orig1 shape: ", y_orig1.shape)

y_orig = to_categorical(y_orig1, num_classes = 10)

print("y_orig shape: ",y_orig.shape)



print("old y_orig1_test shape: ", y_orig1_test.shape)

y_orig_test = to_categorical(y_orig1_test, num_classes = 10)

print("y_orig_test shape: ",y_orig_test.shape)

# Split the train and the validation set for the fitting

from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_orig, y_orig, test_size = 0.2, random_state=42)

print("x_train shape",X_train.shape)

print("x_val shape",X_val.shape)

print("y_train shape",Y_train.shape)

print("y_val shape",Y_val.shape)

# Some examples

index=1

print(Y_train[index])

plt.imshow(X_train[index][:,:,0],cmap='gray')

plt.show()

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



model = Sequential()

model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dense(10, activation = "softmax"))
# Define the optimizer

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
# Compile the model

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
epochs = 10  # for better result increase the epochs

batch_size = 200
%%time

history = model.fit(X_train,Y_train,validation_data=(X_val,Y_val),epochs=epochs

                    ,shuffle=True,steps_per_epoch=X_train.shape[0] // batch_size)
# Plot the loss and accuracy curves for training and validation 

plt.plot(history.history['val_loss'], color='b', label="validation loss")

plt.title("Val Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()

# confusion matrix

import seaborn as sns

# Predict the values from the validation dataset

Y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(Y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()
### Printing classification report

from sklearn.metrics import classification_report

target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8', 'class 9']

print(classification_report(Y_true, Y_pred_classes, target_names=target_names))
##Accuracy Score

from sklearn.metrics import accuracy_score

print("Accuracy on val sample is: %f" %(accuracy_score(Y_true, Y_pred_classes)))
Y_pred_test = model.predict(X_orig_test)

Y_pred_classes_test = np.argmax(Y_pred_test,axis = 1)



image = X_orig_test[0].reshape( 28, 28)

plt.imshow(image)

print(Y_pred_classes_test[0])

Y_true_test = np.argmax(y_orig_test,axis = 1)

print(y_orig_test[0])

i = 120

image = X_orig_test[i].reshape( 28, 28)

plt.imshow(image)

print(Y_pred_classes_test[i])

print(y_orig_test[i])
##Accuracy Score

from sklearn.metrics import accuracy_score

print("Accuracy on test sample is: %f" %(accuracy_score(Y_true_test, Y_pred_classes_test)))
### Printing classification report

from sklearn.metrics import classification_report

target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8', 'class 9']

print(classification_report(Y_true_test, Y_pred_classes_test, target_names=target_names))