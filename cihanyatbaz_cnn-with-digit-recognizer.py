# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# read train
train = pd.read_csv("../input/train.csv")
print(train.shape)
train.head()
# read test
test = pd.read_csv("../input/test.csv")
print(test.shape)
test.head()
# put labels into y_train variable 
Y_train = train["label"]

# Drop label column
X_train = train.drop(labels=["label"], axis=1)
# visualize number of digits classes
plt.figure(figsize =(15,10))
g = sns.countplot(Y_train, palette="icefire")
plt.title("Number of digit classes")
Y_train.value_counts()
# plot for the three number
img = X_train.iloc[9].as_matrix() # as_matrix: Converting to Matrix
img = img.reshape((28,28))
plt.imshow(img, cmap='gray')
plt.title(train.iloc[0,0])
plt.axis("off")
plt.show()
# plot for the seven number
img = X_train.iloc[6].as_matrix()
img = img.reshape((28,28))
plt.imshow(img, cmap='gray')
plt.title(train.iloc[3,0])
plt.axis("off")
plt.show()
X_train = X_train / 255.0
test = test / 255.0
print("X_train shape: ", X_train.shape)
print("test shape: ", test.shape)
# reshape
X_train = X_train.values.reshape(-1,28,28,1)
terst = test.values.reshape(-1,28,28,1)
print("x_train shape: ", X_train.shape)
print("test shape: ", test.shape)
# Label encoding 
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding**
Y_train = to_categorical(Y_train, num_classes = 10)
# split the train and the validation set for the fittig
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=2)
print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("Y_train shape: ", Y_train.shape)
print("Y_test shape: ", Y_test.shape)
# examples for the eight
plt.imshow(X_train[0][:,:,0], cmap='gray')
plt.show()
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding
from keras.models import Sequential # Sequential: A structure with layers in it.
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()
#
model.add(Conv2D(filters=8, kernel_size=(5,5), padding='Same', activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2))) # Max pooling: Transfer the max values ​​in our image to the pooling layer.
model.add(Dropout(0.25)) 
#
model.add(Conv2D(filters=16, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2))) 
model.add(Dropout(0.25))
# fully connected
model.add(Flatten())# flatting : Doing a straight vector by extending our matrix
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))  # softmax: It is a more generalized version of Sigmoid.
# define the optimizer 
optimizer = Adam(lr=0.003, beta_1=0.9, beta_2=0.999)
# loss: If the error is too many we update the weight, until the error is minimized
# categorical_crossentropy : If the classification is more than 2, we use categorical_crossentropy.
model.compile(optimizer = optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
epochs = 100 # for better result increase the epochs
batch_size = 500
# Data augmentation
datagen = ImageDataGenerator(
    featurewise_center = False, # set input mean to 0 over the dataset
    samplewise_center = False,  # set each sample mean to 0 
    featurewise_std_normalization = False,  # divide inputs by std of the dataset
    samplewise_std_normalization = False,   # divide each input by its std
    zca_whitening = False,  # dimension reduction
    rotation_range = 0.5,  # Randomly rotate images in the range 5 degrees
    zoom_range = 0.5,   # Randomly zoom image 5%
    width_shift_range = 0.5,  # Randomly shift images horizontally 5%
    height_shift_range = 0.5,  # Randomly shift images vertically 5%
    horizontal_flip = False, # Randomly flip images
    vertical_flip = False) # Randomly flip images

datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size),
                             epochs = epochs, validation_data = (X_test, Y_test), steps_per_epoch = X_train.shape[0] // batch_size)
# Plot the loss and accuracy curves for training and validation
plt.plot(history.history['val_loss'], color='r', label= "validation loss")
plt.title("Test Loss")
plt.xlabel("Number of epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
# Plot the loss and accuracy curves for training and validation
plt.plot(history.history['val_acc'], color='b', label= "validation accuracy")
plt.title("Test Accuracy")
plt.xlabel("Number of epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
#Confusion Matrix
# Predict the values from the validation dataset
y_pred = model.predict(X_test)
# Convert predictions classes to one hot vectors
y_pred_classes = np.argmax(y_pred, axis=1)
# Convert validation observation to one hot vectors
y_true = np.argmax(Y_test, axis=1)
# Compute the confussion matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes)
# plot the confusion matrix
f,ax = plt.subplots(figsize=(16,8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="Blues", linecolor="Green", fmt='.1f', ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()