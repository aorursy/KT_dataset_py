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
X = np.load("/kaggle/input/sign-language-digits-dataset/Sign-language-digits-dataset/X.npy")
Y = np.load("/kaggle/input/sign-language-digits-dataset/Sign-language-digits-dataset/Y.npy")
print("X.shape: ",X.shape)
print("Y.shape: ",Y.shape)
X = X.reshape(-1,64,64,1)

import matplotlib.pyplot as plt
img_size = 64
plt.subplot(1, 2, 1)
plt.imshow(X[276].reshape(img_size, img_size))
plt.axis('off')
plt.subplot(1, 2, 2)

plt.imshow(X[900].reshape(img_size, img_size))
plt.axis('off')

print(" Max value of X: ",X.max())
print(" Min value of X: ",X.min())
print(" Shape of X: ",X.shape)

print("\n Max value of Y: ",Y.max())
print(" Min value of Y: ",Y.min())
print(" Shape of Y: ",Y.shape)
# Gördüğümüz gibi veri zaten one hot coding işleminden geçmiş.
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state = 1)

print('Shape of x_train: ', x_train.shape)
print('Shape of y_train: ', y_train.shape)
print('....')
print('Shape of x_test: ', x_test.shape)
print('Shape of y_test: ', y_test.shape)
# import libary
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
#from keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(
            rotation_range = 5,        # 5 degrees of rotation will be applied
            zoom_range = 0.1,          # 10% of zoom will be applied
            width_shift_range = 0.1,   # 10% of shifting will be applied
            height_shift_range = 0.1)  # 10% of shifting will be applied

train_gen.fit(x_train)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

# Creating model structure
model = Sequential()
# Adding the first layer of CNN
model.add(Conv2D(filters=20, kernel_size=(4,4), padding='Same', activation='relu', input_shape=(64, 64, 1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.15))
# Adding the second layer of CNN
model.add(Conv2D(filters=30, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.15))
# Flattening the x_train data
model.add(Flatten()) 
# Creating fully connected NN with 4 hidden layers
model.add(Dense(220, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(10, activation='softmax'))
optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.99)
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])

batch_size = 100
epochs = 25
history = model.fit_generator(train_gen.flow(x_train, y_train, batch_size = batch_size), 
                                                  epochs = epochs, 
                                                  validation_data = (x_test, y_test), 
                                                  steps_per_epoch = x_train.shape[0] // batch_size)
# Visiualize the validation loss and validation accuracy progress:

plt.figure(figsize=(13,5))
plt.subplot(1,2,1)
plt.plot(history.history['val_loss'], color = 'r', label = 'validation loss')
plt.title('Validation Loss Function Progress')
plt.xlabel('Number Of Epochs')
plt.ylabel('Loss Function Value')

plt.subplot(1,2,2)
plt.plot(history.history['val_accuracy'], color = 'g', label = 'validation accuracy')
plt.title('Validation Accuracy Progress')
plt.xlabel('Number Of Epochs')
plt.ylabel('Accuracy Value')
plt.show()
# Confusion Matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns

# First of all predict labels from x_test data set and trained model
y_pred = model.predict(x_test)

# Convert prediction classes to one hot vectors
y_pred_classes = np.argmax(y_pred, axis = 1)

# Convert validation observations to one hot vectors
y_true_classes = np.argmax(y_test, axis = 1)

# Create the confusion matrix
confmx = confusion_matrix(y_true_classes, y_pred_classes)
f, ax = plt.subplots(figsize = (8,8))
sns.heatmap(confmx, annot=True, fmt='.1f', ax = ax)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show();