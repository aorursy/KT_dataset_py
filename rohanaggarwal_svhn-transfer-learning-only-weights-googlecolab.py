%tensorflow_version 2.x

import tensorflow

tensorflow.__version__
import random

random.seed(0)
import warnings

warnings.filterwarnings("ignore")
from google.colab import drive

drive.mount('/content/drive/')
project_path = '/content/drive/My Drive/'
import h5py



# Open the file as readonly

h5f = h5py.File(project_path + 'SVHN_single_grey1.h5', 'r')



# Load the training, test and validation se

X_train = h5f['X_train'][:]

y_train = h5f['y_train'][:]

X_test = h5f['X_test'][:]

y_test = h5f['y_test'][:]

# Close this file

h5f.close()
print("X_train shape:", X_train.shape)

print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)

print("y_test shape:", y_test.shape)
%matplotlib inline

import matplotlib.pyplot as plt



columns=10

rows=10



fig=plt.figure(figsize=(8, 8))



for i in range(1,columns*rows+1):

  img=X_train[i]

  fig.add_subplot(rows,columns,i)

  print(y_train[i],end='   ')

  if i % columns == 0:

    print ("")

  plt.imshow(img,cmap='gray')



plt.show()
# Importing OpenCV module for the resizing function

import cv2

import numpy as np



# Create a resized dataset for training and testing inputs with corresponding size

# Here we are resizing it to 28X28 (same input size as MNIST)

X_train_resized=np.zeros((X_train.shape[0],28,28))

for i in range(X_train.shape[0]):

  #using cv2.resize to resize each train example to 28X28 size using Cubic interpolation

  X_train_resized[i,:,:]=cv2.resize(X_train[i],dsize=(28,28),interpolation=cv2.INTER_CUBIC)



X_test_resized = np.zeros((X_test.shape[0], 28, 28))

for i in range(X_test.shape[0]):

  #using cv2.resize to resize each test example to 28X28 size using Cubic interpolation

  X_test_resized[i,:,:] = cv2.resize(X_test[i], dsize=(28, 28), interpolation=cv2.INTER_CUBIC)

  

# We don't need the original dataset anynmore so we can clear up memory consumed by original dataset

del(X_train, X_test)
X_train = X_train_resized.reshape(X_train_resized.shape[0], 28, 28, 1)

X_test = X_test_resized.reshape(X_test_resized.shape[0], 28, 28, 1)
del(X_train_resized, X_test_resized)
X_train = X_train.astype('float32')

X_test = X_test.astype('float32')



X_train /= 255

X_test /= 255
print("X_train shape:", X_train.shape)

print("X_test shape:", X_test.shape)



print("Images in X_train:", X_train.shape[0])

print("Images in X_test:", X_test.shape[0])
from tensorflow.keras.utils import to_categorical



y_train=to_categorical(y_train,num_classes=10)

y_test=to_categorical(y_test,num_classes=10)
print("Label: ", y_train[2])

plt.imshow(X_train[2].reshape(28,28), cmap='gray')
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense



# Initialize the model

model = Sequential()



# Add a Convolutional Layer with 32 filters of size 3X3 and activation function as 'relu' 

model.add(Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=(28, 28, 1)))



# Add a Convolutional Layer with 64 filters of size 3X3 and activation function as 'relu' 

model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))



# Add a MaxPooling Layer of size 2X2 

model.add(MaxPooling2D(pool_size=(2, 2)))



# Apply Dropout with 0.2 probability 

model.add(Dropout(rate=0.2))



# Flatten the layer

model.add(Flatten())



# Add Fully Connected Layer with 128 units and activation function as 'relu'

model.add(Dense(128, activation="relu"))



#Add Fully Connected Layer with 10 units and activation function as 'softmax'

model.add(Dense(10, activation="softmax"))
for l in model.layers:

  print(l.name)
for l in model.layers:

  if 'dense' not in l.name:

    l.trainable=False

  if 'dense' in l.name:

    print(l.name + ' should be trained') 
model.load_weights(project_path + 'cnn_mnist_weights-1.h5')