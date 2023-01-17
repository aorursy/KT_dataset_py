# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.utils import to_categorical

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
%matplotlib inline
input_path = Path("../input")

# Path to training images and corresponding labels provided as numpy arrays
kmnist_train_images_path = input_path/"kmnist-train-imgs.npz"
kmnist_train_labels_path = input_path/"kmnist-train-labels.npz"

# Path to the test images and corresponding labels
kmnist_test_images_path = input_path/"kmnist-test-imgs.npz"
kmnist_test_labels_path = input_path/"kmnist-test-labels.npz"
# Load the training data from the corresponding npz files
kmnist_train_images = np.load(kmnist_train_images_path)['arr_0']
kmnist_train_labels = np.load(kmnist_train_labels_path)['arr_0']

# Load the test data from the corresponding npz files
kmnist_test_images = np.load(kmnist_test_images_path)['arr_0']
kmnist_test_labels = np.load(kmnist_test_labels_path)['arr_0']

print(f"Number of training samples: {len(kmnist_train_images)} where each sample is of size: {kmnist_train_images.shape[1:]}")
print(f"Number of test samples: {len(kmnist_test_images)} where each sample is of size: {kmnist_test_images.shape[1:]}")
# Let's see how the images for different labels look like
random_samples = []
for i in range(10):
    samples = kmnist_train_images[np.where(kmnist_train_labels==i)][:3]
    random_samples.append(samples)

# Converting list into a numpy array
random_samples = np.array(random_samples)

# Visualize the samples
f, ax = plt.subplots(10,3, figsize=(10,20))
for i, j in enumerate(random_samples):
    ax[i, 0].imshow(random_samples[i][0,:,:], cmap='gray')
    ax[i, 1].imshow(random_samples[i][1,:,:], cmap='gray')
    ax[i, 2].imshow(random_samples[i][2,:,:], cmap='gray')
    
    ax[i,0].set_title(str(i))
    ax[i,0].axis('off')
    ax[i,0].set_aspect('equal')
    
    ax[i,1].set_title(str(i))
    ax[i,1].axis('off')
    ax[i,1].set_aspect('equal')
    
    ax[i,2].set_title(str(i))
    ax[i,2].axis('off')
    ax[i,2].set_aspect('equal')
plt.show()
batch_size = 128
num_classes = 10
epochs = 15

# input image dimensions
img_rows, img_cols = 28, 28

# input shape
input_shape = (img_rows, img_cols, 1)
# Process the train and test data in the exact same manner as done for MNIST
x_train = kmnist_train_images.astype('float32')
x_test = kmnist_test_images.astype('float32')
x_train /= 255
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], *input_shape)
x_test = x_test.reshape(x_test.shape[0], *input_shape)

# convert class vectors to binary class matrices
y_train = to_categorical(kmnist_train_labels, num_classes)
y_test = to_categorical(kmnist_test_labels, num_classes)
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

H = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
# Check the test loss and test accuracy 
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1]*100)
# Save the model
model.save('model.h5')
# Plotting the result

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
