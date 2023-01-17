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
import keras
from keras.layers import Dense,Flatten,Dropout,Conv2D,MaxPool2D
from keras.models import Sequential
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt 
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print('Shape of training data and labels are - ',X_train.shape,y_train.shape)
print('Shape of test data and labels are - ',X_test.shape,y_test.shape)
# lets look at training data into much more detail 
print('Training Data - ')
print('Nummber of images - ',X_train.shape[0])
print('Dimensions of an image - ',X_train.shape[1:3])
print('Number of channels - ',X_train.shape[-1])
def show_channels(img):
    plt.imshow(img)
    plt.title('Original image')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    red_channel = img[:,:,0]
    green_channel = img[:,:,1]
    blue_channel = img[:,:,2]
    
    fit,ax = plt.subplots(1,3,figsize = (12,6))
    ax[0].imshow(red_channel,cmap = 'Reds')
    ax[0].set_title('Red Channel')
    ax[1].imshow(green_channel,cmap = 'Greens')
    ax[1].set_title('Green Channel')
    ax[2].imshow(blue_channel,cmap = 'Blues')
    ax[2].set_title('Blue channel')
    for i in range(3):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
idx = np.random.randint(50000)
show_channels(X_train[idx])
labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
sample = np.random.choice(np.arange(50000),10) #to get random indices
 

fig, axes = plt.subplots(2, 5, figsize=(12,4))
axes = axes.ravel()

for i in range(10):
    idx = sample[i]
    axes[i].imshow(X_train[idx])
    axes[i].set_title(labels[y_train[idx][0]])
    axes[i].axis('off')
    plt.subplots_adjust(wspace=1)


# normalizing the data
X_train_norm = X_train / 255
X_test_norm = X_test / 255 
print('Before normalizing - ',X_train[0][15][15])
print('After normalizing - ',X_train_norm[0][15][15])
#one hot encoding
num_classes = 10
y_train_oh = keras.utils.to_categorical(y_train,10)
y_test_oh = keras.utils.to_categorical(y_test,10)
print('Before one hot encoding - ',y_train[0:2])
print('After one hot encoding - ',y_train_oh[0:2])
#define the convnet
model = Sequential()

# CONV => POOL => DROPOUT
model.add(Conv2D(32, (3, 3),input_shape=X_train.shape[1:],activation = 'relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# CONV => POOL => DROPOUT
model.add(Conv2D(64, (3, 3),activation = 'relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# FLATTEN => DENSE => DROPOUT
model.add(Flatten())
model.add(Dense(512,activation = 'relu'))
model.add(Dropout(0.5))

# a softmax classifier
model.add(Dense(num_classes,activation = 'softmax'))

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_norm, y_train_oh, epochs=20, 
                    validation_data=(X_test_norm, y_test_oh))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)


datagen.fit(X_train_norm)

history = model.fit_generator(datagen.flow(X_train_norm,y_train_oh),
                                epochs=25,
                                validation_data=(X_test_norm, y_test_oh))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()
