# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Reading the data

#Note that the data is in a flat format, where one row represents a whole image, with the label.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.sample(5)
test.sample(5)
#Separating the digit label form the train data, and removing it from the original dataframe
train_y = train['label']
train_X = train.drop(columns=['label'])
del(train) #because we need to save some space
#Converting the 1-Dimensional image data into 2D data (will be required for visualization purposes)
train_X = np.array(train_X).reshape(len(train_X), 28, 28)
test = np.array(test).reshape(len(test), 28, 28)
#Let us see the shape (dimensions) of our data
print(train_X.shape, test.shape)
# We need the data to be of the shape (B, H, W, C),
# where B is the batch size, H is height, W is width,
# and C is channels. For our case, we need it to be:
# 42,000x28x28x1 and 28,000x28x28x1 respectively for train and test
train_X = np.reshape(train_X, (42000, 28, 28, 1))
test = np.reshape(test, (28000, 28, 28, 1))
print(train_X.shape, test.shape)
# The above cell tells us that we're 42,000 images, each of 28x28 pixels for train
# and 28,000 images in the same dimensions for test.
# Let's just visualize some cases
for i, x in enumerate(train_X[:12]):
    #plt.figure(figsize=(10, 10))
    plt.subplot(4,3,i+1)
    plt.imshow(train_X[i].reshape(28,28), cmap='gray', interpolation='none')
    # plt.tight_layout()
for i, x in enumerate(test[:12]):
    #plt.figure(figsize=(10, 10))
    plt.subplot(4,3,i+1)
    plt.imshow(test[i].reshape(28,28), cmap='gray', interpolation='none')
    # plt.tight_layout()
def encode_one_hot(x):
    return to_categorical(x)
def decode_one_hot(x):
    return np.argmax(x)
one_hot_y = encode_one_hot(train_y)
model = Sequential()
# input: 28x28 images with 3 channels -> (28, 28, 1) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model.fit(train_X, one_hot_y, epochs=55, validation_split=0.25, batch_size=256, shuffle=True, verbose=1, callbacks=[checkpoint])
# # Reloading from the best weights saved
# model = Sequential()
# # input: 28x28 images with 3 channels -> (28, 28, 1) tensors.
# # this applies 32 convolution filters of size 3x3 each.
# model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))

# model.load_weights("weights.best.hdf5")
# model.compile(loss='categorical_crossentropy', optimizer='adam')
#Getting the predicted labels for each image in the test set
predicted = model.predict(test)
predicted.shape
#Creating the final submission-ready dataframe with ImageId column, and the predicted results.
image_ids = range(1, len(test)+1)
predicted_decoded = list(map(decode_one_hot, predicted))
submission_df = pd.DataFrame({'ImageId': image_ids, 'Label': predicted_decoded})
#Saving the final dataframe as csv, which can be submitted now.
submission_df.to_csv(path_or_buf='submission.csv', index=False)
submission_df.head(10)
