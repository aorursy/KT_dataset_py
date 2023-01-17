# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

import os
print(os.listdir("../input"))

sns.set(style='white', context='notebook', palette='deep')
# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.shape
train.head()
test.head()
#isolate the "label" column from train and store it in y
y_train = train['label']
#drop the label column and rest of the coluimns represents 28*28 image, store in X
X_train = train.drop(labels=["label"], axis=1)

sns.distplot(y_train)
#checking  whether the split was okay or not
X_train.head()
#check is there any null value or not
X_train.isnull().any().sum()
test.isnull().any().sum()
X_train = X_train / 255.0
test = test / 255.0
#Reshape both X_train and test data frames
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
#now lets check a single imge from X_train and test
first = plt.imshow(X_train[0][:,:,0])
y_train = to_categorical(y_train, num_classes=10)
y_train.shape
random_seed = 2
second = plt.imshow(test[0][:,:,0])
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_seed)
y_train.shape
#set the CNN model
#the model architecture Input -> [[Conv2D->relu]*2 -> MaxPool2D]*2 -> Flatten -> Dense -> Dropout -> Output
model = Sequential()
model.add(Conv2D(filters=40, kernel_size=(5,5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=40, kernel_size=(5,5), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=60, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(Conv2D(filters=60, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
#Define the optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#compile the model 
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
plt.imshow(X_train[0][:,:,0])
#learning rate reducing 
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=5, factor=0.5, verbose=1, min_lr=0.00001)
epochs = 1 #use 30 epochs
batch_size = 84
#image data generation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)
X_val.shape

y_val.shape
X_train.shape
#fit the model

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                              epochs=epochs, verbose=2, callbacks=[learning_rate_reduction],
                              validation_data=(X_val, y_val), steps_per_epoch=X_train.shape[0] // batch_size)
#predict the result
result = model.predict(test)

#select the maximum probability of each prediction
result = np.argmax(result, axis=1)

#make it series
result = pd.Series(result, name='Label')

test.shape
my_sumission = pd.concat([pd.Series(range(1, 28001), name='ImageId'), result], axis=1)
my_sumission.to_csv("submission.csv", index=False)