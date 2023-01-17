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
X_test = pd.read_csv("../input/digit-recognizer/test.csv")
train = pd.read_csv("../input/digit-recognizer/train.csv")
Y_train = train["label"]
X_train = train.drop(labels=["label"],axis = 1)
del train
print(X_test.shape)
print(X_train.shape)
print(Y_train.shape)
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)
print(X_train.shape)
print(X_test.shape)
from keras.utils.np_utils import to_categorical
Y_train = to_categorical(Y_train, num_classes=10)
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dropout, Dense, MaxPool2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.20, random_state=10)
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dense(10, activation = "softmax"))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
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
history = model.fit(datagen.flow(X_train, Y_train, batch_size=84), steps_per_epoch=(X_train.shape[0]//84), epochs=3, validation_data=(X_val, Y_val))
Y_pred = model.predict(X_val)
Y_pred_final = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_val, axis = 1)
confusion_matrix(Y_true, Y_pred_final)
res = model.predict(X_test)
res
res
results = np.argmax(res,axis = 1)
results

results = pd.Series(results,name="Label")
test_output = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
test_output.to_csv("test_output.csv", index=False)
