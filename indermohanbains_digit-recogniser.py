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
sample_sub = '../input/digit-recognizer/sample_submission.csv'

test = '../input/digit-recognizer/test.csv'

train = '../input/digit-recognizer/train.csv'
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split



from keras.models import Sequential

from keras.utils.np_utils import to_categorical

from keras.callbacks import LearningRateScheduler

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AvgPool2D, BatchNormalization, Reshape
train_data = pd.read_csv (train)

test_data = pd.read_csv (test)
print (train_data.shape)

print (test_data.shape)
y_train = train_data ['label']



X_train = train_data.drop(labels = ["label"], axis = 1)



X_test = test_data
X_train = X_train/255

X_test  = X_test/255
X_train = X_train.values.reshape (-1, 28,28,1)

X_test = X_test.values.reshape (-1, 28,28,1)
y_train = to_categorical (y_train, num_classes = 10)
datagen = ImageDataGenerator(

        rotation_range = 10,  

        zoom_range = 0.1,  

        width_shift_range = 0.1, 

        height_shift_range = 0.1)
model = Sequential ()



# 2D convolutional layer

model.add(Conv2D(32,kernel_size=3, activation='relu', input_shape=(28,28,1)))

# batch normalization layer

model.add(BatchNormalization())

model.add(Conv2D(32,kernel_size=3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(32,kernel_size=5, strides=2, padding='same', activation='relu'))

model.add(BatchNormalization())

# dropout layer

model.add(Dropout(0.5))





model.add(Conv2D(64, kernel_size=3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=5, strides=2, padding='same' ,activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



# Flattens input

model.add(Flatten())

# fully connected layer

model.add(Dense(128, activation ='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 64)
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size = 64), 

    epochs = 50, steps_per_epoch = X_train.shape[0]//64, validation_data = (X_val, Y_val), verbose=1)
# initial predictions

predictions = model.predict(X_test)



# returns indices of maximum values along axis

predictions = np.argmax(predictions, axis = 1)



# convert to pandas series format

predictions = pd.Series(predictions, name = "Label")
submission = pd.concat([pd.Series(range(1, 28001), name = "ImageId"), predictions], axis = 1)



submission.to_csv("MNIST_top_CNN_submission.csv", index = False)