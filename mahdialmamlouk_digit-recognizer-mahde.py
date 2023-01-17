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
import pandas as pd

import numpy as np



import seaborn as sns



import matplotlib.pyplot as plt

%matplotlib inline



import tensorflow as tf



from tensorflow.keras import layers

from tensorflow.keras.models import Model

from tensorflow.keras import metrics

from tensorflow.keras import backend as K
# Reading the Train and Test Datasets.

import os

print(os.getcwd())

d_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

d_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
d_train.head()
d_test.head()
d_train.isnull().sum()
d_train_data = d_train.loc[:, "pixel0":]

d_train_label = d_train.loc[:, "label"]
d_train_data = d_train_data/255.0

d_test = d_test/255.0
d_train_data = np.array(d_train_data)

d_train_label = np.array(d_train_label)
d_train_data = d_train_data.reshape(d_train_data.shape[0], 28, 28, 1)

print(d_train_data.shape, d_train_label.shape)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Lambda, Flatten, BatchNormalization

from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D

from tensorflow.keras.optimizers import Adadelta

from keras.utils.np_utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.callbacks import LearningRateScheduler
d_train_label = to_categorical(d_train_label)

print(d_train_label.shape)
d_test_arr = np.array(d_test)

d_test_arr = d_test_arr.reshape(d_test_arr.shape[0], 28, 28, 1)

print(d_test_arr.shape)
model = Sequential()

#First Hidden layer

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

#Dropout to avoid overfitting

model.add(Dropout(0.25))

#Second Hidden layer

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

#Dropout to avoid overfitting

model.add(Dropout(0.25))

#Flatten output of conv

model.add(Flatten())

#Fully Connected layer

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

#Output layer

model.add(Dense(10, activation = "softmax"))

model.compile(optimizer = "adam", loss = tf.losses.categorical_crossentropy, metrics = ['accuracy'])

hist = model.fit(d_train_data, d_train_label, epochs = 60)
predictions = model.predict(d_test_arr)
predictions_label = []



for i in predictions:

    predictions_label.append(np.argmax(i))
submission =  pd.DataFrame({

        "ImageId": d_test.index+1,

        "Prediction": predictions_label

    })



submission.to_csv('my_submission.csv', index=False)
submission.head()