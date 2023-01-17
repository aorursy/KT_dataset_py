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
train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
y = train_df['label']
X = train_df.drop('label',axis=1)
X = X.values.reshape(-1,28,28,1)
from matplotlib import pyplot as plt
y[45]
X = X / 255
y[23]
X.shape
from keras.utils import to_categorical
y_cat = to_categorical(y)
y_cat[0].shape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(2,2),input_shape=(28,28,1), activation='relu',))
model.add(MaxPooling2D(pool_size=(1, 1)))

model.add(Conv2D(filters=32, kernel_size=(2,2),input_shape=(28,28,1), activation='relu',))
model.add(MaxPooling2D(pool_size=(1, 1)))

model.add(Conv2D(filters=64, kernel_size=(2,2),input_shape=(28,28,1), activation='relu',))
model.add(MaxPooling2D(pool_size=(1, 1)))


model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
model.fit(X,y_cat,epochs=5)
test_df = test_df/255
test_df = test_df.values.reshape(-1,28,28,1)
test_pred = model.predict(test_df)
predictions = np.argmax(test_pred,axis = 1)
predictions = pd.Series(predictions,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predictions],axis = 1)

submission.to_csv("submission_final.csv",index=False)

