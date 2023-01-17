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
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Lambda, Dropout

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator as IDG
# Reading the train and test Data

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

sample = pd.read_csv('../input/sample_submission.csv')
print(train_data.columns)

print(sample.columns)
def prep_data(raw_csv):

    label = tf.keras.utils.to_categorical(raw_csv['label'],10)

    train = raw_csv.loc[:,raw_csv.columns != 'label']

    train = train.values.reshape(raw_csv.shape[0], 28,28,1)

    train = train/255.0

    return train, label



X_train, y_train = prep_data(train_data)

X_test = (test_data.values.reshape(test_data.shape[0], 28,28,1))/255.0
model = Sequential()

model.add(Conv2D(30, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(28, 28, 1)))

model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))

model.add(Flatten())

model.add(Dense(120, activation='relu'))

model.add(Dense(10, activation='softmax'))



model.compile(loss=tf.keras.losses.categorical_crossentropy,

              optimizer=tf.train.RMSPropOptimizer(0.01),

              metrics=['accuracy'])

call_hist = model.fit(X_train, y_train,

          batch_size=128,

          epochs=6,

          validation_split = 0.25)
predictions = model.predict_classes(X_test)

imageid = list(range(1,len(predictions)+1))

print(len(imageid))
submission_df = pd.DataFrame({'ImageId':imageid, 'Label':predictions})

submission_df.to_csv("submission.csv", index=False, header=True)
op = pd.read_csv("submission.csv")
print(op)