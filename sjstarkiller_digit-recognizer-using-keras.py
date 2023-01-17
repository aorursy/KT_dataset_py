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
sample = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
sample.head()
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
train.head()
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test.head()
x=train.drop('label',axis=1)
x=x.to_numpy().reshape((-1,28,28,1))
x=x/255.0
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from keras.layers.normalization import BatchNormalization
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', 
    input_shape=(28, 28 ,1)),
    MaxPooling2D(),
    Dropout(0.2),
    BatchNormalization(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    BatchNormalization(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    BatchNormalization(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(10)

])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


history = model.fit(x, train.label, epochs=15)

x_t=test.to_numpy().reshape((-1,28,28,1))
x_t=x_t/255.0
pred=model.predict(x_t)
pred=[i.argmax()for i in pred]

output=pd.DataFrame({'ImageId': test.index+1,
                       'Label': pred})
output.to_csv('submission.csv', index=False)
output.head()
output.shape
