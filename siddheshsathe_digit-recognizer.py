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
df = pd.read_csv('../input/train.csv')
df['label'].value_counts()
max_num_samples = max(df['label'].value_counts())
from sklearn.model_selection import train_test_split
x = df.drop('label', axis=1).values

y = df['label'].values
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.optimizers import SGD

dropout = True

dropoutVal = 0.2

epochs = 25
def model():

    model = Sequential()

    model.add(Dense(500, input_shape=(784,), activation='relu'))

    if dropout:

        model.add(Dropout(dropoutVal))

    model.add(Dense(300, activation='relu'))

    if dropout:

        model.add(Dropout(dropoutVal))

    model.add(Dense(300, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    

    model.compile(loss='sparse_categorical_crossentropy', optimizer=SGD(0.001), metrics=['accuracy'])

    model.summary()

    return model
model = model()
history = model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=epochs)
df_test = pd.read_csv('../input/test.csv')

df_test.head()

df_test.index += 1

pred = model.predict(df_test.values)

finalsub = []

for p in pred:

    finalsub.append(np.argmax(p))
submission = pd.DataFrame({

    'ImageId': df_test.index,

    'Label': finalsub

})
submission.to_csv('submission.csv', index=False)