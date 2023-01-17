# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from keras import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
X = df_train.drop(columns=['label'])

y = df_train['label']



X = X / 255

y = to_categorical(y, num_classes=10)
model = Sequential()

model.add(Dense(units=30, input_shape=[784,]))

model.add(Dropout(0.22))

model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)

model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))