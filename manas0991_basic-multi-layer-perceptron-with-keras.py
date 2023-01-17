# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import Dense, Activation

from sklearn.preprocessing import OneHotEncoder

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/digit-recognizer/train.csv', dtype="float")

X_train = df.iloc[:,1:].values/255.0

y_train = df.iloc[:,0].values[:,np.newaxis]

onehotencoder = OneHotEncoder(categories='auto')

y_train = onehotencoder.fit_transform(y_train).toarray()
model = Sequential()

model.add(Dense(units=200, activation='relu', input_dim=784))

model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_split=0.1, epochs=10, verbose=1)
df = pd.read_csv('../input/digit-recognizer/test.csv', dtype="float")

X_test = df.values/255.0

y_pred = model.predict(X_test, verbose=1)

y_pred = np.argmax(y_pred, axis = 1)

pred = pd.DataFrame({"ImageId": range(1,len(y_pred)+1), "Label": y_pred})

pred.to_csv('results.csv', index=False, header=True)