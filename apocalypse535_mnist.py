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
filenameTrain = "/kaggle/input/digit-recognizer/train.csv"

dataframeTrain = pd.read_csv(filenameTrain)

# print(dataframe.head)

filenameTest = "/kaggle/input/digit-recognizer/test.csv"

dataframeTest = pd.read_csv(filenameTest)
X_train = dataframeTrain.iloc[:,1:].values.astype('float32')

X_train = X_train / 255.0

# print(len(X_train))

X_Train = X_train.reshape(len(X_train),784)



y_train = dataframeTrain.iloc[:,0].values

# y_Train = y_train.reshape(len(y_train), 1)



X_test = dataframeTest.iloc[:,:].values.astype('float32')

X_test = X_test / 255.0

# print(len(X_test[0]))

X_Test = X_test.reshape(len(X_test), 784)



# y_test = dataframeTest.iloc[:,0].values

# y_Test = y_test.reshape(len(y_test), 1)



# X_train = X_train.reshape(len(X_train),784)

# print(y_train.shape)
from keras.utils import to_categorical

train_y = to_categorical(y_train)

# test_y = to_categorical(y_Test, num_classes=10)

# print(y_train[0])

# print(y_train.shape)

# y_train = y_train.reshape(len(y_train), 1)

# print(y_train.shape)

from keras.models import Sequential

from keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(512, input_shape=(784,), activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model.fit(X_train, train_y,batch_size=128, epochs=10,verbose=2)
predictions = model.predict_classes(X_Test, verbose=2)

# print(type(predictions))

# print(predictions)

data = {'ImageId': list(range(1, len(X_Test) + 1)), 'Label': predictions}

result = pd.DataFrame(data)

# result["ImageId"] = list(range(1, len(X_Test) + 1))

# result["Label"] = predictions

result.to_csv("submission.csv", index=False)

print("Amount of test points:", len(X_Test))

print("Amount of predictions:", len(predictions))



result.head()