# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pandas as pd
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
trainset = pd.read_csv('../input/fashion-mnist_train.csv')
testset = pd.read_csv('../input/fashion-mnist_test.csv')
trainset.head()
train_label = pd.get_dummies(trainset['label'])
del trainset['label']
test_label = pd.get_dummies(testset['label'])
del testset['label']
import numpy as np
trainset = trainset.values.reshape(-1,28,28,1)
testset = testset.values.reshape(-1,28,28,1)
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

input_shape=(28,28,1)
nClasses=10
def createModel():
    model = Sequential()
    model.add(Conv2D(28, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(28, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))
     
    return model
model1 = createModel()
model1.summary()
batch_size = 256
epochs = 10
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model1.fit(trainset, train_label, batch_size=batch_size, epochs=epochs, verbose=1, 
                   validation_data=(testset, test_label))
 
print(model1.evaluate(testset, test_label))
