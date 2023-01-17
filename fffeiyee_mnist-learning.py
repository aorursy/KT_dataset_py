# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def prepare_data():
    data = pd.read_csv("../input/train.csv")
    data = np.array(data)
    X = data[:,1:] / 255 # division by 255 is to normalize pixel intensities to between 0 and 1
    y = data[:,0]
    # transform y to be one-hot
    n = len(y)
    y_one_hot = np.zeros((n,10), dtype=int)
    for i in range(n):
        y_one_hot[i, int(y[i])] = 1
    # split labeled data into train and validation datasets, ratio 2/3 and 1/3
    split = n//3*2
    X_train = X[:split,:]
    y_train = y_one_hot[:split,:]
    X_validate = X[split:,:]
    y_validate = y_one_hot[split:,:]
    train = [X_train,y_train]
    validate = [X_validate,y_validate]
    test = pd.read_csv("../input/test.csv")
    test = np.array(test) / 255
    return train, validate, test
train,validate, test = prepare_data()
train[0] = np.reshape(train[0],[-1,28,28,1])
validate[0] = np.reshape(validate[0],[-1,28,28,1])
test = np.reshape(test,[-1,28,28,1])

print(train[0].shape, train[1].shape, validate[0].shape, validate[1].shape, test.shape)
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(1024, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
model.compile(optimizer = Adam() , loss = "categorical_crossentropy", metrics=["accuracy"])
epochs = 3 # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 128
history = model.fit(train[0], train[1], batch_size = batch_size, epochs = epochs, 
          validation_data = (validate[0], validate[1]), verbose = 2)
history = model.fit(train[0], train[1], batch_size = batch_size, epochs = epochs, 
          validation_data = (validate[0], validate[1]), verbose = 2)
history = model.fit(train[0], train[1], batch_size = 256, epochs = epochs, 
          validation_data = (validate[0], validate[1]), verbose = 1)
history = model.fit(train[0], train[1], batch_size = 300, epochs = epochs, 
          validation_data = (validate[0], validate[1]), verbose = 1)
predictions = model.predict_classes(test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("SUB.csv", index=False, header=True)
submissions[:100]
