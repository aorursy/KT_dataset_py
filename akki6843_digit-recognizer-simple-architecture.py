# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.datasets import fashion_mnist 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.head())
print(test.head())
X_train = (train.iloc[:,1:]).as_matrix()
print(X_train.shape)
Y_train = train.iloc[:,0].as_matrix()
print(Y_train.shape)
X_test = test.iloc[:,:].as_matrix()
print(X_test.shape)
print(X_train.shape[0])
img_rows, img_cols = 28, 28
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = to_categorical((Y_train))
print((Y_train.shape))
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
def create_model():
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=1, momentum=0.99, epsilon=0.001))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(2500, activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model
model = create_model()
model.summary()
def train():
    model = create_model()
    model.compile(optimizer=Adam(lr=0.003, beta_1=0.9, beta_2=0.999), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size= 64, epochs=50, verbose=2,shuffle=False, validation_split=0.35)
    model.save_weights('./mnist.h5')
train()
model = create_model()
model.load_weights('./mnist.h5')

preds = model.predict_classes(X_test, verbose=0)
df = pd.DataFrame(preds, columns=['Label'])
id = pd.DataFrame([range(1,len(preds)+1)]).transpose()
# id = pd.DataFrame(id, columns=['ImageId'])
pp = pd.concat([id, df],axis = 1)
sub = pd.DataFrame(pp)
sub.rename(columns={0:'ImageId'})
sub.to_csv("./Submission.csv")
print(sub.head())
print(sub.tail())
