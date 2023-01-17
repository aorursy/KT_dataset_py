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
# load train and test set into pandas dataframe
# check null 
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test.isnull().any().describe()
train.isnull().any().describe()
# visulize class distributions to do stratifiedKfold or not
# no need to do that
import matplotlib.pyplot as plt
%matplotlib inline
plt.hist(train['label'], bins=20)
train['label'].value_counts()
# random shuffle train set and split into train and dev set
# enable onehot encoder
# reshape X to be fitted into CNN
from sklearn.model_selection import train_test_split

np.random.seed(1)
idx = np.random.permutation(len(train))
train_shuffle = train.iloc[idx]
train_X = train_shuffle.drop('label', axis = 1)
train_Y = train_shuffle['label']
train_X = train_X.values.reshape(-1,28,28,1)

train_Y = pd.get_dummies(train_Y)
train_X = train_X/255.
X_train, X_dev, Y_train, Y_dev = train_test_split(train_X, train_Y, test_size = 0.1, random_state=False)
import tensorflow as tf
from keras.layers import Input, Flatten, Dense, Activation, ZeroPadding2D, BatchNormalization, Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D, Dropout
from keras.models import Model

def model():
    X_input = Input((28,28,1))
    X = BatchNormalization(axis = -1)(X_input)
    X = Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu')(X)
    X = MaxPooling2D((2,2))(X)
    X = Dropout(0.25)(X)
    
    X = BatchNormalization(axis=-1)(X)
    X = Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu')(X)
    X = MaxPooling2D((2,2))(X)
    X = Dropout(0.25)(X)
    
    X = Flatten()(X)
    X = Dense(256, activation = 'relu')(X)
    X = Dropout(0.25)(X)
    X = Dense(10, activation = 'softmax')(X)
    
    model = Model(inputs = X_input, outputs = X)
    return model
MINST_model = model()
MINST_model.compile(optimizer = 'Adam', loss = "categorical_crossentropy", metrics=['accuracy'])
MINST_model.fit(x=X_train, y=Y_train, epochs= 10, batch_size = 86, validation_data = (X_dev, Y_dev), verbose = 1)
X_test = test.values.reshape(-1,28,28,1)
X_test = X_test/255.
y_test = MINST_model.predict(X_test)
y_test = np.argmax(y_test, axis = 1)
y_test_pred = pd.DataFrame(y_test, index=test.index)
y_test_pred.rename(columns={0: 'Label'}, inplace=True)
y_test_pred.index.names=['ImageID']
y_test_pred.head()
y_dev_pred = MINST_model.predict(X_dev)
y_dev_pred = np.argmax(y_dev_pred, axis=1)
y_dev_true = np.argmax(Y_dev.values, axis=1)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_dev_true, y_dev_pred)

