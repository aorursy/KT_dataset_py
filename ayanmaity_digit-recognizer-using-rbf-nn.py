# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/digit-recognizer/train.csv')
df.shape
train_df = df.sample(frac=0.8,random_state=200)
test_df = df.drop(train_df.index)
train_rbf = np.load('../input/ml-assignment-3/train_rbf.npy')
test_rbf = np.load('../input/ml-assignment-3/test_rbf.npy')
train_rbf.shape,test_rbf.shape
import tensorflow as tf 
label_train = train_df['label'].values
train_df = train_df.drop(['label'],axis=1)
train_x = train_df.as_matrix()/255
label_test = test_df['label'].values
test_df = test_df.drop(['label'],axis=1)
test_x = test_df.as_matrix()/255
train_rbf = np.power(train_rbf,6)
test_rbf = np.power(test_rbf,6)
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False)
train_y = enc.fit_transform(label_train.reshape(-1,1),)
test_y = enc.transform(label_test.reshape(-1,1),)
train_y
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

model1=  Sequential()
model1.add(Dense(10,input_shape=(784,),activation='softmax'))

model1.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])
hist = model1.fit(train_rbf,train_y,epochs=100,batch_size=150,verbose=1)
hist1 = model1.fit(train_rbf,train_y,epochs=100,batch_size=150,verbose=1)
model1.evaluate(test_rbf,test_y, batch_size=150)
model1.evaluate(train_rbf,train_y, batch_size=150)
