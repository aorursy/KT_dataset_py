# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
data = pd.read_csv('../input/training.csv')
y = data.iloc[:,1:2].values
x_train =  data.iloc[:,3:].values
data.info()

labelencoder_X_1 = LabelEncoder()
x_train[:, 1] = labelencoder_X_1.fit_transform(x_train[:, 1]) 
labelencoder_X_2 = LabelEncoder()
transform_columns = [0,2,3,4,5,6,7,8,10,12,13,14,27,24,23,25] 

def convert_cols_to_string(x,col_ids):
    for column_id in col_ids:
        x[:, column_id] = x_train[:,column_id].astype('str')
    return x_train

def one_hot_encode_for_columns(x,col_ids):
    x = convert_cols_to_string(x, col_ids)
    for column_id in col_ids:
        x[:, column_id] = labelencoder_X_2.fit_transform(x_train[:, column_id])
    return x_train

x = one_hot_encode_for_columns(x_train, transform_columns)
model = Sequential()
model.add(Dense(10, input_dim=31, kernel_initializer='normal', activation='tanh'))
model.add(Dense(10,kernel_initializer='normal', activation='tanh'))
model.add(Dense(10,kernel_initializer='normal', activation='sigmoid'))
model.compile(loss = keras.losses.sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
model.fit(x=x, y=y, epochs=10, validation_split=0.2)

