# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # it is a wrapper for matplotlib
import matplotlib as mlp
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/BlackFriday.csv')
dataset.head()
dataset.describe()
# check out the dataset's shape
dataset.shape
# understanding dataset
dataset.dtypes
dataset.info()
# which columns have null values
dataset.isna().any()
dataset['Product_Category_2'].unique()
dataset['Product_Category_3'].unique()
# fill the NaN with 0
dataset.fillna(0, inplace=True)
dataset.isna().any()
dataset['Product_Category_2'].unique()
# make the catagories float to int
dataset['Product_Category_2'] = dataset['Product_Category_2'].astype(int)
dataset['Product_Category_3'] = dataset['Product_Category_3'].astype(int)
dataset['Product_Category_2'].unique()
# No need to use Product_ID, User_ID
dataset.drop(columns=['User_ID', 'Product_ID'], inplace=True)
dataset.head()
dataset.shape
sns.countplot(dataset['Gender'])
sns.countplot(dataset['Age'])
# count plot age data while using gender to classify it
sns.countplot(dataset['Age'], hue=dataset['Gender'])
sns.countplot(dataset['Marital_Status'], hue=dataset['Age'])
sns.countplot(dataset['Age'], hue=dataset['Marital_Status'])
sns.countplot(dataset['Marital_Status'])
from pandas.plotting import scatter_matrix
scatter_matrix(dataset)
dataset.dtypes
# build a keras model only use numeric data now
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(5,)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1))
# compile model
model.compile(optimizer='rmsprop', loss='mse',metrics=['accuracy'])
# prepare data
X = dataset[['Occupation', 'Marital_Status', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3']]
X.shape
y = dataset['Purchase']
y.shape
y[0]
X.values[0] # X is DataFrame, X[0] is wrong
type(y) # y is series, so y[0] is fine
# train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=7)
X_train.shape
X_test.shape
# split some data for validation
partial_X_train, X_val,partial_y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=7)
# train begin
history = model.fit(partial_X_train, partial_y_train, epochs=20, batch_size=512, validation_data=(X_val, y_val))
model.save('20181225-1.h5')
