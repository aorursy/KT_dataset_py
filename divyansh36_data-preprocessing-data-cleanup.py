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
# importing matplotlib labrary
import matplotlib.pyplot
%matplotlib inline
# creating dataset from csv file
dataset = pd.read_csv('../input/CleanupData.csv')
dataset
# creating dependant and indepdant variable(X and y)
X = dataset.iloc[: , :-1 ].values
X
y = dataset.iloc[: , 3].values
y
# importing imputer from sklearn library
# using imputer to handle missing vaule
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy= 'mean' ,axis=0)
imputer.fit(X[: , 1:3])
X[: ,1:3] = imputer.transform(X[:, 1:3])
X
# handling categorical data
from sklearn.preprocessing import LabelEncoder
labelX = LabelEncoder()
X[: ,0 ] = labelX.fit_transform(X[: ,0 ])
X
# Creating dummy matrix
from sklearn.preprocessing import OneHotEncoder
Hotencoder = OneHotEncoder(categorical_features=[0])
X = Hotencoder.fit_transform(X).toarray()
np.set_printoptions(suppress=True)
X
labelY = LabelEncoder()
y = labelY.fit_transform(y)
y
# Preapairing train & test datasheet
from sklearn.model_selection import train_test_split
X_train, X_test, y_train , y_test = train_test_split(X,y, train_size= .8)
y_test
X_train
# Scaling 
from sklearn.preprocessing import StandardScaler
Scale_X = StandardScaler()
X_train = Scale_X.fit_transform(X_train)
X_test = Scale_X.transform(X_test)
X_test
