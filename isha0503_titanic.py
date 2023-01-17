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
dataset_train = pd.read_csv('../input/train.csv')
dataset_train = dataset_train.dropna(how='any',axis=0) 
X_train = dataset_train[['Pclass','Sex','Age', 'SibSp', 'Parch', 'Embarked']].values
y_train = dataset_train.iloc[:, 1].values

dataset_test = pd.read_csv('../input/test.csv')
dataset_test = dataset_test.dropna(how='any',axis=0) 
X_test = dataset_test[['Pclass','Sex','Age', 'SibSp', 'Parch', 'Embarked']].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X_train[:, 1] = labelencoder_X_1.fit_transform(X_train[:, 1])
X_test[:, 1] = labelencoder_X_1.transform(X_test[:, 1])

labelencoder_X_5 = LabelEncoder()
X_train[:, 5] = labelencoder_X_5.fit_transform(X_train[:, 5])
X_test[:, 5] = labelencoder_X_5.transform(X_test[:, 5])
onehotencoder = OneHotEncoder(categorical_features = [5])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_test = onehotencoder.transform(X_test).toarray()
X_train = X_train[:, 1:]
X_test = X_test[:, 1:]

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
predictions = logisticRegr.predict(X_test)