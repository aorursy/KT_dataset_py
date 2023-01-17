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
import sklearn
from sklearn import cluster
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score , r2_score

data = pd.read_csv('../input/Iris.csv')
print(data.shape)
data.loc[data['Species'] == 'Iris-setosa'] = 0
data.loc[data['Species'] == 'Iris-versicolor'] = 1
data.loc[data['Species'] == 'Iris-virginica'] = 2
train , test = train_test_split(data , test_size = 0.3)
x_train = train.iloc[: , 0:5]
y_train = train.iloc[: , 5]
x_validation = test.iloc[: , 0:5]
y_validation = test.iloc[: , 5]
print(x_train.shape)
model = cluster.KMeans(n_clusters = 3)
model.fit(x_train)
y_predict = model.predict(x_validation)
print(y_predict)
#print(r2_score(y_validation , y_predict))
print(accuracy_score(y_validation , y_predict)*100)
