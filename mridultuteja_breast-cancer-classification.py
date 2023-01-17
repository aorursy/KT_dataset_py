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
def impcln():
    df = pd.read_csv('../input/data.csv')
    df = df.drop('id',axis=1)
    y = df['diagnosis']
    df = df.drop(['diagnosis','Unnamed: 32'], axis=1 )
    X = df
    return X,y
from sklearn.model_selection import train_test_split
def splitt():
    X,y = impcln()
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=4)
    return X_train,X_test,y_train,y_test
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = splitt()
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train,y_train),
knn.predict(X_test)
print(knn.score(X_test,y_test))
