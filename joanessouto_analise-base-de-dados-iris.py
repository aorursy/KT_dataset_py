# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing the necessary libraries

from sklearn import datasets

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
#loading the iris database

Iris = datasets.load_iris()
Iris.data
Iris.target
#dividing the data into training and testing

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(Iris.data, Iris.target, test_size=0.3)
X_train.shape, Y_train.shape
Y_train.shape, Y_test.shape
from sklearn.neighbors import KNeighborsClassifier

#specifying the k parameter (n_neighbors = 3)

knn = KNeighborsClassifier(n_neighbors=3)
#Training the algorithm

knn.fit(X_train, Y_train)
#predicting the data

result = knn.predict(X_test)

result
#validation metrics

from sklearn import metrics

print(metrics.classification_report(Y_test, result))
#confusion matrix

pd.crosstab(Y_test, result, rownames=['real value'], colnames=['predicted value'],margins=True)
#using cross validation

score = cross_val_score(knn, X_test,result,cv=10)

score
score.mean()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
#!pip install category_encoders
#using Pipeline

pip_1 = Pipeline([

    ('scaler', StandardScaler()),

    ('knn', KNeighborsClassifier())

])
#training the data

pip_1.fit(X_train, Y_train)
#predicting the data

predict = pip_1.predict(X_test)

predict
#checking the accuracy of the result

pip_1.score(X_test, Y_test)