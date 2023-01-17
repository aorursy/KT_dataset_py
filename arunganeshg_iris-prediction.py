# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from IPython.display import HTML

HTML('<iframe src=http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data width=300 height=200></iframe>')
from sklearn.datasets import load_iris
iris = load_iris()

type(iris)
print(iris.data)
print(iris.feature_names)
print(iris.target_names)
print(iris.target)
print(type(iris.data))
x=iris.data

y=iris.target
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1)

print(knn)
knn.fit(x,y)
prediction = knn.predict([[2,4,3,1]])
print(prediction)
prediction = knn.predict([[3,4,5,6],[2,3,3,1]])
print(prediction)
from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression()

logistic.fit(x,y)
logisticreg = logistic.predict([[1,2,3,4]])

print(logisticreg)
logisticreg = logistic.predict([[3,4,5,6],[2,3,3,1]])

print(logisticreg)
logisticreg = logistic.predict([[5.1, 3.5, 1.4, 0.2],[5.9,3.0,5.1,1.8],[4.9,2.4,3.3,1.0]])

print(logisticreg)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.90, random_state=42)
y_predict = logistic.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_predict, y_test))
y_predict_2 = knn.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_predict_2, y_test))