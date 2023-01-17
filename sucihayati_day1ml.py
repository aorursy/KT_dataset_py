# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
iris = pd.read_csv("../input/iris/Iris.csv")
iris.head(5)
fig = iris[iris.Species== 'Iris-setosa'].plot(kind='scatter',x='SepalLengthCm' ,y='SepalWidthCm' ,color='orange' ,label='Setosa')

iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm' ,y='SepalWidthCm' ,color='blue' ,label='versicolor' , ax=fig)

iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm' ,y='SepalWidthCm' ,color='black' ,label='virginica' , ax=fig)

fig.set_xlabel(" Sepal Length ")

fig.set_ylabel(" Sepal Width ")

fig.set_title(" Sepal Lenth vs Sepal Width ")

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()
iris.drop('Id' ,axis=1 ,inplace=True)
iris.head(5)
from sklearn.model_selection import train_test_split

train, test = train_test_split(iris, test_size = 0.3)

print(train.shape)

print(test.shape)
train_X = train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalLengthCm']]

train_Y = train.Species

test_X = test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalLengthCm']]

test_Y = test.Species
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics
model=KNeighborsClassifier(n_neighbors=3)

model.fit(train_X, train_Y)

prediction=model.predict(test_X)

print('The accuracy of the KNN is' ,metrics.accuracy_score(prediction, test_Y))