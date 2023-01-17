# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
iris = pd.read_csv('../input/Iris.csv')
iris.head()
print(iris.Species.value_counts())
iris.info()
_ = sns.scatterplot(x= 'SepalLengthCm', y='SepalWidthCm', data = iris, hue = 'Species').set_title('Sepal Dimensions')
_ = sns.scatterplot(x= 'PetalLengthCm', y='PetalWidthCm', data = iris, hue = 'Species').set_title('Petal Dimensions')
iris.drop('Id', axis =1, inplace= True)
_= iris.hist()
ax = sns.pairplot(iris, hue ='Species')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
iris.keys()
X = iris[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris[['Species']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43, stratify=y)
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
prediction = knn.predict(X_test)
print(prediction)
knn.score(X_test, y_test)
k_value = np.arange(1,11)
train_accuracy = np.empty(len(k_value))
test_accuracy = np.empty(len(k_value))

for i,k in enumerate(k_value):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train.values.ravel())
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)
    
print('Train Accuracy: ',train_accuracy)
print('Test Accuracy: ',test_accuracy)
_ = plt.plot(k_value, test_accuracy, label = 'Test Accuracy')
_ = plt.plot(k_value, train_accuracy, label = 'Train Accuracy')
plt.xlabel('number')
plt.ylabel('accuracy')
_ = plt.legend()
X_s = iris[['SepalLengthCm','SepalWidthCm']]
y_s = iris[['Species']]
X_p = iris[['PetalLengthCm', 'PetalWidthCm']]
y_p = iris[['Species']]
X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(X_s, y_s, random_state = 43, stratify = y)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_s_train, y_s_train.values.ravel())
accuracy_s = knn.score(X_s_test, y_s_test)
print('Sepal accuracy: ',accuracy_s)
X_p_train, X_p_test, y_p_train, y_p_test = train_test_split(X_p, y_p, random_state = 43, stratify = y)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_p_train, y_p_train.values.ravel())
accuracy_p = knn.score(X_p_test, y_p_test)
print('Petal accuracy: ', accuracy_p)
