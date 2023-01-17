#normal import

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt



# visualization import

import seaborn as sb



# Disregard warning

import warnings 

warnings.filterwarnings('ignore')
iris = pd.read_csv('../input/iris/Iris.csv') 
iris.head()
print(iris.info())
iris.describe()
iris['Species'].value_counts()
iris_virginica = iris[iris['Species'] == 'Iris-virginica']

iris_setosa = iris[iris['Species'] == 'Iris-setosa']

iris_versicolor = iris[iris['Species'] == 'Iris-versicolor']
iris_virginica.describe()
iris_setosa.describe()
iris_versicolor.describe()
sepalLength = sb.FacetGrid(iris, col = 'Species')

sepalLength.map(plt.hist, 'SepalLengthCm', bins = 10)
sepalWidth = sb.FacetGrid(iris, col = 'Species')

sepalWidth.map(plt.hist, 'SepalWidthCm', bins = 10)
petalLength = sb.FacetGrid(iris, col = 'Species')

petalLength.map(plt.hist, 'PetalLengthCm', bins = 10)
petalWidth = sb.FacetGrid(iris, col = 'Species')

petalWidth.map(plt.hist, 'PetalWidthCm', bins = 10)
iris['Species'] = iris['Species'].map({'Iris-setosa' : 1, 'Iris-versicolor' : 2, 'Iris-virginica' : 3}).astype(int)
petal_width_range = [0.0, 1.0, 1.7, 2.5]

petal_width_level = [0, 1, 2]

iris['PetalWidthRange'] = pd.cut(iris['PetalWidthCm'], bins = petal_width_range, labels = petal_width_level)
petal_length_range = [0.0, 2, 4.85, 7]

petal_length_level = [0, 1, 2]

iris['PetalLengthRange'] = pd.cut(iris['PetalLengthCm'], bins = petal_length_range, labels = petal_length_level)
iris['SepalLength'] = pd.qcut(iris['SepalLengthCm'], 3)

iris['SepalLength'].value_counts()
sepal_length_range = [0.0, 5.4, 6.3, 7.9]

sepal_length_level = [0, 1, 2]

iris['SepalLengthRange'] = pd.cut(iris['SepalWidthCm'], bins = sepal_length_range, labels = sepal_length_level)
iris['SepalWidth'] = pd.qcut(iris['SepalWidthCm'], 3)

iris['SepalWidth'].value_counts()
sepal_width_range = [0.0, 2.9, 3.2, 4.4]

sepal_width_level = [0, 1, 2]

iris['SepalWidthRange'] = pd.cut(iris['SepalWidthCm'], bins = sepal_width_range, labels = sepal_width_level)
drop_features = ['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'SepalLength', 'SepalWidth']

iris = iris.drop(drop_features, axis = 1)
# Classification module

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
X_train = iris.drop('Species', axis = 1)

Y_train = iris['Species']
logis_classify = LogisticRegression()

logis_classify.fit(X_train, Y_train)

logis_score = logis_classify.score(X_train, Y_train)

print("The score of Logistic Regression is : " + str(logis_score))
svm_classify = SVC()

svm_classify.fit(X_train, Y_train)

svm_score = svm_classify.score(X_train, Y_train)

print("The score of SVM is : " + str(svm_score))
knn_classify = KNeighborsClassifier(n_neighbors = 2)

knn_classify.fit(X_train, Y_train)

knn_score = knn_classify.score(X_train, Y_train)

print("The score of KNN is : " + str(knn_score))
tree_classify = DecisionTreeClassifier()

tree_classify.fit(X_train, Y_train)

tree_score = tree_classify.score(X_train, Y_train)

print("The score of Decision tree is : " + str(tree_score))
forest_classify = RandomForestClassifier()

forest_classify.fit(X_train, Y_train)

forest_score = forest_classify.score(X_train, Y_train)

print("The score of Random Forest is : " + str(forest_score))