%matplotlib inline

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.image as mpimg
img = mpimg.imread('../input/irisfiles/iris.jpeg')



plt.figure(figsize = (12,8))

plt.imshow(img)
import os



iris = pd.read_csv('../input/iris/Iris.csv')

iris.drop('Id',axis=1,inplace=True)



from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(iris, test_size=0.2, random_state=42)
train_X, train_y = train_set.drop('Species', axis=1), train_set['Species']

test_X, test_y   = test_set.drop('Species', axis=1), test_set['Species']
iris.info()
prep_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler()),

])



train_X_prep = prep_pipeline.fit_transform(train_X)
tree_clf = DecisionTreeClassifier(max_depth=2)



tree_clf.fit(train_X, train_y)
predictions = tree_clf.predict(train_X)

target = train_y
acc = accuracy_score(predictions, target)

acc
conf_mx = confusion_matrix(target, predictions)

conf_mx
from sklearn.tree import export_graphviz



export_graphviz( tree_clf,

                out_file="iris_tree_train.dot", 

                feature_names=iris.columns[:-1], 

                class_names=iris['Species'].unique(), rounded=True,

                filled=True

        )
img = mpimg.imread('../input/irisfiles/tree-train.png')



plt.figure(figsize = (8,8))

plt.imshow(img)
lin_svm = LinearSVC(C=1, loss='hinge')

lin_svm.fit(train_X_prep, train_y)
test_X_prep = prep_pipeline.fit_transform(test_X)

target = test_y 



predictions = lin_svm.predict(test_X_prep)
acc = accuracy_score(predictions, target)

acc
print(confusion_matrix(target, predictions))

target.unique()
iris.columns
plt.figure(figsize=(12,2.7))



plt.subplot(121)

plt.scatter(iris['SepalLengthCm'][iris['Species']=='Iris-setosa'], iris['SepalWidthCm'][iris['Species']=='Iris-setosa'] , color='r', label="Iris-setosa")

plt.scatter(iris['SepalLengthCm'][iris['Species']=='Iris-versicolor'], iris['SepalWidthCm'][iris['Species']=='Iris-versicolor'], color='b', label="Iris-versicolor")

plt.scatter(iris['SepalLengthCm'][iris['Species']=='Iris-virginica'], iris['SepalWidthCm'][iris['Species']=='Iris-virginica'], color='y', label="Iris-virginica")

plt.xlabel("Sepal length", fontsize=14)

plt.ylabel("Sepal width", fontsize=14)





plt.subplot(122)

plt.scatter(iris['PetalLengthCm'][iris['Species']=='Iris-setosa'], iris['PetalWidthCm'][iris['Species']=='Iris-setosa'] , color='r', label="Iris-setosa")

plt.scatter(iris['PetalLengthCm'][iris['Species']=='Iris-versicolor'], iris['PetalWidthCm'][iris['Species']=='Iris-versicolor'], color='b', label="Iris-versicolor")

plt.scatter(iris['PetalLengthCm'][iris['Species']=='Iris-virginica'], iris['PetalWidthCm'][iris['Species']=='Iris-virginica'], color='y', label="Iris-virginica")

plt.xlabel("Petal length", fontsize=14)

plt.ylabel("Petal width", fontsize=14)

plt.legend(loc="best")









plt.show()