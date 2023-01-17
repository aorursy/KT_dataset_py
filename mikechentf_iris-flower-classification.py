# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', None)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
iris = pd.read_csv('/kaggle/input/iris/Iris.csv')
iris.head()
iris.info()
iris.describe()
iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].corr()
plt.figure(figsize=(10,8))

sns.heatmap(iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].corr(), vmin=-1.0, vmax=1.0, annot=True, linewidths=2)

plt.show()
iris.groupby('Species').describe()
plt.figure(figsize=(8,8))

ax = sns.boxplot(x="Species", y="SepalLengthCm", data=iris).set_title('Sepal Length')

plt.show()
plt.figure(figsize=(8,8))

ax = sns.boxplot(x="Species", y="SepalWidthCm", data=iris).set_title('Sepal Width')

plt.show()
plt.figure(figsize=(8,8))

ax = sns.boxplot(x="Species", y="PetalLengthCm", data=iris).set_title('Petal Length')

plt.show()
plt.figure(figsize=(8,8))

ax = sns.boxplot(x="Species", y="PetalWidthCm", data=iris).set_title('Petal Width')

plt.show()
# We take 80% of data into training, and 20% into test

# For each set, a third belonds to each type of Iris

iris.drop(['Id'], axis=1, inplace=True)

training = pd.concat([iris[:40], iris[50:90], iris[100:140]])

test = pd.concat([iris[40:50], iris[90:100], iris[140:]])

training_X = training[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

training_y = training['Species']

test_X  = test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

test_y = test['Species']
print('Training set:', training_X.shape)

print('Test set:', test_X.shape)
from sklearn.linear_model import LogisticRegression

LR_classifier = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200).fit(training_X, training_y)

print('Training accuracy:', LR_classifier.score(training_X, training_y))

print('Test accuracy:', LR_classifier.score(test_X, test_y))
from sklearn.tree import DecisionTreeClassifier

dTree_classifier = DecisionTreeClassifier(criterion="entropy").fit(training_X, training_y)

print('Training accuracy:', dTree_classifier.score(training_X, training_y))

print('Test accuracy:', dTree_classifier.score(test_X, test_y))
from sklearn.neighbors import KNeighborsClassifier

KNN_classifier = KNeighborsClassifier().fit(training_X, training_y)

print('Training accuracy:', KNN_classifier.score(training_X, training_y))

print('Test accuracy:', KNN_classifier.score(test_X, test_y))
from sklearn.svm import LinearSVC

SVC_classifier = LinearSVC(multi_class='crammer_singer', max_iter=3000).fit(training_X, training_y)

SVC_classifier.score(training_X, training_y)

print('Training accuracy:', SVC_classifier.score(training_X, training_y))

print('Test accuracy:', SVC_classifier.score(test_X, test_y))
from sklearn.tree import plot_tree

plt.figure(figsize=(10,10))

plot_tree(dTree_classifier)

plt.show()