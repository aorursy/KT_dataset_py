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
iris = pd.read_csv('/kaggle/input/iris/Iris.csv')
iris = iris.drop('Id', axis = 1)
#Exploring the dataset for dtype and null values
iris.info()
import seaborn as sns

#Visulaising data using paiplot with regression as the 'kind'.
sns.pairplot(iris, hue = 'Species', kind = 'reg', height = 4)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

train, test = train_test_split(iris, test_size = 0.3, random_state = 5)
train_X = train[X]
train_y = train.Species
test_X = test[X]
test_y = test.Species
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state = 5)

clf.fit(train_X, train_y)
prediction_clf = clf.predict(test_X)

score_clf = round(accuracy_score(prediction_clf, test_y), 5)

print('The prediction accuray of Random Forest Classifier with train, test split is ' + str(score_clf))
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state = 5)

lr.fit(train_X, train_y)
prediction_lr = lr.predict(test_X)

score_lr = round(accuracy_score(prediction_lr, test_y), 5)

print('The prediction accuray of Logistic Regression with train, test split is ' + str(score_lr))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(train_X, train_y)
prediction_knn = knn.predict(test_X)

score_knn = round(accuracy_score(prediction_knn, test_y), 5)

print('The prediction accuray of K-Nearest Neighbours Classifier with train, test split is ' + str(score_knn))
from sklearn.svm import LinearSVC

svc = LinearSVC(random_state = 5)

svc.fit(train_X, train_y)
prediction_svc = svc.predict(test_X)

score_svc = round(accuracy_score(prediction_svc, test_y), 5)

print('The prediction accuray of Linear Support Vector Classifier with train, test split is ' + str(score_svc))
import matplotlib.pyplot as plt

#Finding correlation in the dataset
corr = iris.corr()

#Building a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype = np.bool))

#Building a plot
f, ax = plt.subplots(figsize = (11, 9))

#Generationg custom divergig colour map
cmap = sns.diverging_palette(10, 220, as_cmap = True)

sns.heatmap(corr, mask = mask, cmap = cmap, vmax = 0.3, center = 0, square = True, linewidths = 0.5, annot = True, cbar_kws = {'shrink' : 0.75})
#Taking new parameters into account
Z = ['PetalLengthCm', 'SepalWidthCm']

train_Z = train[Z]
test_Z = test[Z]
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state = 5)

clf.fit(train_Z, train_y)
prediction_clfz = clf.predict(test_Z)

score_clfz = round(accuracy_score(prediction_clfz, test_y), 5)

print('The prediction accuray of Random Forest Classifier with train, test split is ' + str(score_clfz))
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state = 5)

lr.fit(train_Z, train_y)
prediction_lrz = lr.predict(test_Z)

score_lrz = round(accuracy_score(prediction_lrz, test_y), 5)

print('The prediction accuray of Logistic Regression with train, test split is ' + str(score_lrz))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(train_Z, train_y)
prediction_knnz = knn.predict(test_Z)

score_knnz = round(accuracy_score(prediction_knnz, test_y), 5)

print('The prediction accuray of K-Nearest Neighbours Classifier with train, test split is ' + str(score_knnz))
from sklearn.svm import LinearSVC

svc = LinearSVC(random_state = 5)

svc.fit(train_Z, train_y)
prediction_svcz = svc.predict(test_Z)

score_svcz = round(accuracy_score(prediction_svcz, test_y), 5)

print('The prediction accuray of Linear Support Vector Classifier with train, test split is ' + str(score_svcz))