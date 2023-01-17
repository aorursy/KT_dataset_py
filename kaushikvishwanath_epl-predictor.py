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
#Team names formatted to integer type
data = pd.read_csv("../input/results.csv")
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
data.drop('season', axis = 1, inplace = True)
train, test = train_test_split(data, test_size = 0.2)
train_X = train[['home_team', 'away_team', 'home_goals', 'away_goals']]
train_Y = train.result
test_X = test[['home_team', 'away_team', 'home_goals', 'away_goals']]
test_Y = test.result
def svm_function():
    model_svm = svm.SVC(gamma = 'auto')
    model_svm.fit(train_X, train_Y)
    prediction = model_svm.predict(test_X)
    print('Accuracy of SVM: ', metrics.accuracy_score(prediction, test_Y))
    

def knn():
    model_knn = KNeighborsClassifier(n_neighbors = 3)
    model_knn.fit(train_X, train_Y)
    prediction2 = model_knn.predict(test_X)
    print('Accuracy of KNN: ', metrics.accuracy_score(prediction2, test_Y))
    

def nb():
    model_nb = GaussianNB()
    model_nb.fit(train_X, train_Y)
    prediction3 = model_nb.predict(test_X)
    print('Accuracy of Naive Bayes: ', metrics.accuracy_score(prediction3, test_Y))
    

def dt():
    model_dt = tree.DecisionTreeClassifier()
    model_dt.fit(train_X, train_Y)
    prediction4 = model_dt.predict(test_X)
    print('Accuracy of Decision Tree: ', metrics.accuracy_score(prediction4, test_Y))
    
svm_function()
knn()
nb()
dt()
