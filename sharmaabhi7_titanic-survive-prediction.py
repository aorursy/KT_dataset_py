import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn import svm

import matplotlib.pyplot as plt

from sklearn.cross_validation import KFold

import sys

train_path = "../input/train.csv"

test_path = "../input/test.csv"

train_data = pd.read_csv(train_path,delimiter=',')

test_data = pd.read_csv(test_path,delimiter=',')
train_data['Sex'].replace(['male','female'],['0','1'],inplace=True)

train_data['Age'].fillna((train_data['Age'].mean()),inplace=True)

data = np.array(train_data.iloc[:,4:6])

label = np.array(train_data.iloc[:,1:2])
kf = KFold(n=len(data),n_folds=5,shuffle=True)

for train_index,test_index in kf:

    X_train, X_test = data[train_index], data[test_index]

    y_train, y_test = label[train_index], label[test_index]

    clf = svm.SVC(kernel='linear',gamma=2)

    clf.fit(X_train,y_train)

    accuracy=clf.score(X_test,y_test)

    print(accuracy)
test_data['Sex'].replace(['male','female'],[ '0','1'],inplace=True)

test_data['Age'].fillna((test_data['Age'].mean()),inplace=True)

t_data = np.array(test_data.iloc[:,3:5])
clf.predict(t_data)