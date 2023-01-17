# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data1 = pd.read_csv('../input/train.csv');
#Displays the first row of the data frame loaded from csv
data = data1.head(15000)
data[:10]
label = data.iloc[:,0]
pixels = data.iloc[:,1:785]
test_images, train_images, test_labels, train_labels = train_test_split(pixels, label, train_size = 0.8, random_state = 2);
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
x = clf.score(test_images,test_labels)
print(x)
test_images[test_images>0]=1
train_images[train_images>0]=1
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
x = clf.score(test_images,test_labels)
print(x)
#Create a dictionary of possible parameters
params_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
          'gamma': [0.0001, 0.001, 0.01, 0.1],
          'kernel':['linear','rbf'] }

#Create the GridSearchCV object
grid_clf = GridSearchCV(svm.SVC(), params_grid)

#Fit the data with the best possible parameters
grid_clf = grid_clf.fit(train_images, train_labels.values.ravel())

#Print the best estimator with it's parameters
print (grid_clf.best_params_)
clf = svm.SVC(C=10.0,gamma=0.01,kernel='rbf')
clf.fit(train_images, train_labels.values.ravel())
x = clf.score(test_images,test_labels)
print(x)