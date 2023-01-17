# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
iris = load_iris()
X = iris.data
y = iris.target
all_acc=[]
number=[]
#(hidden layers, corresponding nodes)
for j in range(3,15):
    for i in range(1,40):
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.30,random_state=0)
        #random_state=0 for non random results
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(i,j), random_state=0)
        clf.fit(X_train,Y_train)
        Y_pred=clf.predict(X_test)
        acc=accuracy_score(Y_test,Y_pred)
        print(acc)
        all_acc.append(acc)
        number.append(i)
        #print(confusion_matrix(Y_test, Y_pred))
        #print(classification_report(Y_test,Y_pred))
print('the max is'+str(max(all_acc)))
plt.plot(number,all_acc)
plt.show()












































