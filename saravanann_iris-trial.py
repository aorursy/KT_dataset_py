# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Iris.csv")
data = data.replace('Iris-setosa', 0)
data = data.replace('Iris-versicolor', 1)
data = data.replace('Iris-virginica', 2)
data = data.as_matrix()
features = data[:,1:-1]
labels = data[:, -1]
from matplotlib import pyplot as plt
for index, i in enumerate(features):
    if labels[index] == 0:
        plt.scatter(i[2], i[3], color='r')
    elif labels[index] == 1:
        plt.scatter(i[2], i[3], color='b')
    else:
        plt.scatter(i[2], i[3], color='g')
plt.show()
features = features[:,2:]

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, 
                                                                            test_size=0.3)
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import f1_score, recall_score, precision_score
print("F1 Score: ", f1_score(labels_test, pred))
print("Recall Score: ", recall_score(labels_test, pred))
print("Precision Score: ", precision_score(labels_test, pred))
