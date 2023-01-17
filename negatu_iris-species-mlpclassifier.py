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
iris_all = pd.read_csv("../input/Iris.csv")

iris_all.head()
print("length of iris_all : ", len(iris_all))
from sklearn import model_selection as skm

iris_train, iris_test = skm.train_test_split(iris_all, test_size=0.33)
print(len(iris_train), ", ", len(iris_test))
x_train = np.array(iris_train.drop(['Id', 'Species'],1))

y_train = np.array(iris_train['Species'])

x_test = np.array(iris_test.drop(['Id', 'Species'],1))

y_test = np.array(iris_test['Species'])

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(alpha=1e-07, hidden_layer_sizes=(1,))

clf.fit(x_train, y_train)

clf.score(x_test, y_test)
clf = MLPClassifier(alpha=1e-07, hidden_layer_sizes=(100,))

clf.fit(x_train, y_train)

clf.score(x_test, y_test)
clf = MLPClassifier(alpha=1e-07, hidden_layer_sizes=(2,))

clf.fit(x_train, y_train)

clf.score(x_test, y_test)
clf = MLPClassifier(alpha=1e-07, hidden_layer_sizes=(4,4))

clf.fit(x_train, y_train)

clf.score(x_test, y_test)