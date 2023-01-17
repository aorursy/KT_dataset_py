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
from sklearn.datasets import load_iris
iris_datasets = load_iris()
print("keys of iris_datasets: \n {}".format(iris_datasets.keys()))
print("Shape of data: {}".format(iris_datasets["data"].shape))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_datasets["data"], iris_datasets['target'], random_state = 0)
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))
# X_trainのデータからDataFrameを作る、
# iris_dataset.feature_namesの文字列を使ってカラムに名前を付ける
iris_dataframe = pd.DataFrame(X_train, columns = iris_datasets.feature_names)

#データフレームからscatter matrixを作成し、y_trainにしたがって色を付ける
grr = pd.scatter_matrix(iris_dataframe, c = y_train, figsize = (15, 15), marker = 'o', hist_kwds = {"bins" : 20}, s = 60, alpha = .8)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_datasets["target_names"][prediction]))
y_pred = knn.predict(X_test)
print("Test set predictions: \n {}".format(y_pred))
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
X_train, X_test, y_train, y_test = train_test_split(iris_datasets["data"], iris_datasets["target"], random_state = 0)
knn = KNeighborsClassifier(n_neighbors = 1)
# fit メソッドは訓練データを含むNumPy配列 X_trainとそれに対応する
knn.fit(X_train, y_train)
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
