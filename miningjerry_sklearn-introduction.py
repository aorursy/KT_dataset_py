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
import numpy as np

import pandas as pd

import seaborn

from sklearn.linear_model import LinearRegression

from scipy import stats

import pylab as pl

import matplotlib.pyplot as plt

%matplotlib inline



seaborn.set()

from IPython.display import Image

Image("http://scikit-learn.org/dev/_static/ml_map.png", width=800)
from sklearn.datasets import load_iris

data = load_iris()



n_samples, n_features = data.data.shape

print(data.keys())

print(n_samples, n_features)

print(data.data.shape)

print(data.target.shape)

print(data.target_names)

print(data.feature_names)
x_index =  1

y_index =2



formatter = plt.FuncFormatter(lambda i, *args: data.target_names[int(i)])

plt.scatter(data.data[:,x_index], data.data[:, y_index],

           c=data.target, cmap=plt.cm.get_cmap('RdYlBu',3))

plt.colorbar(ticks=[0,1,2], format=formatter)

plt.clim(-0.5, 2.5)

plt.xlabel(data.feature_names[x_index])

plt.ylabel(data.feature_names[y_index])
from sklearn import neighbors, datasets

data = datasets.load_iris()



X, y = data.data, data.target



clf = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform')

clf.fit(X,y)



X_test = [3,4,2,5]

y_pred = clf.predict([X_test,])

print(y_pred)

print(data.target_names[y_pred])

print(data.target_names)

print(clf.predict_proba([X_test, ]))
from fig_code import plot_iris_knn

plot_iris_knn()