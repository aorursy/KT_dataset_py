# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import svm

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn import svm



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import svm





iris = pd.read_csv("/kaggle/input/iris/Iris.csv")







iris_x = iris.loc[:,"SepalLengthCm":"SepalWidthCm"]

iris_y = iris.loc[:,"Species"]



for i,deger in enumerate(iris_y):

    if deger == "Iris-setosa":

        iris_y[i] = 0

    elif deger == "Iris-versicolor":    

        iris_y[i] = 1

    elif deger == "Iris-virginica":    

        iris_y[i] = 2

        

iris_y=iris_y.astype("float64")



iris_x = iris_x.to_numpy()



iris_y = iris_y.to_numpy()



def plot_class_map(clf, iris_x, iris_y, title="", **params):

    C = 1.0  # SVM regularization parameter



    clf.fit(iris_x, iris_y)



    

    





    

    x_min = iris_x[:,0].min()

    x_max = iris_x[:,0].max()

    y_min = iris_x[:,1].min()

    y_max = iris_x[:,1].max()



    h = (x_max / x_min)/100

    

    

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)



  

    plt.pcolormesh(xx, yy, Z , cmap=plt.cm.Paired)

    plt.contour(xx, yy, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])



    

    

    plt.scatter(iris_x[:,0], iris_x[:,1], c=iris_y, zorder=10, cmap=plt.cm.Paired, edgecolor='k', s=20)

   





    plt.title(title)

    plt.show()

# Linear

clf = svm.SVC(kernel='linear',C=1,gamma=1)

plot_class_map(clf, iris_x, iris_y, 'SVC with linear kernel')



# RBF

clf = svm.SVC(kernel='rbf',C=1,gamma=1)

plot_class_map(clf, iris_x, iris_y, 'SVC with rbf kernel')



# RBF 

clf = svm.SVC(kernel='poly',C=1,gamma=1, degree=3)

plot_class_map(clf, iris_x, iris_y, 'SVC with polynomial kernel (3 degrees)')


