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
from sklearn.linear_model import LogisticRegression

from sklearn import datasets

import numpy as np



%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt



iris = datasets.load_iris()



print(list(iris.keys()))



#print(iris)



X = iris["data"][:,3:]  # petal width

y = (iris["target"]==2).astype(np.int)



log_reg = LogisticRegression(penalty="l2")

log_reg.fit(X,y)



X_new = np.linspace(0,3,1000).reshape(-1,1)

y_proba = log_reg.predict_proba(X_new)



plt.plot(X,y,"b.")

plt.plot(X_new,y_proba[:,1],"g-",label="Iris-Virginica")

plt.plot(X_new,y_proba[:,0],"b--",label="Not Iris-Virginca")

plt.xlabel("Petal width", fontsize=14)

plt.ylabel("Probability", fontsize=14)

plt.legend(loc="upper left", fontsize=14)

plt.show()



log_reg.predict([[1.7],[1.5]])
iris = datasets.load_iris()



X = iris["data"][:,(2,3)]  # petal length, petal width

y = iris["target"]



softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=5)

softmax_reg.fit(X,y)



X_new = np.linspace(0,3,1000).reshape(-1,2)



plt.plot(X[:, 0][y==1], X[:, 1][y==1], "y.", label="Iris-Versicolor")

plt.plot(X[:, 0][y==0], X[:, 1][y==0], "b.", label="Iris-Setosa")



plt.legend(loc="upper left", fontsize=14)



plt.show()