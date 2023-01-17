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

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.datasets import load_iris

iris=load_iris()

for keys in iris.keys() :

    print(keys)
X=iris.data

y=iris.target
#we are going to divide half the data into training set to train our modal and rest into testing set

from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5)
type(iris.data)
print(iris.data)
print(y)
iris['feature_names']
len(iris.data)
len(iris.target)
%matplotlib inline

plt.plot(X,y)
from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5)
from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

logreg.fit(X_train,y_train)
print(X_train.shape)

print(X_test.shape)
print(y_train.shape)

print(y_test.shape)
print('X_train')

print(X_train)

print('X_test')

print(X_test)
X_new=[4.4,3.2,1.3,0.2]

y_pred=logreg.predict(X_new)
print(y_pred)
y_pred=logreg.predict(X_test)

print(y_pred)
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)

print(metrics.accuracy_score(y_test,y_pred))
# locating better values of K for higher accuracy

k_range= range (1,26)

scores=[]

for k in k_range :

    knn=KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train,y_train)

    y_pred=knn.predict(X_test)

    scores.append(metrics.accuracy_score(y_test,y_pred))

%matplotlib inline

plt.plot(k_range,scores)
