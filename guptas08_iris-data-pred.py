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
#Loading Dataset from sklearn

from sklearn.datasets import load_iris

from sklearn import metrics

iris=load_iris()
print (iris.data)

x=iris.data
print (iris.feature_names)
print (iris.target)

y=iris.target
print(iris.target_names)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x,y)
y_predi=knn.predict(x)
print(metrics.accuracy_score(y,y_predi))
new=[[3,5,4,2],[5,4,3,2]]



knn.predict(new)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=4)
print (x_train.shape)

print (x_test.shape)
print (y_train.shape)

print (y_test.shape)
k_range=range(1,26)

scores=[]

for k in k_range:

    knn=KNeighborsClassifier(n_neighbors=k)

    knn.fit(x_train,y_train)

    y_pred=knn.predict(x_test)

    scores.append(metrics.accuracy_score(y_test,y_pred))
print(scores)
import matplotlib.pyplot as plt

%matplotlib inline



plt.plot(k_range,scores)

plt.xlabel('Value of K for KNN')

plt.ylabel('Testing Accuracy')