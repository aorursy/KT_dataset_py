# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input 

#directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns
for keys in iris.keys() :

    print(keys)
#load the iris data as it is well known so we have it in our sklearn datasets 

#other wise its csv file can also be called and read with the help of pandas

from sklearn.datasets import load_iris
iris=load_iris()
#target of iris data .. see as it is a numoy array so it understands only numerical value thus we 

# have 0,1,2 one for each three types of flowers 

print(iris['target'])
#data means the values we are using to interpreate type of flower or you can say features .

# these are sepal length length and petal width here and petal length and petal width.

print(iris['data'])
X= iris.data

y=iris.target
#let"s check the size of iris data and its target

print(len(iris.data))

print(len(iris.target))
plt.plot(X,y)

plt.show()
print(iris.feature_names)
# Reformat column names

import re

new_feature = []

for feature in iris['feature_names']:

    new_feature.append(re.sub(r'(\w+) (\w+) \((\w+)\)',r'\1_\2_\3',feature))

print(new_feature)
#importing k neighbour classifier

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1)

print(knn)
knn.fit(X,y)
X.reshape(1,-1)

len(iris.data)
len(iris.target)
X.reshape(-1,1)
#run this code in jupyter notebook and you get the predicted answer

knn.predict([2,3,1,4])
iris['feature_names']
plt.scatter(iris['sepal length (cm)'],y)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X,y)

#predicting the data 

y_pred=logreg.predict(X)

y_pred
#to check the accuracy of prediction

from sklearn import metrics

print (metrics.accuracy_score(y,y_pred))
knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(X,y)

y_pred= knn.predict(X)

print(metrics.accuracy_score(y,y_pred))
#for different values of k we find different accuracies