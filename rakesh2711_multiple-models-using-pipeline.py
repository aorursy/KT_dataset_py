# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB,GaussianNB

from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/sonar-data-set/sonar.all-data.csv')



## Separate Training & Validation Dataset

from sklearn.model_selection import train_test_split

X = data.values[:,0:60]

Y = data.values[:,60]

X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size = 0.2, random_state=42)

pipe1 = Pipeline([('LR', LogisticRegression())])

pipe1.fit(X_train ,Y_train)

print(accuracy_score(Y_val, pipe1.predict(X_val)))
pipe2 = Pipeline([('scaled' , StandardScaler()),

                 ('LR' ,LogisticRegression())])

pipe2.fit(X_train ,Y_train)

print(accuracy_score(Y_val, pipe2.predict(X_val)))
pipelines = []

pipelines.append(('scaledLR' , (Pipeline([('scaled' , StandardScaler()),('LR' ,LogisticRegression())]))))

pipelines.append(('scaledKNN' , (Pipeline([('scaled' , StandardScaler()),('KNN' ,KNeighborsClassifier())]))))

pipelines.append(('scaledDT' , (Pipeline([('scaled' , StandardScaler()),('DT' ,DecisionTreeClassifier())]))))

pipelines.append(('scaledSVC' , (Pipeline([('scaled' , StandardScaler()),('SVC' ,SVC())]))))

pipelines.append(('scaledMNB' , (Pipeline([('scaled' , StandardScaler()),('MNB' ,GaussianNB())]))))



model_name = []

results = []

for pipe ,model in pipelines:

    kfold = KFold(n_splits=10, random_state=42)

    crossv_results = cross_val_score(model , X_train ,Y_train ,cv =kfold , scoring='accuracy')

    results.append(crossv_results)

    model_name.append(pipe)

    msg = "%s: %f (%f)" % (model_name, crossv_results.mean(), crossv_results.std())

    print(msg)

    

# Compare different Algorithms

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(model_name)

plt.show()