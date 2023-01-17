# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression,SGDClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/voice.csv')

data.head(5)
data.describe()
sns.set()

fig=plt.gcf()

fig.set_size_inches(10,10)

sns.countplot(data['label'])

plt.show()
data=data.replace('male',1)

data=data.replace('female',0)

data.label.sample(5)
corr = data.corr()

fig=plt.gcf()

fig.set_size_inches(18,10)

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=90,

    horizontalalignment='right'

);
x_train,x_test,y_train,y_test=train_test_split(data.drop('label',axis=1),data['label'],test_size=0.25)

x_train.shape,y_train.shape
lr=LogisticRegression()

nb=GaussianNB()

knn=KNeighborsClassifier(n_neighbors=2)

sgd=SGDClassifier(loss='modified_huber',shuffle=True)

dtree=DecisionTreeClassifier(max_depth=10,min_samples_leaf=15)

rfm=RandomForestClassifier(n_estimators=70)

svc=SVC(kernel='linear')

lr.fit(x_train,y_train)

nb.fit(x_train,y_train)

knn.fit(x_train,y_train)

sgd.fit(x_train,y_train)

dtree.fit(x_train,y_train)

rfm.fit(x_train,y_train)

svc.fit(x_train,y_train)

print('Accuracy of various algorithms on test data:-')

print('Logsitc Regression: '+str(lr.score(x_test,y_test)))

print('Naïve Bayes: '+str(nb.score(x_test,y_test)))

print('Stochastic Gradient Descent: '+str(sgd.score(x_test,y_test)))

print('K-Nearest Neighbours: '+str(knn.score(x_test,y_test)))

print('Decision Tree: '+str(dtree.score(x_test,y_test)))

print('Random Forest: '+str(rfm.score(x_test,y_test)))

print('Support Vector Machine: '+str(svc.score(x_test,y_test)))
