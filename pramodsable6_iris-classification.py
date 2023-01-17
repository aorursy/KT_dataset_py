import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

import os
iris = pd.read_csv('../input/Iris.csv')
print(os.listdir('../input'))
iris.tail()
iris.Species.unique()
iris.shape
le = LabelEncoder()
iris.Species=le.fit_transform(iris.Species)
iris.tail()
X=iris.drop('Species',axis=1)   #petal length and petal width as features

y=iris.Species
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=1,stratify = y)
print('Label counts in y: ',np.bincount(y))
print('Label counts in y_train: ',np.bincount(y_train))
print('Label counts in y_test: ',np.bincount(y_test))
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
import matplotlib.pyplot as plt

plt.hist(iris.PetalLengthCm,color='red')

plt.hist(iris.PetalWidthCm,color='blue')

plt.hist(iris.SepalLengthCm,color='yellow')

plt.hist(iris.SepalWidthCm,color='magenta')
import matplotlib.pyplot as plt

plt.hist(X_train_std[:,0],color='red')

plt.hist(X_train_std[:,1],color='blue')

plt.hist(X_train_std[:,2],color='yellow')

plt.hist(X_train_std[:,3],color='magenta')
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(X_train_std, y_train)
lr.score(X_test_std,y_test)


import matplotlib.pyplot as plt

from mlxtend.plotting import plot_decision_regions

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=1)

tree.fit(X_train_std, y_train)

tree.score(X_test_std,y_test)
y_pred=tree.predict(X_test_std)
accuracy_score(y_test,y_pred)


import matplotlib.pyplot as plt

from mlxtend.plotting import plot_decision_regions

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='gini',random_state=1)

tree.fit(X_train_std, y_train)
tree.score(X_test_std,y_test)
import matplotlib.pyplot as plt

from mlxtend.plotting import plot_decision_regions

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='gini',max_depth=2,random_state=1)

tree.fit(X_train_std, y_train)

tree.score(X_test_std,y_test)
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy',n_estimators=25,random_state=1)
forest.fit(X_train_std, y_train)
forest.score(X_test_std,y_test)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5,p=2)
knn.fit(X_train_std,y_train)
knn.score(X_test_std,y_test)