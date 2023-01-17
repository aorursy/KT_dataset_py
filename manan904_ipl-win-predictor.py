import numpy as np

import pandas as pd

from sklearn import cross_validation

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

import pickle
df=pd.read_csv('../input/match_data.csv')

df.head()
X=df.values[:,0:1]

y=df.values[:,1]

X.shape
X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)

clf=KNeighborsClassifier()

clf.fit(X_train,y_train)
predict=clf.predict(X_test)
accuracy=accuracy_score(y_test,predict)*100

print (accuracy)
example=np.array([[162]])

check=clf.predict(example)

check