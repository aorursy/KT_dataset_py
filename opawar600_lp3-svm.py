# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/heart.csv")
df.columns
sns.countplot(df.target)
sns.scatterplot(x='age',y='chol',hue='target',data=df)
sns.pairplot(df,hue='target',)
from sklearn.svm import SVC

clf = SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,

  decision_function_shape='ovr', degree=3, gamma=1e-05, kernel='rbf',

  max_iter=-1, probability=False, random_state=None, shrinking=True,

  tol=0.001, verbose=False)
from sklearn.model_selection import train_test_split

X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal']]

y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,stratify=df.target)
clf.fit(X_train,y_train)

clf.score(X_test,y_test)
from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(y_test,clf.predict(X_test))
print(classification_report(y_test,clf.predict(X_test)))
from sklearn.model_selection import GridSearchCV
param = {'C':[0.1,1,10,100,1000,10000],'gamma':[0.00000001,0.00001,0.000001]}
grid = GridSearchCV(SVC(), param,verbose=3)
grid.fit(X_train,y_train)
grid.best_estimator_
df.iloc[0:5:2]