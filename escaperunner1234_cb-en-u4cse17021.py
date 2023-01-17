# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/15cse380-nndl-eval/train.csv")
data.head()

data.shape
data.isnull().sum()
data.fillna(0,inplace=True)
from sklearn.model_selection import train_test_split

Y=data.is_promoted
X=data.drop('is_promoted',axis=1)
Y
X=pd.get_dummies(X)
X.shape

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.5)
x_train.head()
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
pred = logreg.predict(x_test)
pred
logreg.score(x_train,y_train)
from sklearn.metrics import log_loss
log_loss(y_test,pred)
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_learning_curves
title = "Learning Curves (Logistic Regression)"
clf = KNeighborsClassifier(n_neighbors=10)

plot_learning_curves(x_train, y_train, x_test, y_test, clf)

df = pd.read_csv("../input/15cse380-nndl-eval/test.csv")
df.head()
df.shape
df.isnull().sum()
df.fillna(0,inplace=True)
df.shape
df = pd.get_dummies(df)

df.shape
y_train=y_train[0:5808]
logreg.fit(df,y_train)
x_test = x_test[0:5808]
pred = logreg.predict(x_test)
logreg.score(df,y_train)
y_test=y_test[0:5808]
log_loss(y_test,pred)
