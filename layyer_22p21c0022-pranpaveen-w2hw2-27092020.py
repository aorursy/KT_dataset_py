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
from matplotlib import pyplot as plt
import seaborn as sns
data = pd.read_csv('/kaggle/input/titanic/train.csv')
data.head()
data.info()
data.isna().sum()
data = data.drop(columns=["Name","Ticket",'Cabin',"Age"])
data["Sex"] = data["Sex"].astype('category')
data["Embarked"] = data["Embarked"].astype('category')
data["Sex_num"] = data["Sex"].cat.codes
data["Embarked_num"] = data["Embarked"].cat.codes
data.head()
X = data.drop(columns=["Sex","Embarked","Survived"]).values
y = data["Survived"].values
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
kf.get_n_splits(X)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def train_model(model,X_train,y_train,X_test,y_test):
    model.fit(X_train,y_train)
    
    tree_pred = model.predict(X_test)
    print("\t\tAccuracy:",accuracy_score(y_test,tree_pred))
    print(classification_report(y_test,tree_pred))
print("------Decision Tree------")
for i,(train_index, test_index) in enumerate(kf.split(X)):
    print("Fold number ",i)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = DecisionTreeClassifier(criterion="entropy")
    train_model(model,X_train,y_train,X_test,y_test)
print("------Naive bayes------")
for i,(train_index, test_index) in enumerate(kf.split(X)):
    print("Fold number ",i)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = GaussianNB()
    train_model(model,X_train,y_train,X_test,y_test)
print("------Neural Network------")
for i,(train_index, test_index) in enumerate(kf.split(X)):
    print("Fold number ",i)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = MLPClassifier(hidden_layer_sizes=(128,64,16))
    train_model(model,X_train,y_train,X_test,y_test)
