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
df = pd.read_csv('/kaggle/input/titanic/train.csv')



df.info()

df.head()
data = df.drop(columns=["PassengerId","Name","Ticket",'Cabin',"Age"])

data.head()
data["Sex"] = data["Sex"].astype('category')

data["Embarked"] = data["Embarked"].astype('category')

data["Sex_num"] = data["Sex"].cat.codes

data["Embarked_num"] = data["Embarked"].cat.codes

data.head()
X = data.drop(columns=["Sex","Embarked","Survived"]).values

y = data["Survived"].values
# 5-fold

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, random_state=1, shuffle=True)

print('fold = ',kf.get_n_splits(X))
from sklearn.model_selection import cross_val_score

from sklearn import metrics
# Decision Tree

from sklearn.tree import DecisionTreeClassifier

print("# Decision Tree")

model = DecisionTreeClassifier()

scores = cross_val_score(model, X, y, scoring='accuracy', cv=kf, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

for i,(train_index, test_index) in enumerate(kf.split(X)):

    print("Fold number ",i+1)

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    model = DecisionTreeClassifier()

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    print('Accuracy: %.3f ' % (metrics.accuracy_score(y_test,y_pred)))

    print(metrics.classification_report(y_test,y_pred))

    
# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

print('# Gaussian Naive Bayes')

model = GaussianNB()

scores = cross_val_score(model, X, y, scoring='accuracy', cv=kf, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

for i,(train_index, test_index) in enumerate(kf.split(X)):

    print("Fold number ",i+1)

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    model = GaussianNB()

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:",metrics.accuracy_score(y_test,y_pred))

    print(metrics.classification_report(y_test,y_pred))
# Neural Network

from sklearn.neural_network import MLPClassifier

print('# Neural Network')

model = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)

scores = cross_val_score(model, X, y, scoring='accuracy', cv=kf, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

for i,(train_index, test_index) in enumerate(kf.split(X)):

    print("Fold number ",i+1)

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    model = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=800)

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:",metrics.accuracy_score(y_test,y_pred))

    print(metrics.classification_report(y_test,y_pred))