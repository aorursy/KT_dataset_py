# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print("Hello World!!!")

import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from matplotlib import pyplot as plt

%matplotlib inline

from sklearn.tree import export_graphviz
train = pd.read_csv('../input/train.csv')

test1 = pd.read_csv('../input/test.csv')

gender_submission = pd.read_csv('../input/gender_submission.csv')

print("train shape ===================================================")

print(train.shape)

print("train indexes Columns dtypes NaN ===================================================")

#for line in zip(train.index, train.columns, train.dtypes,  train.isnull().any()):

#    print(line)

print(train.info())

print("test1 shape ===================================================")

print(test1.shape)

print("test indexes Columns dtypes NaN ===================================================")

#for line in zip(test1.index, test1.columns, test1.dtypes,  test1.isnull().any()):

#    print(line)

print(test1.info())

print("gender_submission shape ===================================================")

print(gender_submission.shape) 

print("gender_submission indexes Columns dtypes NaN ===================================================")

print(gender_submission.info())
train.head(10)
test1.head(10)
gender_submission.head()
test1['Survived'] = gender_submission['Survived']

tmp1 = train['Age'].isnull().sum()

tmp2 = train['Cabin'].isnull().sum()

print(" {0} {1} ".format(tmp1 , tmp2))

tmp1 = test1['Age'].isnull().sum()

tmp2 = test1['Cabin'].isnull().sum()

print(" {0} {1} ".format(tmp1 , tmp2))

print("shape before ===================================================")

print(" {0} {1} ".format(train.shape, test1.shape))

#train.dropna(subset=['Age'], inplace=True)

#test1.dropna(subset=['Age'], inplace=True)

train['Age'].fillna(0, inplace=True)

test1['Age'].fillna(0, inplace=True)

test1['Fare'].fillna(0, inplace=True)

#Зачем то удаляет лишнее:

#train.dropna(axis='columns', inplace=True)

#test1.dropna(axis='columns', inplace=True)

train.drop(columns=['Cabin'], inplace=True)

test1.drop(columns=['Cabin'], inplace=True)

print("shape after ===================================================")

print(" {0} {1} ".format(train.shape, test1.shape))
X_train = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

Y_train = train[['Survived']]

X_test = test1[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

Y_test = test1[['Survived']]

lb_gender = LabelEncoder()

X_train['Sex'] = lb_gender.fit_transform(X_train['Sex'])

X_test['Sex'] = lb_gender.fit_transform(X_test['Sex'])

X_train = pd.get_dummies(X_train, columns=['Pclass'])

X_test = pd.get_dummies(X_test, columns=['Pclass'])

sc1 = StandardScaler()

X_train[['Age', 'Fare']] = sc1.fit_transform(X_train[['Age', 'Fare']] )

X_test[['Age', 'Fare']] = sc1.fit_transform(X_test[['Age', 'Fare']] )
X_train.head()
X_test.head()
dc = DecisionTreeClassifier(random_state=0)

Y_pred = dc.fit(X_train, Y_train)
print(X_train.shape)

print(X_test.shape)
dc.score(X_test, Y_test)
X_train2 = X_train.append(X_test)

Y_train2 = Y_train.append(Y_test)

print(X_train2.shape)

print(Y_train2.shape)
dc3 = DecisionTreeClassifier(random_state=0)

tree_para = {'criterion':['gini','entropy'],'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,15,20]}

clf = GridSearchCV(dc3, tree_para, cv=4)

clf.fit(X_train2, Y_train2)
clf.best_params_
clf.best_estimator_.feature_importances_

plt.barh(np.arange(len(clf.best_estimator_.feature_importances_)), clf.best_estimator_.feature_importances_)

plt.yticks(np.arange(len(X_train.columns)),X_train.columns)
clf.best_score_
dc2 = DecisionTreeClassifier(random_state=0, criterion='gini', max_depth=3)

scores = cross_val_score(dc2, X_train2, Y_train2, cv=4)

scores
def get_grid(data):

    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1

    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1

    return np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))



def test_clf(clf, X, y, cmap=None, fit_clf=False):

   xx,yy = get_grid(X.values)

   if fit_clf:

       clf.fit(X, y)

   if type(y) is pd.DataFrame:

       y = y.values

   predicted = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

   plt.figure(figsize=(8, 8))

   plt.pcolormesh(xx, yy, predicted, cmap='spring')

   plt.scatter(X.values[:, 0], X.values[:, 1], c=y.reshape(-1), cmap=plt.cm.Set1, s=100)

   if fit_clf:

       return clf

def get_tree_dot_view(clf, feature_names=None, class_names=None):

    print(export_graphviz(clf, out_file=None, filled=True, feature_names=feature_names, class_names=class_names))
Xcut = X_train[[X_train.columns[0],X_train.columns[4]]]

Xcut.head()

clf2 = test_clf(DecisionTreeClassifier(random_state=0, criterion='gini', max_depth=3), Xcut , Y_train, fit_clf=True)
X_train.columns[0]

X_train.columns[4]
get_tree_dot_view(clf2, feature_names=['sex','fare'])