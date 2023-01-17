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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
train_data
test_data.head(10)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
train_number = train_data.shape[0]
train_label = train_data['Survived']
train_drop = train_data.drop('Survived', axis=1)
train_drop.head(10)
print(train_drop.shape, test_data.shape)
train = train_drop.append(test_data)
train.shape
train_number
train = train.drop('PassengerId', axis=1)
train = train.drop('Name', axis=1)
# male = 1, female = 0;C = 0, Q = 1, S = 2
train['Sex'].replace('male', 1, inplace=True)
train['Sex'].replace('female', 0, inplace=True)
train['Embarked'].replace('C', 0, inplace=True)
train['Embarked'].replace('Q', 1, inplace=True)
train['Embarked'].replace('S', 2, inplace=True)
train.head(10)
heatmap_name = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
heatmap_dataframe = train[heatmap_name]
plt.figure(figsize=(15, 10))
sns.heatmap(heatmap_dataframe.corr(), cmap = 'Reds')
plt.show()
train.isnull().sum()
plt.figure(figsize=(15, 10))
train.isnull().sum().plot(kind='bar')
plt.show()

print('carbin : ', train['Cabin'].isnull().sum() / train.shape[0])
train = train.drop('Cabin', axis=1)
train['Age'] = train['Age'].fillna(train['Age'].mode()[0])
train['Fare'] = train['Fare'].fillna(train['Fare'].mode()[0])
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
train.isnull().sum()
train = pd.get_dummies(train).reset_index(drop=True)
train.shape
train_data = train[:train_number]
test_data = train[train_number : ]
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import svm
#x_train, x_test, y_train, y_test = train_test_split(train_data, train_label, test_size = 0.3, random_state = 0)
Clf = svm.SVC(kernel='rbf', C=0.5)
scores = cross_val_score(Clf, train_data, train_label, cv=5)
print('SVM : ', scores.mean(), scores.std())
GNB = GaussianNB()
scores = cross_val_score(GNB, train_data, train_label, cv=5)
print('GNB: ', scores.mean(), scores.std())
DTree = tree.DecisionTreeClassifier()
scores = cross_val_score(DTree, train_data, train_label, cv=5)
print('DTree : ', scores.mean(), scores.std())
KNN = KNeighborsClassifier()
scores = cross_val_score(KNN, train_data, train_label, cv=5)
print('KNN : ', scores.mean(), scores.std())
RTree = RandomForestClassifier(n_estimators = 12)
scores = cross_val_score(RTree, train_data, train_label, cv=5)
print('RTree : ', scores.mean(), scores.std())
ETree = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(ETree, train_data, train_label, cv=5)
print('ETree : ', scores.mean(), scores.std())
DecisionTreeBagging = BaggingClassifier(DTree, n_estimators = 100, max_features = 0.5)
scores = cross_val_score(DTree, train_data, train_label, cv=5)
print('DecisionTreeBagging : ', scores.mean(), scores.std())
DecisionTreeBagging = BaggingClassifier(RTree, n_estimators = 100, max_features = 0.5)
scores = cross_val_score(RTree, train_data, train_label, cv=5)
print('RandomForestBagging : ', scores.mean(), scores.std())
Ada = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(Ada, train_data, train_label, cv=5)
print('Ada : ', scores.mean(), scores.std())
GBT = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
scores = cross_val_score(GBT, train_data, train_label, cv=5)
print('GBT : ', scores.mean(), scores.std())
blending = VotingClassifier(estimators=[('DTree', DTree), ('RTree', RTree), ('ETree', ETree), ('Ada', Ada), ('GBT', GBT)], voting='soft', weights = [2, 2, 2, 2, 1])
scores = cross_val_score(blending, train_data, train_label, cv=5)
print('blending : ', scores.mean(), scores.std())
estimators = [('DTree', DTree), ('RTree', RTree), ('ETree', ETree), ('Ada', Ada), ('Blending', blending)]
Stack = StackingClassifier(estimators=estimators, final_estimator=GBT)
scores = cross_val_score(Stack, train_data, train_label, cv=5)
print('Stack : ', scores.mean(), scores.std())
blending.fit(train_data, train_label)
Stack.fit(train_data, train_label)
print('finish')
y_bleanding = blending.predict(train_data)
y_Stack = Stack.predict(train_data)
print(accuracy_score(y_bleanding, train_label))
print(accuracy_score(y_Stack, train_label))
y_test = blending.predict(test_data)
summition = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
summition = summition.drop('Survived', axis=1)
summition['Survived'] = y_test
summition.to_csv('result.csv', index = False)
summition
