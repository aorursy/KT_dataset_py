# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import sklearn as sk
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('/kaggle/input/titanic/train.csv')
#data = data.fillna(0, axis=1)
data['Sex'] = LabelEncoder().fit_transform(data['Sex'])
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
Y = LabelEncoder().fit_transform(data['Survived'])
print(X[:5])
print(Y[:5])
sur, tot = 0, len(Y)
for i in Y:
    sur = sur + i
print('Total = ', tot)
print('Survived = ', sur)
class_weight = {0: (sur/tot)**0.5, 1: (1 - (sur/tot))**0.5}
print(class_weight)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer


X = SimpleImputer().fit_transform(X)
#Y = SimpleImputer().fit_transform(Y)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.3)


clf = DecisionTreeClassifier(max_depth = 9, class_weight= class_weight, random_state= 1)
clf.fit(X_train, Y_train)
acc = clf.score(X_val, Y_val)
print('Accuracy = ', acc)
pred = clf.predict(X_val)
conf_mat = pd.DataFrame(confusion_matrix(Y_val, pred), columns= ['Pred Dead', 'Pred Surv'], index= ['Act Dead', 'Act Surv'])
print(conf_mat)
from sklearn import tree
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(100,100))
_ = tree.plot_tree(clf)
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score


rf = RandomForestClassifier(n_estimators= 120, max_depth = 9, random_state= 2, class_weight= class_weight)
rf.fit(X_train, Y_train)
pred_rf = rf.predict(X_val)
acc_rf = accuracy_score(Y_val, pred_rf)
print('With RF Classifier, accuracy = ', acc_rf)
conf_mat_rf = pd.DataFrame(confusion_matrix(Y_val, pred_rf), columns= ['Pred Dead', 'Pred Surv'], index= ['Act Dead', 'Act Surv'])
print(conf_mat_rf)
bag_clf = BaggingClassifier(DecisionTreeClassifier(max_depth= 5, class_weight= class_weight), n_estimators= 120, max_samples= 1.0, random_state= 3)
bag_clf.fit(X_train, Y_train)
pred_bag = bag_clf.predict(X_val)
acc_bag = accuracy_score(Y_val, pred_bag)
print('With Bagging Classifier, accuracy = ', acc_bag)
conf_mat_bag = pd.DataFrame(confusion_matrix(Y_val, pred_bag), columns= ['Pred Dead', 'Pred Surv'], index= ['Act Dead', 'Act Surv'])
print(conf_mat_bag)
# F-1 Scores
from sklearn.metrics import f1_score
f1_dtree = f1_score(Y_val, pred)
f1_rf = f1_score(Y_val, pred_rf)
f1_bag = f1_score(Y_val, pred_bag)
print('-------F1 Scores---------')
print('Decision Tree: ', f1_dtree)
print('Random Forest: ', f1_rf)
print('Bagging Classifier: ', f1_bag)
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
test_data['Sex'] = LabelEncoder().fit_transform(test_data['Sex'])
X_test = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
X_test = SimpleImputer().fit_transform(X_test)
pred_test_bag = rf.predict(X_test)
#test_data['Survived'] = pred_test_bag
#test_data.to_csv('Submission.csv')
sub = pd.DataFrame()
sub['PassengerId'] = test_data['PassengerId']
sub['Survived'] = pred_test_bag
sub.to_csv('Submission.csv', index=False)
