# Data file
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Impory
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier 
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
train = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_train.csv', header=0)
train.head(10)
test = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_test.csv', header=0)
test.head(10)
X_train = train.drop(columns='label')
y_train = train['label']
X_test = test.drop(columns='label')
y_test = test['label']
# RandomForest==============

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print('='*20)
print('RandomForestClassifier')
print(f'accuracy of train set: {rf.score(X_train, y_train)}')
print(f'accuracy of test set: {rf.score(X_test, y_test)}')



# SVC==============

svc = SVC(verbose=True, random_state=0)
svc.fit(X_train, y_train)
print('='*20)
print('SVC')
print(f'accuracy of train set: {svc.score(X_train, y_train)}')
print(f'accuracy of test set: {svc.score(X_test, y_test)}')

# LinearSVC==============

lsvc = LinearSVC(verbose=True)
lsvc.fit(X_train, y_train)
print('='*20)
print('LinearSVC')
print(f'accuracy of train set: {lsvc.score(X_train, y_train)}')
print(f'accuracy of test set: {lsvc.score(X_test, y_test)}')

# k-近傍法（k-NN）==============

knn = KNeighborsClassifier(n_neighbors=3) #引数は分類数
knn.fit(X_train, y_train)
print('='*20)
print('KNeighborsClassifier')
print(f'accuracy of train set: {knn.score(X_train, y_train)}')
print(f'accuracy of test set: {knn.score(X_test, y_test)}')

# 決定木==============

decisiontree = DecisionTreeClassifier(max_depth=3, random_state=0)
decisiontree.fit(X_train, y_train)
print('='*20)
print('DecisionTreeClassifier')
print(f'accuracy of train set: {decisiontree.score(X_train, y_train)}')
print(f'accuracy of test set: {decisiontree.score(X_test, y_test)}')

# SGD Classifier==============

sgd = SGDClassifier()
sgd.fit(X_train, y_train)
print('='*20)
print('SGD Classifier')
print(f'accuracy of train set: {sgd.score(X_train, y_train)}')
print(f'accuracy of test set: {sgd.score(X_test, y_test)}')

# Gradient Boosting Classifier==============

gradientboost = GradientBoostingClassifier(random_state=0)
gradientboost.fit(X_train, y_train)
print('='*20)
print('GradientBoostingClassifier')
print(f'accuracy of train set: {gradientboost.score(X_train, y_train)}')
print(f'accuracy of test set: {gradientboost.score(X_test, y_test)}')

# XGBClassifier==============

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
print('='*20)
print('XGB Classifier')
print(f'accuracy of train set: {xgb.score(X_train, y_train)}')
print(f'accuracy of test set: {xgb.score(X_test, y_test)}')

# LGBMClassifier==============

lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)
print('='*20)
print('LGBM Classifier')
print(f'accuracy of train set: {lgbm.score(X_train, y_train)}')
print(f'accuracy of test set: {lgbm.score(X_test, y_test)}')

# CatBoostClassifier==============

catboost = CatBoostClassifier()
catboost.fit(X_train, y_train)
print('='*20)
print('CatBoost Classifier')
print(f'accuracy of train set: {catboost.score(X_train, y_train)}')
print(f'accuracy of test set: {catboost.score(X_test, y_test)}')


# VotingClassifier==============

from sklearn.ensemble import VotingClassifier

# voting に使う分類器を用意する
estimators = [
  ("rf", rf),
  ("svc", svc),
  ("lsvc", lsvc),
  ("knn", knn),
  ("decisiontree", decisiontree),
  ("sgd", sgd),
  ("gradientboost", gradientboost),
]

vote = VotingClassifier(estimators=estimators)
vote.fit(X_train, y_train)
print('='*20)
print('VotingClassifier')
print(f'accuracy of train set: {vote.score(X_train, y_train)}')
print(f'accuracy of test set: {vote.score(X_test, y_test)}')

