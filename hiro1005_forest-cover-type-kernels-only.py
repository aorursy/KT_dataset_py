# Data files
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Import
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
train = pd.read_csv('/kaggle/input/forest-cover-type-kernels-only/train.csv.zip', header=0)
train.head(10)
test = pd.read_csv('/kaggle/input/forest-cover-type-kernels-only/test.csv.zip', header=0)
test.head(5)
submission = pd.read_csv('/kaggle/input/forest-cover-type-kernels-only/sample_submission.csv.zip', header=0)
submission.head(10)
train['Cover_Type'].value_counts()
target_col = 'Cover_Type'
drop_col = ['Id', 'Cover_Type']

train_feature = train.drop(columns=drop_col)
train_target = train['Cover_Type']
test_feature = test.drop(columns='Id')
submission_id = test['Id'].values

X_train, X_test, y_train, y_test = train_test_split(train_feature, train_target, test_size=0.2, random_state=0)
# RandomForest==============

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print('='*20)
print('RandomForestClassifier')
print(f'accuracy of train set: {rf.score(X_train, y_train)}')
print(f'accuracy of test set: {rf.score(X_test, y_test)}')

rf_prediction = rf.predict(test_feature)
rf_prediction


# Create submission data
# rf_submission = pd.DataFrame({"Id":submission_id, "Cover_Type":rf_prediction})
# rf_submission.to_csv("RandomForest_submission.csv", index=False)


# SVC==============

svc = SVC(verbose=True, random_state=0)
svc.fit(X_train, y_train)
print('='*20)
print('SVC')
print(f'accuracy of train set: {svc.score(X_train, y_train)}')
print(f'accuracy of test set: {svc.score(X_test, y_test)}')

svc_prediction = svc.predict(test_feature)
svc_prediction

# LinearSVC==============

lsvc = LinearSVC(verbose=True)
lsvc.fit(X_train, y_train)
print('='*20)
print('LinearSVC')
print(f'accuracy of train set: {lsvc.score(X_train, y_train)}')
print(f'accuracy of test set: {lsvc.score(X_test, y_test)}')

lsvc_prediction = lsvc.predict(test_feature)
lsvc_prediction

# k-近傍法（k-NN）==============

knn = KNeighborsClassifier(n_neighbors=3) #引数は分類数
knn.fit(X_train, y_train)
print('='*20)
print('KNeighborsClassifier')
print(f'accuracy of train set: {knn.score(X_train, y_train)}')
print(f'accuracy of test set: {knn.score(X_test, y_test)}')

knn_prediction = knn.predict(test_feature)
knn_prediction

# 決定木==============

decisiontree = DecisionTreeClassifier(max_depth=3, random_state=0)
decisiontree.fit(X_train, y_train)
print('='*20)
print('DecisionTreeClassifier')
print(f'accuracy of train set: {decisiontree.score(X_train, y_train)}')
print(f'accuracy of test set: {decisiontree.score(X_test, y_test)}')

decisiontree_prediction = decisiontree.predict(test_feature)
decisiontree_prediction

# SGD Classifier==============

sgd = SGDClassifier()
sgd.fit(X_train, y_train)
print('='*20)
print('SGD Classifier')
print(f'accuracy of train set: {sgd.score(X_train, y_train)}')
print(f'accuracy of test set: {sgd.score(X_test, y_test)}')

sgd_prediction = sgd.predict(test_feature)
sgd_prediction

# Gradient Boosting Classifier==============

gradientboost = GradientBoostingClassifier(random_state=0)
gradientboost.fit(X_train, y_train)
print('='*20)
print('GradientBoostingClassifier')
print(f'accuracy of train set: {gradientboost.score(X_train, y_train)}')
print(f'accuracy of test set: {gradientboost.score(X_test, y_test)}')

gradientboost_prediction = gradientboost.predict(test_feature)
gradientboost_prediction

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

vote_prediction = vote.predict(test_feature)
vote_prediction
