# Data file

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

from sklearn.linear_model import SGDClassifier

from sklearn.svm import LinearSVC

from sklearn.svm import SVC
train = pd.read_csv('/kaggle/input/costa-rican-household-poverty-prediction/train.csv', header=0)

train.head()
test = pd.read_csv('/kaggle/input/costa-rican-household-poverty-prediction/test.csv', header=0)

test.head()
submission = pd.read_csv('/kaggle/input/costa-rican-household-poverty-prediction/sample_submission.csv', header=0)

submission.head()
train_mid = train.copy()

train_mid['train_or_test'] = 'train'



test_mid = test.copy()

test_mid['train_or_test'] = 'test'



test_mid['Target'] = 9



alldata = pd.concat([train_mid, test_mid], sort=False, axis=0).reset_index(drop=True)



print('The size of the train data:' + str(train.shape))

print('The size of the test data:' + str(test.shape))

print('The size of the submission data:' + str(submission.shape))

print('The size of the alldata data:' + str(alldata.shape))
# Deleting unnecessary column

del alldata['idhogar']

del alldata['dependency']

del alldata['edjefa']

del alldata['edjefe']



# Median number insert to NaN in alldata

alldata.fillna(alldata.median(), inplace=True)



# Split alldata into train and test

train = alldata.query('train_or_test == "train"')

test = alldata.query('train_or_test == "test"')
alldata
# Check how many column show and change number of it show 

pd.get_option("display.max_columns")

pd.set_option('display.max_columns', 150)



# Check how many rows show and change number of it show 

pd.get_option("display.max_rows")

pd.set_option('display.max_rows', 150)
alldata.dtypes
# Definition Target feature and drop some unnecessary column

target_col = 'Target'

drop_col = ['Id', 'Target', 'train_or_test']



train_feature = train.drop(columns=drop_col)

train_target = train[target_col]

test_feature = test.drop(columns=drop_col)

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
