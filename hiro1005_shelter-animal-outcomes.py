# Data file
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
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
train = pd.read_csv('/kaggle/input/shelter-animal-outcomes/train.csv.gz', header=0)
train.head(10)
test = pd.read_csv('/kaggle/input/shelter-animal-outcomes/test.csv.gz', header=0)
test.head(10)
submission = pd.read_csv('/kaggle/input/shelter-animal-outcomes/sample_submission.csv.gz', header=0)
submission.head(10)
train_mid = train.copy() 
train_mid['train_or_test'] = 'train'

test_mid = test.copy()
test_mid['train_or_test'] = 'test'

test_mid['OutcomeType'] = 9 # Put 9 to Survived column temporary
alldata = pd.concat([train_mid, test_mid], sort=False, axis=0).reset_index(drop=True) 

# Check all of data
print('The size of the train data:' + str(train.shape))
print('The size of the test data:' + str(test.shape))
print('The size of the submission data:' + str(submission.shape))
print('The size of the alldata data:' + str(alldata.shape))
alldata
breed = pd.get_dummies(alldata['Breed'])
color = pd.get_dummies(alldata['Color'])
sexuponOutcome = pd.get_dummies(alldata['SexuponOutcome'])
animalType = pd.get_dummies(alldata['AnimalType'])
outcomeSubtype = pd.get_dummies(alldata['OutcomeSubtype'])

del alldata['Color']
del alldata['Breed']
del alldata['SexuponOutcome']
del alldata['AnimalType']
del alldata['OutcomeSubtype']

# With or without a name, the importance may change. will do One-Hot encording if name have or not
alldata['Name'].isnull()
alldata['Name'].value_counts()

# Replacing NaN to Int 0
alldata['Name'] = alldata['Name'].fillna(0)
# Regular expression Extract the name of an English word and insert Int 1
alldata['Name'] = alldata['Name'].replace(r"\b[\\u\\l]+\b",1, regex=True)

# Insert 1
if alldata['Name'] is not 0:
  alldata['Name'].fillna(1)

# Datetime also important features. But delete temporary
del alldata['DateTime']
del alldata['AgeuponOutcome']

# Marge all of One-Hot column
alldata = pd.concat([alldata, breed, color, sexuponOutcome, animalType, outcomeSubtype], axis=1)

# Split alldata into train and test
train = alldata.query('train_or_test == "train"')
test = alldata.query('train_or_test == "test"')

train['OutcomeType'] = train['OutcomeType'].replace("Adoption",0).replace("Died",1).replace("Euthanasia",2).replace("Return_to_owner",3).replace("Transfer",4)

target_col = 'OutcomeType'
drop_col = ['AnimalID', 'OutcomeType', 'train_or_test', 'ID', 'Name']

train_feature = train.drop(columns=drop_col)
train_target = train[target_col]
test_feature = test.drop(columns=drop_col)

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
# rf_submission = pd.DataFrame({"ID":submission['ID'], "OutcomeType":rf_prediction})
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

vote_prediction = vote.predict(test_feature)
vote_prediction
