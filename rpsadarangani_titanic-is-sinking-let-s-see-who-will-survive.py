# import all the needed package

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
train = pd.read_csv('../input/train.csv', index_col=0)

test = pd.read_csv('../input/test.csv', index_col=0)
# train dataframe head

train.head()
# test dataframe head

test.head()
# let check info for dataframe

def check_info(train, test):

  print(train.info())

  print('_'*40)

  print(test.info())

  print('_'*40)
check_info(train,test)
# Check Data Description to have idea of the data

def describe_info(train, test):

  print(train.describe())

  print('_'*40)

  print(test.describe())

  print('_'*40)
describe_info(train, test)
TrainAgeMean = round(train['Age'].mean())

TestAgeMean = round(test['Age'].mean())

print('Train Age Mean :-', TrainAgeMean)

print('Test Age Mean :-', TestAgeMean)
train['Age'] = train['Age'].fillna(value=TrainAgeMean)

test['Age'] = test['Age'].fillna(value=TestAgeMean)

TestFareMode = test['Fare'].mode()[0]

print(TestFareMode)
test['Fare'] = test['Fare'].fillna(value=TestFareMode)
train.drop(columns=['Cabin'], axis=1, inplace=True)

test.drop(columns=['Cabin'], axis=1, inplace=True)
check_info(train, test)
import seaborn as sns

sns.set(style="whitegrid")
corr = train.corr()
f, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(corr, annot=True, linewidths=.5, ax=ax)
train['Family'] = train['Parch'] + train['SibSp']

test['Family'] = test['Parch'] + test['SibSp']
corr = train.corr()

f, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(corr, annot=True, linewidths=.5, ax=ax)
# Value Count Chart

ax = train['Survived'].value_counts().plot.bar()

ax.set_xlabel("Not Survived or Survived")

ax.set_ylabel("Number of People Survied or Not Survived")
# Number of People who survived the even according to Sex

train[train['Survived'] == 1].Sex.value_counts().plot.bar()
# Number of People who not survived the even according to Sex

train[train['Survived'] == 0].Sex.value_counts().plot.bar()
train[(train['Age'] <= 18)].Sex.value_counts()
train[(train['Age'] <= 18) & (train['Sex'] == 'male')].Survived.value_counts().plot.bar()
train[(train['Age'] <= 18) & (train['Sex'] == 'female')].Survived.value_counts().plot.bar()
train[train['Survived'] == 1]['Embarked'].value_counts().plot.bar()
train[train['Survived'] == 0]['Embarked'].value_counts().plot.bar()
# Lets check the distubution of Fare

data = train['Fare']

sns.distplot(data)
sns.boxplot(data)
check_info(train,test)
train.drop(columns=['Name'], axis=1, inplace=True)

test.drop(columns=['Name'], axis=1, inplace=True)
train.drop(columns=['Ticket'], axis=1, inplace=True)

test.drop(columns=['Ticket'], axis=1, inplace=True)
categorical_feature_mask = train.dtypes==object

categorical_cols = train.columns[categorical_feature_mask].tolist()

categorical_cols
test_categorical_feature_mask = test.dtypes==object

test_categorical_cols = test.columns[test_categorical_feature_mask].tolist()

test_categorical_cols
# import labelencoder

from sklearn.preprocessing import LabelEncoder

# instantiate labelencoder object

le = LabelEncoder()
train[categorical_cols] = train[categorical_cols].apply(lambda col: le.fit_transform(col.astype(str)))
test[categorical_cols] = test[categorical_cols].apply(lambda col: le.fit_transform(col.astype(str)))
corr = train.corr()

f, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(corr, annot=True, linewidths=.5, ax=ax)
# Import all libary for Sklearn Model Evalulation

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
# Let's Create our Machine Learning Model to check how it is working but before that we will create a function which will print all the ROC, AUC and Confusion Matix so we will not have to performance these step mannualy.

def run_model(model, name, Xtrain, Xtest, ytrain, ytest):

  print(name + 'Model Details')

  model.fit(Xtrain, ytrain)

  ypred = model.predict(Xtest)

  print("F1 score is: {}".format(f1_score(ytest, ypred)))

  print("AUC Score is: {}".format(roc_auc_score(ytest, ypred)))
from xgboost import XGBClassifier

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier

from sklearn import tree

from sklearn.linear_model import LogisticRegression
gbc = GradientBoostingClassifier()

rfc = RandomForestClassifier()

xgb = XGBClassifier()

ada = AdaBoostClassifier()

clf = tree.DecisionTreeClassifier()

log = LogisticRegression()
model_list = [gbc, rfc, xgb, ada, clf]
model_name = ['Gradient Boosting', 'Random Forest', 'XGBoost', 'Ada Boost', 'Decision Tree']
from sklearn.model_selection import train_test_split
X = train.drop(columns=['Survived'], axis=1)

y = train.Survived
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, test_size=0.2)
gbc.fit(Xtrain, ytrain)

gbcpred = gbc.predict(Xtest)

print("Accuracy score is: {}".format(accuracy_score(ytest, gbcpred)))

print("F1 score is: {}".format(f1_score(ytest, gbcpred)))

print("AUC Score is: {}".format(roc_auc_score(ytest, gbcpred)))
rfc.fit(Xtrain, ytrain)

rfcpred = rfc.predict(Xtest)

print("Accuracy score is: {}".format(accuracy_score(ytest, rfcpred)))

print("F1 score is: {}".format(f1_score(ytest, rfcpred)))

print("AUC Score is: {}".format(roc_auc_score(ytest, rfcpred)))
xgb.fit(Xtrain, ytrain)

xgbpred = xgb.predict(Xtest)

print("Accuracy score is: {}".format(accuracy_score(ytest, xgbpred)))

print("F1 score is: {}".format(f1_score(ytest, xgbpred)))

print("AUC Score is: {}".format(roc_auc_score(ytest, xgbpred)))
ada.fit(Xtrain, ytrain)

adapred = ada.predict(Xtest)

print("Accuracy score is: {}".format(accuracy_score(ytest, adapred)))

print("F1 score is: {}".format(f1_score(ytest, adapred)))

print("AUC Score is: {}".format(roc_auc_score(ytest, adapred)))
clf.fit(Xtrain, ytrain)

clfpred = clf.predict(Xtest)

print("Accuracy score is: {}".format(accuracy_score(ytest, clfpred)))

print("F1 score is: {}".format(f1_score(ytest, clfpred)))

print("AUC Score is: {}".format(roc_auc_score(ytest, clfpred)))
log.fit(Xtrain, ytrain)

logpred = log.predict(Xtest)

print("Accuracy score is: {}".format(accuracy_score(ytest, logpred)))

print("F1 score is: {}".format(f1_score(ytest, logpred)))

print("AUC Score is: {}".format(roc_auc_score(ytest, logpred)))
test['Survived'] = xgb.predict(test)
test.head()
test.to_csv('survived.csv')