# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import sklearn.preprocessing as preprocessing
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from scipy.stats import skew 


import os
print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(train.shape, test.shape)
print(test.head())
# Any results you write to the current directory are saved as output.
# Check for null values in training and test set
print('Null values in Training set:', train.isnull().sum().sum(), ', Total values in Training set:', train.isnull().count().sum())
print('Null values in Test set:', test.isnull().sum().sum(), ', Total values in Test set:', test.isnull().count().sum())
# Some data vizualization for accessing scaling, standardization and normalization need
skewness = train.iloc[:,2:].apply(lambda x: skew(x.dropna()))
print('Skewness in data')
print(skewness.sort_values(ascending=False)[:10])
train[['margin16', 'shape2']].hist()
# Lets prepare training and test set for ML models
le = preprocessing.LabelEncoder().fit(train.species)
labels = le.transform(train.species)
classes = le.classes_

test_id = test.id
train_df = train.drop(['id', 'species'], axis=1)
test_df = test.drop(['id'], axis=1)
print(train_df.head(2))
#print(le.classes_)
#print(train.species[:10], labels[:10])
# We want to scale the data for better performance on ML, we will use standardscaler

scaler = preprocessing.StandardScaler().fit(train_df)
print(scaler)

train_df = pd.DataFrame(scaler.transform(train_df), columns=train_df.columns)
test_df = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)

# Visualize standardscaler transformation
sns.set()
scaler = preprocessing.StandardScaler().fit(train[['shape2', 'shape3', 'shape1', 'margin16']])
scaled_train = scaler.transform(train[['shape2', 'shape3', 'shape1', 'margin16']]) 
df_dist = pd.DataFrame({'shape3_nt': train['shape3'], 'shape3_tsf': scaled_train[:,1]})
#print(df_dist.head())
df_dist.hist()
feature_corr = train_df.corr(method='pearson')
sns.set()
sns.clustermap(feature_corr)
# We will keep 30% data for test and rest for training
sss = StratifiedShuffleSplit(y=labels, test_size=0.2, random_state=0, n_iter=1)

for train_ind, test_ind in sss:
    print(len(train_ind), len(test_ind))
    print(test_ind[:5])
    x_train, x_test = train_df.iloc[train_ind,], train_df.iloc[test_ind,]
    y_train, y_test = labels[train_ind], labels[test_ind]
print(x_test.head(2), y_test[:2])
from sklearn.metrics import accuracy_score, log_loss
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# Grid search for parameter estimation
def gridSearch(model, parameters, scoring='accuracy'):
    clf = GridSearchCV(model, parameters, scoring)
    return clf
# Let us start by SVM
parameters = {'kernel': ('linear', 'rbf'), 'C': [0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1, 10]}
svc = SVC(probability=True, cache_size=1000)
clf = gridSearch(svc, parameters)
print(clf)
clf.fit(x_train, y_train)
print(clf.best_params_)
print(clf.best_score_)
train_predictions = clf.predict(x_test)
acc = accuracy_score(y_test, train_predictions)
print("Accuracy: {:.4%}".format(acc))
# Let us start by SVMnu
parameters = {'kernel': ('rbf',), 'gamma': [0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1]}
nusvc = NuSVC(probability=True, cache_size=1000)
nuclf = gridSearch(nusvc, parameters)
print(nuclf)
nuclf.fit(x_train, y_train)
print(nuclf.best_params_)
print(nuclf.best_score_)
nu_train_predictions = nuclf.predict(x_test)
nu_acc = accuracy_score(y_test, nu_train_predictions)
print("Accuracy: {:.4%}".format(nu_acc))

nu_train_predictions = nuclf.predict_proba(x_test)
nu_ll = log_loss(y_test, rf_train_prediction)
print("Log Loss: {}".format(nu_ll))
rf_clf = RandomForestClassifier(n_estimators=1000, random_state=0)
rf_clf.fit(x_train, y_train)
rf_train_prediction = rf_clf.predict(x_test)
rf_acc = accuracy_score(y_test, rf_train_prediction)
print("Accuracy: {:.4%}".format(rf_acc))

rf_train_prediction = rf_clf.predict_proba(x_test)
rf_ll = log_loss(y_test, rf_train_prediction)
print("Log Loss: {}".format(rf_ll))
# Predicting the actual set
nu_test_predict = rf_clf.predict(test_df)
test_predict = clf.predict(test_df)
acc = accuracy_score(test_predict, nu_test_predict)
print("Aggrement between two SVM linear and rbf models on prediction: {:.4%}".format(acc))
# Predicting probability of class for the actual test set
test_predict_prob = rf_clf.predict_proba(test_df)
# Format DataFrame
submission = pd.DataFrame(test_predict_prob, columns=classes)
submission.insert(0, 'id', test_id)
submission.reset_index()
print(submission.head())
submission.to_csv('prc_rf_submission.csv', index=False)

