# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
__author__ = 'HeartburntXiaoxue'
'''
this starter uses only phone_brand and device_model to predict both the gender and age
'''
import datetime
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import random
import zipfile
import time
import shutil
from sklearn.metrics import log_loss, precision_score
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

random.seed(2016)

def read_train_test():
    # read events
    print('read events...')
    events = pd.read_csv('../input/events.csv',
                         dtype={'event_id': np.int, 'device_id': np.str, 'timestamp':np.str,
                               'longitude': np.float, 'latitude': np.float})  # force the device_id to be a string
    events['counts'] = events.groupby(['device_id'])['event_id'].transform('count')  # number of events for each device
    
    # read brand and model
    print('read brands...')
    br = pd.read_csv('../input/phone_brand_device_model.csv', dtype={'device_id': np.str, 'phone_brand':np.str, 'device_model':np.str})
    br.drop_duplicates('device_id', inplace = True)
    le = LabelEncoder()
    br.phone_brand = le.fit_transform(br.phone_brand)
    br.device_model = le.fit_transform(br.device_model)
    
    # train
    print('read train...')
    train = pd.read_csv('../input/gender_age_train.csv', dtype={'device_id': np.str})
    train.group = le.fit_transform(train.group)
    # train = train.drop(['group'])  # will drop the group info, but keep gender and age
    train = train.replace({'gender': {'M': 0, 'F': 1}})
    train = pd.merge(train, br, how='left', on='device_id', left_index=True)

    # test
    print('read test...')
    test = pd.read_csv("../input/gender_age_test.csv", dtype={'device_id': np.str})
    test = pd.merge(test, br, how='left', on='device_id', left_index=True)
    test.drop_duplicates() # why are there duplicates after merging?
    
    # features
    features = list(test.columns.values)
    features.remove('device_id')

    return train, test, features, le
train,test,features, group_encoder = read_train_test()

def run_classifier(clf, train,test,features,target, random_state = 0):
    start_time = time.time()
    
    X_train, X_valid = train_test_split(train,test_size=0.3,random_state=random_state)
    #print('Length train:', len(X_train.index))
    #print('Length test:', len(X_valid.index))
    y_train = X_train[target]
    y_valid = X_valid[target]
    
    clf = clf.fit(X_train[features], y_train)
    
    #print('validating...')
    check = clf.predict_proba(X_valid[features])
    score = log_loss(y_valid.tolist(),check)
    
    #print('predict test set...')
    test_prediction = clf.predict_proba(test[features])
    
    #print('training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction, score
    
clfs = [LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier(),AdaBoostClassifier(),GaussianNB()]
names = ['logit','DecisionTree', 'RandomForest', 'AdaBoost', 'GaussianNaiveBayes']

for i, clf in enumerate(clfs):
    print(names[i])
    test_pred, score = run_classifier(clf,train,test,features,'group')
    print('score: {}'.format( score,2))
#clfs = [LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier(),AdaBoostClassifier(),GaussianNB()]
#names = ['logit','DecisionTree', 'RandomForest', 'AdaBoost', 'GaussianNaiveBayes']
from sklearn.grid_search import GridSearchCV
def run_classifier_tuned(clf, grid, train,test,features,target, random_state = 0):
    start_time = time.time()
    
    X_train, X_valid = train_test_split(train,test_size=0.3,random_state=random_state)
    #print('Length train:', len(X_train.index))
    #print('Length test:', len(X_valid.index))
    y_train = X_train[target]
    y_valid = X_valid[target]
    clf_grid = GridSearchCV(clf, grid, cv=5,
                       scoring='log_loss')
    
    
    clf_grid = clf_grid.fit(X_train[features], y_train)
    print("Best parameters set found on development set:")
    print(clf_grid.best_params_)
    
    #print('validating...')
    check = clf_grid.predict_proba(X_valid[features])
    score = log_loss(y_valid.tolist(),check)
    
    #print('predict test set...')
    test_prediction = clf_grid.predict_proba(test[features])
    
    #print('training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction, score
    
clf = AdaBoostClassifier()
names = 'RF'
param_grid = {"base_estimator" : [RandomForestClassifier(), DecisionTreeClassifier()]
             }
test_pred, score = run_classifier_tuned(clf,param_grid,train,test,features,'group')
print('score: {}'.format( score,2))
print('read events...')
events = pd.read_csv('../input/events.csv',
                     dtype={'event_id': np.int, 'device_id': np.str, 'timestamp':np.str,
                            'longitude': np.float, 'latitude': np.float})  # force the device_id to be a string
events['counts'] = events.groupby(['device_id'])['event_id'].transform('count')  # number of events for each device

events.head()

