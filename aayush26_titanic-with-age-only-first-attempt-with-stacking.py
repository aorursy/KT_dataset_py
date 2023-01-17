# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn



from sklearn.model_selection import KFold



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head(3)
age = train['Age']

age.head()
age.fillna(age.mean(), inplace=True)

age.describe
ntrain = age.shape[0]

SEED = 0 # for reproducibility

NFOLDS = 2 # set folds for out-of-fold prediction

kf = KFold(n_splits=NFOLDS, random_state=SEED)
test_age = test['Age']

test_age.fillna(test_age.mean(), inplace=True)

ntest = test_age.shape[0]

print(ntest)
# Copied from https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python

def get_oof(clf, x_train, y_train, x_test):

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))



    for i, (train_index, test_index) in enumerate(kf.split(x_train,y_train)):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]

#         print(str(train_index) + " : " + str(test_index) + " : "+str(i))

        clf.fit(x_tr, y_tr)



        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i, :] = clf.predict(x_test)



    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
# Support Vector Classifier parameters 

svc_params = {

    'kernel' : 'linear',

    'C' : 0.025,

    'random_state': SEED

    }



# AdaBoost parameters

ada_params = {

    'n_estimators': 500,

    'learning_rate' : 0.75,

    'random_state': SEED

}



# Random Forest parameters

rf_params = {

    'n_jobs': -1,

    'n_estimators': 500,

     'warm_start': True, 

     #'max_features': 0.2,

    'max_depth': 6,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

    'verbose': 0

}
# Going to use these 2 base models for the stacking

from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier)

from sklearn.svm import SVC
# svc = SVC(**svc_params)

ada = AdaBoostClassifier(**ada_params)

rf = RandomForestClassifier(**rf_params)
y_train = train['Survived'].ravel()

print(y_train)
# Class to extend the Sklearn classifier

class SklearnHelper(object):

    def __init__(self, clf, seed=0, params=None):

        params['random_state'] = seed

        self.clf = clf(**params)



    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        return self.clf.predict(x)

    

    def fit(self,x,y):

        return self.clf.fit(x,y)

    

    def feature_importances(self,x,y):

        print(self.clf.fit(x,y).feature_importances_)
# Create our OOF train and test predictions. These base results will be used as new features

x_train = age.values

x_test = test_age.values



x_train = x_train.reshape(-1, 1)

x_test = x_test.reshape(-1, 1)





ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 

rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest

# svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier



print("Training is complete")
print(rf.fit(x_train,y_train).feature_importances_)

# rf_feature = rf.feature_importances(x_train,y_train)

print(ada.fit(x_train,y_train).feature_importances_)

# svc_feature = svc.feature_importances(x_train, y_train)
rf_features = rf.fit(x_train,y_train).feature_importances_

ada_features = ada.fit(x_train,y_train).feature_importances_

feature_dataframe = pd.DataFrame( {

     'Random Forest feature importances': rf_features,

      'AdaBoost feature importances': ada_features,

    })
print(feature_dataframe)
x_train = np.concatenate(( rf_oof_train, ada_oof_train), axis=1)

x_test = np.concatenate(( rf_oof_test, ada_oof_test), axis=1)
import xgboost as xgb



gbm = xgb.XGBClassifier(

    #learning_rate = 0.02,

 n_estimators= 2000,

 max_depth= 4,

 min_child_weight= 2,

 #gamma=1,

 gamma=0.9,                        

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread= -1,

 scale_pos_weight=1).fit(x_train, y_train)

predictions = gbm.predict(x_test)
print(predictions)
StackingSubmission = pd.DataFrame({ 'PassengerId': test['PassengerId'],

                            'Survived': predictions })

StackingSubmission.to_csv("StackingSubmission.csv", index=False)