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

from sklearn import tree, ensemble, metrics, linear_model, preprocessing, model_selection, feature_selection

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import cluster, preprocessing, linear_model, tree, model_selection, feature_selection

from sklearn import base, ensemble, decomposition, metrics, pipeline, datasets, impute

from skopt import gp_minimize, space, gbrt_minimize, dummy_minimize, forest_minimize

from functools import partial

import os

import lightgbm as lgb

import xgboost as xgb

import catboost as cb

from sklearn import ensemble, preprocessing, tree, model_selection, feature_selection, pipeline, metrics, svm

from imblearn import under_sampling, over_sampling, combine

from imblearn import pipeline as imb_pipeline

from imblearn import ensemble as imb_ensemble

from sklearn.model_selection import StratifiedKFold

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Create class which performs Label Encoding - if required

class categorical_encoder:

    def __init__(self, columns, kind = 'label', fill = True):

        self.kind = kind

        self.columns = columns

        self.fill = fill

        

    def fit(self, X):

        self.dict = {}

        self.fill_value = {}

        

        for col in self.columns:

            label = preprocessing.LabelEncoder().fit(X[col])

            self.dict[col] = label

            

            # To fill

            if self.fill:

                self.fill_value[col] = X[col].mode()[0]

                X[col] = X[col].fillna(self.fill_value[col])

                

        print('Label Encoding Done for {} columns'.format(len(self.columns)))

        return self

    def transform(self, X):

        for col in self.columns:

            if self.fill:

                X[col] = X[col].fillna(self.fill_value[col])

                

            X.loc[:, col] = self.dict[col].transform(X[col])

        print('Transformation Done')

        return X



def missing(df):

    print(df.isna().sum().sort_values(ascending = False)*100/df.shape[0])
train = pd.read_csv(r'/kaggle/input/janatahack-healthcare-analytics-part-2/train.csv')

test = pd.read_csv(r'/kaggle/input/janatahack-healthcare-analytics-part-2/test.csv')
train.head(2)
train['Type of Admission'].value_counts()
def preprocess_data(data):

    ### Create New Features

    # For Age

    age_dict = {'0-10': 5,

               '11-20': 15,

               '21-30': 25,

               '31-40': 35,

               '41-50': 45,

               '51-60': 55,

               '61-70': 65,

               '71-80': 75,

               '81-90': 85,

               '91-100': 95}

    data['Age'] = data['Age'].replace(age_dict)

    

    # Severity of Illness

    sev_dict = {'Minor': 0,

               'Moderate': 1,

               'Extreme': 2}

    data['Severity of Illness'] = data['Severity of Illness'].replace(sev_dict)

    

    # Type of Admission

    ad_dict = {'Trauma': 0,

              'Emergency': 1,

              'Urgent': 2}

    data['Type of Admission'] = data['Type of Admission'].replace(ad_dict)

    return data
'''cat_text = ['Hospital_type_code', 'Hospital_region_code', 'Department', 'Ward_Type', 'Ward_Facility_Code',

           'Type of Admission', 'Severity of Illness', 'Age']'''

cat_text = ['Hospital_type_code', 'Hospital_region_code', 'Department', 'Ward_Type', 'Ward_Facility_Code']

# Ideas - Age is ordinal - encode it as such
# Indicators

train['which'] = 1

test['which'] = 0



# Merge

data = pd.concat([train, test], axis = 0, ignore_index = True)



# Operations

data = data.fillna(data.median())

data = preprocess_data(data)

encoder = categorical_encoder(columns = cat_text, fill = False).fit(data)

data = encoder.transform(data)



# Split Back

train = data.loc[data['which'] == 1, :].drop('which', axis = 1)

test = data.loc[data['which'] == 0, :].drop('which', axis = 1)
train.columns
X_cols = ['case_id', 'Hospital_code', 'Hospital_type_code', 'City_Code_Hospital',

       'Hospital_region_code', 'Available Extra Rooms in Hospital',

       'Department', 'Ward_Type', 'Ward_Facility_Code', 'Bed Grade',

       'patientid', 'City_Code_Patient', 'Type of Admission',

       'Severity of Illness', 'Visitors with Patient', 'Age',

       'Admission_Deposit']

X_cols = ['Hospital_code', 'Hospital_type_code', 'City_Code_Hospital',

       'Hospital_region_code', 'Available Extra Rooms in Hospital',

       'Department', 'Ward_Type', 'Ward_Facility_Code', 'Bed Grade',

       'patientid', 'City_Code_Patient', 'Type of Admission',

       'Severity of Illness', 'Visitors with Patient', 'Age',

       'Admission_Deposit']







X_train = train[X_cols]

y_train = train['Stay']



X_test = test.drop('Stay', axis = 1)[X_cols]
#model = ensemble.RandomForestClassifier(n_estimators = 550, max_depth = 15, n_jobs = -1, max_features = .7)

model = lgb.LGBMClassifier(n_estimators = 1000, max_depth = 6, learning_rate = .1)

model.fit(X_train, y_train)
sub = pd.DataFrame()

sub['case_id'] = test['case_id']

sub['Stay'] = model.predict(X_test)

sub.to_csv('SSUB.csv', index = None)
import pickle

import time

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials



def objective(x):

    #print(x)

    hyperparams = {'n_estimators': int(x['n_estimators']), 'max_depth': int(x['max_depth']),

                   'learning_rate': x['learning_rate'],'max_features': x['max_features'], 

                   'min_data_in_leaf': int(x['min_data_in_leaf']), 'class_weight': x['class_weight']}

    # split Data

    kf = model_selection.KFold(5)

    accuracy = []

    for train_index, test_index in kf.split(X_train, y_train):

        xtrain, xtest = X_train.values[train_index, :], X_train.values[test_index, :]

        ytrain, ytest = y_train.values[train_index], y_train.values[test_index]

        

        model = lgb.LGBMClassifier(**hyperparams).fit(xtrain, ytrain)

        

        accuracy.append(metrics.accuracy_score(ytest, model.predict(xtest)))

        

    return -np.mean(accuracy)



space = {'n_estimators': hp.quniform('n_estimators', 50, 1000, 10),

        'max_depth': hp.quniform('max_depth', 2, 10, 1),

        'learning_rate': hp.uniform('learning_rate', .01, .2),

        'max_features': hp.uniform('max_features', .1, .99),

        'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 50, 1),

        'class_weight': hp.choice('class_weight', ['balanced', None])}



trials = Trials()

results = fmin(objective,

    space=space,

    algo=tpe.suggest,

    max_evals=10,

    trials=trials)



print(results)
hyperparams = {'n_estimators': int(results['n_estimators']), 'max_depth': int(results['max_depth']),

                   'learning_rate': results['learning_rate'],'max_features': results['max_features'], 

                   'min_data_in_leaf': int(results['min_data_in_leaf']), 'class_weight': None}



model = lgb.LGBMClassifier(**hyperparams).fit(X_train, y_train)
sub = pd.DataFrame()

sub['case_id'] = test['case_id']

sub['Stay'] = model.predict(X_test)

sub.to_csv('SSUB.csv', index = None)