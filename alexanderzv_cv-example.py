# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import linear_model

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings('ignore')
pd.options.display.max_columns = 100
#FOR Kaggle

sample = pd.read_csv('/kaggle/input/mlbio1/sample_submission.csv')

test = pd.read_csv('/kaggle/input/mlbio1/test.csv')

train = pd.read_csv('/kaggle/input/mlbio1/train.csv')



#For local

# sample = pd.read_csv('healthcare-dataset-stroke-data/sample_submission.csv')

# test = pd.read_csv('healthcare-dataset-stroke-data/test.csv')

# train = pd.read_csv('healthcare-dataset-stroke-data/train.csv')
train.head()
#counts of nan

for i in train.columns:

    print(i, (pd.isnull(train[i])).sum() )
train['bmi'].head()   #bmi is numeric column
# fill with mean

mean_bmi = train['bmi'].mean()

train['bmi'] = train['bmi'].fillna(mean_bmi)

test['bmi'] = test['bmi'].fillna(mean_bmi)
# fill with median

# median_bmi = train['bmi'].median()

# train['bmi'] = train['bmi'].fillna(median_bmi)

# test['bmi'] = test['bmi'].fillna(median_bmi)
# fill with zero



# train['bmi'] = train['bmi'].fillna(0)

# test['bmi'] = test['bmi'].fillna(0)
train['smoking_status'].head()   #smoking_status is categorical column
train['smoking_status'] = train['smoking_status'].fillna('nan')

test['smoking_status'] = test['smoking_status'].fillna('nan')
train['work_type'].value_counts()
for i in train['work_type'].unique():

    print(i)

    train['work_type_is_{}'.format(i)] = (train['work_type'] == i)*1

    test['work_type_is_{}'.format(i)] = (test['work_type'] == i)*1
train.head()
(train.groupby(['smoking_status'])['stroke'].agg(['mean'])).to_dict()
smoking_status_target_enc_dict = (train.groupby(['smoking_status'])['stroke'].agg(['mean'])).to_dict()['mean']
smoking_status_target_enc_dict
train['smoking_status_target_enc'] = train['smoking_status'].replace(smoking_status_target_enc_dict)

test['smoking_status_target_enc'] = test['smoking_status'].replace(smoking_status_target_enc_dict)
# Binary variables 

train['ever_married'].value_counts() 
train['ever_married'] = train['ever_married'].replace({'Yes':1, 'No':0 })

test['ever_married'] = test['ever_married'].replace({'Yes':1, 'No':0 })
train.head()
train['Residence_type'].value_counts()
train['Residence_type'] = train['Residence_type'].replace({'Urban':1, 'Rural':0 })

test['Residence_type'] = test['Residence_type'].replace({'Urban':1, 'Rural':0 })
# smoking_status binary encoding

for i in train['smoking_status'].unique():

    print(i)

    train['smoking_status_is_{}'.format(i)] = (train['smoking_status'] == i)*1

    test['smoking_status_is_{}'.format(i)] = (test['smoking_status'] == i)*1
train.head()
train['gender'].value_counts()
(train.groupby(['gender'])['stroke'].agg(['mean']))
gender_target_enc_dict = (train.groupby(['gender'])['stroke'].agg(['mean'])).to_dict()['mean']



train['gender_target_enc'] = train['gender'].replace(gender_target_enc_dict)

test['gender_target_enc'] = test['gender'].replace(gender_target_enc_dict)
train.head()
#from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
# I select 7 features
features = [  'age',

            'hypertension' , 'heart_disease' , 'ever_married' ,

            'avg_glucose_level' ,'bmi']


clf = linear_model.SGDClassifier(max_iter=1000,  loss='log', penalty = 'elasticnet')

#DIY



def my_cross_validation_for_roc_auc( clf, X, y ,cv=5):

    X = np.array(X.copy())

    y = np.array(y.copy())

    kf = KFold(n_splits=cv)

    kf.get_n_splits(X)

    scores = []

    for train_index, test_index in kf.split(X):

        #print("TRAIN:", train_index, "TEST:", test_index)

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        

        clf.fit(X_train, y_train)

        prediction_on_this_fold = clf.predict_proba(X_test)[:,1]

        

        score = roc_auc_score(y_score=prediction_on_this_fold, y_true=y_test)

        scores.append(score)

        

    return scores

        

scores = my_cross_validation_for_roc_auc(clf, train[features] , train['stroke'])

scores
# mean score on train dataset

np.mean(scores)
train.head()
all_features = [ 'age', 'hypertension', 'heart_disease', 'ever_married',

        'Residence_type', 'avg_glucose_level', 'bmi',

         'work_type_is_children',

       'work_type_is_Private', 'work_type_is_Never_worked',

       'work_type_is_Self-employed', 'work_type_is_Govt_job',

       'smoking_status_target_enc', 'smoking_status_is_nan',

       'smoking_status_is_never smoked', 'smoking_status_is_formerly smoked',

       'smoking_status_is_smokes', 'gender_target_enc']
features_scores = {}

for f in all_features:

    scores = my_cross_validation_for_roc_auc(clf, train[[f]] , train['stroke'])

    print(f,  np.mean(scores))

    features_scores[f] = np.mean(scores)
features_2=['age', 'hypertension', 'heart_disease', 'ever_married', 'avg_glucose_level', 'bmi', 

             'work_type_is_children', 'work_type_is_Self-employed', 'smoking_status_target_enc']
scores = my_cross_validation_for_roc_auc(clf, train[features_2] , train['stroke'])

scores
np.mean(scores)


clf = linear_model.SGDClassifier(max_iter=1000,  loss='log', penalty = 'elasticnet')



clf.fit(train[features_2], train['stroke'])
from lightgbm import LGBMClassifier
lgb =   LGBMClassifier(n_estimators=100, max_depth=5)
scores = my_cross_validation_for_roc_auc(lgb, train[features_2] , train['stroke'])
scores
np.mean(scores)