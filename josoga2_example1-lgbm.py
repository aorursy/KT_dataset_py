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
sample = pd.read_csv('/kaggle/input/mlbio1/sample_submission.csv')

test = pd.read_csv('/kaggle/input/mlbio1/test.csv')

train = pd.read_csv('/kaggle/input/mlbio1/train.csv')
sample
train.head()
train['smoking_status'].fillna('nan').value_counts()
mean_bmi = train['bmi'].mean()

train['bmi'] = train['bmi'].fillna(mean_bmi)

test['bmi'] = test['bmi'].fillna(mean_bmi)
test.head()
train['ever_married'].value_counts()
ever_married_dict = {'Yes': 1, 'No':0 }
train['ever_married'] =train['ever_married'].replace(ever_married_dict)

test['ever_married'] = test['ever_married'].replace(ever_married_dict)
train['smoking_status'].unique()
train.sample(5)
categorical_features = [c for c, col in enumerate(train.columns) if 'cat' in col]

feautures_list = ['age','avg_glucose_level', 'hypertension', 'bmi', 'ever_married']
import lightgbm as lg



train_data = train[feautures_list]

train_label = train['stroke']

test_data = test[feautures_list]



gbm_train = lg.Dataset(train_data, label = train_label, categorical_feature=categorical_features)

gbm_test = lg.Dataset(test, categorical_feature=categorical_features)

parameters = {

    'application': 'gbdt',

    'objective': 'regression',

    'metric': 'auc',

    'is_unbalance': 'true',

    'boosting': 'gbdt',

    'num_leaves': 31,

    'feature_fraction': 0.5,

    'bagging_fraction': 0.5,

    'bagging_freq': 20,

    'learning_rate': 0.0001,

    'verbose': 0

    

}



gbm_model = lg.train(parameters, gbm_train, num_boost_round=20,)
prediction = gbm_model.predict(test_data, num_iteration=gbm_model.best_iteration)
prediction
sample['stroke'] = prediction
sample.head(1)
sample.to_csv('wale1.csv', index = None )
pd.read_csv("wale.csv")