# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import warnings

# warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
train  = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')

print('Traning shape: ',train.shape)

train.head()
train['TARGET'].astype(int).plot.hist()

train['TARGET'].value_counts()
test  = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')

TEST = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')

print('Testing shape: ',test.shape)

test.head()
prev_app = pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv')

print('previous application shape: ',prev_app.shape)

prev_app.head()
prev_cat_features = [

    f for f in prev_app.columns if prev_app[f].dtype == 'object'

]

for f in prev_cat_features:

    prev_app[f], _ = pd.factorize(prev_app[f])

    

avg_prev = prev_app.groupby('SK_ID_CURR').mean()

cnt_prev = prev_app[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()

avg_prev['nb_app'] = cnt_prev['SK_ID_PREV']

del avg_prev['SK_ID_PREV']



train = train.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')

test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')

train.head()
train['NAME_CONTRACT_TYPE_x'].unique()
drop_feats = ['SK_ID_CURR','NAME_CONTRACT_TYPE_x','TARGET']

features = [i for i in train.columns if i not in drop_feats]
train = train[features+['TARGET']]

test = test[features]
train = train.fillna(0)

test= test.fillna(0)
train.dtypes.sample(10)
x_train = train.loc[:, train.columns != 'TARGET']

y_train = train['TARGET']
one_hot_encoded_training_predictors = pd.get_dummies(x_train)

one_hot_encoded_test_predictors = pd.get_dummies(test)

one_hot_encoded_training_predictors.shape

x_train, x_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors, join='left', axis=1)

one_hot_encoded_training_predictors
x_train.shape
x_train = x_train.fillna(0)

x_test= x_test.fillna(0)
model_params = {

            'n_jobs': 16,

            'n_estimators': 1000,

            'max_features': 0.5,

            'max_depth': 20,

            'min_samples_leaf': 100,

            'verbose' : 1,

            'max_samples' : 0.5,

            'oob_score' : True,

                }

model = RandomForestClassifier(**model_params)
model.fit(x_train,y_train) 
TEST['TARGET'] = model.predict_proba(x_test)[:,1]

TEST[['SK_ID_CURR', 'TARGET']].to_csv('submission.csv', index=False, float_format='%.8f')
test.shape

# test.head()