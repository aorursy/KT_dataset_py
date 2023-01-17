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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import accuracy_score

import xgboost as xgb

from sklearn.model_selection import train_test_split

import optuna
df = pd.read_csv('/kaggle/input/input/crowdfunding.csv')
succes_rate = round(df['state'].value_counts() / len(df['state']) * 100,2)

print('before:',df.shape)

df = df[(df['state'] =='failed') | (df['state'] == 'successful')]

df['state'] = df['state'].map({'failed' : 0,'successful' : 1})

print('after:',df.shape)
df['deadline'] = pd.to_datetime(df['deadline'],format = '%Y-%m-%d %H:%M:%S')

df['launched'] = pd.to_datetime(df['launched'],format = '%Y-%m-%d %H:%M:%S')



df['dulation'] = (df['deadline'] - df['launched']).dt.days



df['quarter'] = df['launched'].dt.quarter

df['month'] = df['launched'].dt.month

df['year'] = df['launched'].dt.year

df['dayoweek'] = df['launched'].dt.dayofweek
df = df.drop(columns=['ID','deadline','goal','launched','pledged','backers','usd pledged','usd_pledged_real'])

df.head()
df['name_len'] = df['name'].str.len()

df['num_word'] = df['name'].apply(lambda x: len(str(x).split(' ')))

df.drop(columns=['name'],inplace =True)
df.isnull().sum()
df['name_len'] = df['name_len'].fillna(0)

df.isnull().sum()
df = pd.get_dummies(df,['category','main_category','currency','country'])

X =df.drop(columns='state')

y = df['state']



print(X.shape)

print(y.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

print(X_train.shape,y_train.shape)

print(X_test.shape,y_test.shape)



#X_trainval, X_test, y_trainval, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)

#print('Size of trainings set: {}, validation set: {}, test set: {}'.format(X_train.shape, X_valid.shape, X_test.shape))
def Objective(trial):

   

    params = {

        'max_depth': trial.suggest_int('max_depth',3,10),

        'tree_method':'auto',

        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 1e3),

        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 1e3),

        'min_child_weight': trial.suggest_int('min_child_weight',1,20),

        'gamma' : trial.suggest_discrete_uniform('gamma',0.1,1,0.1),

        'eta' : trial.suggest_loguniform('eta', 0.001,0.3),

        'subsample': trial.suggest_discrete_uniform('subsample', 0.1,0.5,0.05),

        'objective':'binary:logistic',

        'eval_metric':'logloss',

        'predictor':'cpu_predictor'

    }

    

    dtrain = xgb.DMatrix(X_train,label=y_train)

    dtest = xgb.DMatrix(X_test,label=y_test)



    xgb_model = xgb.train(

                 params = params,

                 dtrain=dtrain,

                 num_boost_round=1000,

                 early_stopping_rounds = 10,

                 evals=[(dtest,'test')])

 



    y_pred = xgb_model.predict(xgb.DMatrix(X_test),ntree_limit= xgb_model.best_ntree_limit)

    predictions = [round(pred) for pred in y_pred]



    score_metric = accuracy_score(y_test,predictions)



    

    return score_metric
study = optuna.create_study(direction='maximize')

study.optimize(Objective,n_trials = 10)
print(study.best_value)

print(study.best_params)

print(study.trials[0].datetime_start)

print(study.trials[9].datetime_complete)
params = { 

    'max_depth' : study.best_params['max_depth'],

    'tree_method' : 'auto',

    'reg_lambda': study.best_params['reg_lambda'],

    'reg_alpha': study.best_params['reg_alpha'],

    'min_child_weight' : study.best_params['min_child_weight'],

    'gamma':study.best_params['gamma'],

    'eta' : study.best_params['eta'],

    'subsample':study.best_params['subsample'],

    'objective': 'binary:logistic',

    'eval_metric':'logloss',

    'predictor':'cpu_predictor',

}



dtrain = xgb.DMatrix(X_train,label=y_train)

dtest = xgb.DMatrix(X_test,label=y_test)



xgb_model_fn = xgb.train(params=params,

               dtrain=dtrain,

               num_boost_round=1000,

               early_stopping_rounds = 10,

               evals=[(dtest,'test')])



#y_pred = xgb_model_fn.predict(xgb.DMatrix(X_test),ntree_limit= xgb_model_fn.best_ntree_limit)

#predictions = [round(pred) for pred in y_pred]



#score_metric = accuracy_score(y_test,predictions)

#print(score_metric)
fig, ax = plt.subplots(figsize=(12,8))

xgb.plot_importance(xgb_model_fn,max_num_features=20,height=0.5,ax=ax)

plt.show()