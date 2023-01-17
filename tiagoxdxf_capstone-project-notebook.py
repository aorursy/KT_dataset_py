import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import missingno as msno
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier


def train_model_lgb(application_train,train_target_label): 
    #create validation data
    x_train, x_val, y_train, y_val = train_test_split(application_train, train_target_label, test_size=0.2, random_state=18)
    #create lgb data
    lgb_train = lgb.Dataset(data=x_train, label=y_train)
    lgb_eval = lgb.Dataset(data=x_val, label=y_val)

    #fit the model
    params = {'task': 'train', 'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc', 
          'learning_rate': 0.01, 'num_leaves': 48, 'num_iteration': 5000, 'verbose': 0 ,
          'colsample_bytree':.8, 'subsample':.9, 'max_depth':7, 'reg_alpha':.1, 'reg_lambda':.1, 
          'min_split_gain':.01, 'min_child_weight':1}
    model = lgb.train(params, lgb_train, valid_sets=lgb_eval, early_stopping_rounds=150, verbose_eval=200)
    lgb.plot_importance(model, figsize=(12, 25), max_num_features=100);
    preds = model.predict(application_test)
    export_lgb = pd.DataFrame()
    export_lgb['SK_ID_CURR'] = application_test['SK_ID_CURR']
    export_lgb['TARGET'] = preds
    export_lgb.to_csv("lgb_baseline.csv", index=False)
    export_lgb.head()
    return export_lgb

def train_model_xgboost(application_train,train_target_label): 
    x_train, x_val, y_train, y_val = train_test_split(application_train, train_target_label, test_size=0.2, random_state=18)
    
    params = { 'objective':'binary:logistic','booster':"gbtree", 'eval_metric':'auc', 'nthread': 4,
            'eta':0.05,'gamma':0,'max_depth':7, 'subsample':0.7, 'colsample_bytree':0.8,
            'colsample_bylevel':0.8,'min_child_weight':1,
            'alpha':0,'random_state':42,'nrounds':2000,'n_estimators':2000 }
    #params = {'max_depth':2, 'eta':1, 'silent':1, 'objective': 'binary:logistic'}
    model = xgb.XGBClassifier(**params)
    model.fit(x_train, y_train, eval_set=[(x_val,y_val)], verbose=10, early_stopping_rounds = 50)
    preds = model.predict_proba(application_test)
    print('preds shape', str(preds.shape))
    export_xgboost = pd.DataFrame()
    export_xgboost['SK_ID_CURR'] = application_test['SK_ID_CURR']
    export_xgboost['TARGET'] = preds[:,1]
    export_xgboost.to_csv("xboost_baseline.csv", index=False)
    export_xgboost.head()
    fig, ax = plt.subplots(figsize=(14, 25))
    xgb.plot_importance(model,ax = ax, max_num_features=100)
    return export_xgboost


print(os.listdir("../input"))

application_train = pd.read_csv('../input/application_train.csv')
application_test = pd.read_csv('../input/application_test.csv')


# Any results you write to the current directory are saved as output.

print('Size of application_train data', application_train.shape)
print('Size of application_test data', application_test.shape)


# One hot encodding categorical data
application_train = pd.get_dummies(application_train)
application_test = pd.get_dummies(application_test)
#equal train set columns to test set columns
train_target_label = application_train['TARGET']
application_train, application_test = application_train.align(application_test, join = 'inner', axis = 1 )
train_model_lgb(application_train,train_target_label)
train_model_xgboost(application_train,train_target_label)

