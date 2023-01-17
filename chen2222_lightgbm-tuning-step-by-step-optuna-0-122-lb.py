import numpy as np

import os

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

import optuna

from sklearn import linear_model

from sklearn import metrics

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from sklearn import preprocessing

from sklearn.model_selection import cross_val_score

from optuna.samplers import RandomSampler, GridSampler, TPESampler

import sklearn

import xgboost as xgb

from scipy.misc import derivative

from sklearn.metrics import mean_squared_error

import pickle

import category_encoders as ce 

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

import random

import lightgbm as lgb

import random
def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
seed_everything(0)
df_train =  pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

df_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

y_train = np.log(df_train['SalePrice'])

X_train = df_train.drop(['SalePrice','Id'],axis=1)

X_test = df_test.drop('Id',axis=1)

print('Train data: ',X_train.shape)

print('Test data: ',X_test.shape)
cat_nominal_features = pickle.load(open('../input/features-housing/cat_nominal_features.p', "rb" ))

cat_ordinal_features = pickle.load(open('../input/features-housing/cat_ordinal_features.p', "rb" ))

num_features = pickle.load(open('../input/features-housing/num_features.p', "rb" ))

cat_features = cat_nominal_features + cat_ordinal_features

print('Number of numeric features: ',len(num_features))

print('Number of ordinal features: ',len(cat_ordinal_features))

print('Number of nominal featuures: ',len(cat_nominal_features))
X_train[cat_ordinal_features] = X_train[cat_ordinal_features].fillna(np.nan)

X_train[cat_nominal_features] = X_train[cat_nominal_features].fillna(np.nan)

X_test[cat_ordinal_features] = X_test[cat_ordinal_features].fillna(np.nan)

X_test[cat_nominal_features] = X_test[cat_nominal_features].fillna(np.nan)
ord_mapping=[{'col': 'Street', 'mapping': {'Grvl': 1, 'Pave': 2}},

        {'col': 'Alley', 'mapping': {np.nan:0,'Grvl': 1, 'Pave': 2}}, 

        {'col': 'Utilities', 'mapping': {'NoSeWa': 1, 'AllPub':2}},

        {'col': 'ExterQual', 'mapping': {'Po':1,'Fa':2,'TA':3,

                                        'Gd':4,'Ex':5}},

        {'col': 'ExterCond', 'mapping': {'Po':1,'Fa':2,'TA':3,

                                        'Gd':4,'Ex':5}},

        {'col': 'BsmtCond', 'mapping': {np.nan:0,'Po':1,'Fa':2,'TA':3,

                                        'Gd':4,'Ex':5}},

        {'col': 'BsmtQual', 'mapping': {np.nan:0,'Po':1,'Fa':2,'TA':3,

                                        'Gd':4,'Ex':5}},

        {'col': 'BsmtExposure', 'mapping': {np.nan:0,'No':1,'Mn':2,'Av':3,

                                        'Gd':4}},

        {'col': 'BsmtFinType1', 'mapping': {np.nan:0,'Unf':1,'LwQ':2,'Rec':3,

                                        'BLQ':4,'ALQ':5,'GLQ':6}},

        {'col': 'BsmtFinType2', 'mapping': {np.nan:0,'Unf':1,'LwQ':2,'Rec':3,

                                        'BLQ':4,'ALQ':5,'GLQ':6}},     

        {'col': 'HeatingQC', 'mapping': {'Po':1,'Fa':2,'TA':3,

                                        'Gd':4,'Ex':5}},

        {'col': 'CentralAir', 'mapping': {'Y':1,'N':0}},

        {'col': 'KitchenQual', 'mapping': {'Po':1,'Fa':2,'TA':3,

                                        'Gd':4,'Ex':5}}, 

        {'col': 'Functional', 'mapping': {'Typ':8,'Min1':7,'Min2':6,

                                        'Mod':5,'Maj1':4,'Maj2':3,

                                         'Sev':2,"Sal":1}},                                    

        {'col': 'FireplaceQu', 'mapping': {np.nan:0,'Po':1,'Fa':2,'TA':3,

                                        'Gd':4,'Ex':5}},

        {'col': 'GarageFinish', 'mapping': {np.nan:0,'Unf':1,'RFn':2,'Fin':3}},

        {'col': 'GarageQual', 'mapping': {np.nan:0,'Po':1,'Fa':2,'TA':3,

                                        'Gd':4,'Ex':5}},

        {'col': 'GarageCond', 'mapping': {np.nan:0,'Po':1,'Fa':2,'TA':3,

                                        'Gd':4,'Ex':5}},

        {'col': 'PavedDrive', 'mapping': {'N':1,'P':2,'Y':3}},

        {'col': 'PoolQC', 'mapping': {np.nan:0,'Fa':1,'TA':2,

                                        'Gd':3,'Ex':4}},

        {'col': 'Fence', 'mapping': {np.nan:0,'MnWw':1,'GdWo':2,'MnPrv':3,

                                        'GdPrv':4}}]

mapping_cols = ['Street','Alley','Utilities','ExterQual','ExterCond','BsmtCond','BsmtQual','BsmtExposure',

               'BsmtFinType1','BsmtFinType2','HeatingQC','CentralAir','KitchenQual',

               'Functional','FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive',

               'PoolQC','Fence']



def construct_ord_nom(ordinal_features,nominal_features):

    mapp = []

    ord_features = []

    for c in ordinal_features:

        if c not in mapping_cols:

            continue

        idx = mapping_cols.index(c)

        mapp.append(ord_mapping[idx])

        ord_features.append(c)

    ce_ord = ce.OrdinalEncoder(cols=ord_features,mapping=mapp,

                               handle_unknown='return_nan',handle_missing='return_nan')

    ce_nom = ce.OneHotEncoder(cols=nominal_features,handle_unknown='return_nan',handle_missing='return_nan')

    return ce_ord,ce_nom 



def get_CT(ord_features,nom_features,num_features_new,ce_ord,ce_nom):

    numeric_transformer = Pipeline(steps=[

            ('imputer', SimpleImputer(strategy='constant',fill_value=-1)),

            ])

    ce_ord, ce_nom = construct_ord_nom(ord_features,nom_features)

    ct1 = ColumnTransformer(

            transformers=[

                ('nominal',ce_nom,nom_features),

                ('ordinal',ce_ord,ord_features),

                ('num',numeric_transformer,num_features_new)

                ],remainder = 'passthrough')

    return ct1
ce_ord, ce_nom = construct_ord_nom(cat_ordinal_features,cat_nominal_features)

CT = get_CT(cat_ordinal_features,cat_nominal_features,num_features,ce_ord,ce_nom)
X_train_new = CT.fit_transform(X_train)

X_test_new = CT.transform(X_test)

print(X_train_new.shape)

print(X_test_new.shape)

dtrain = lgb.Dataset(X_train_new,y_train)
class cparams():

    def __init__(self): 

        self.seed = 0

        self.num_iterations = 100 # Default = 100

        self.learning_rate = 0.1 # Default = 0.1

        self.num_leaves = 31 #Default = 31

        self.min_child_samples = 20 #Default = 20

        self.min_child_weight = 0.001 #Default = 0.001

        self.bagging_fraction = 1.0

        self.feature_fraction = 1.0

        self.bagging_freq = 0

        self.alpha = 0.25

        self.gamma = 2.0

        self.l1 = 0.0

        self.l2 = 0.0

        

    def calibrate(self,num_round):

        

        param = {'boosting_type': 'gbdt', 

                'objective': 'regression',

                'metric': 'rmse', 

                'learning_rate': self.learning_rate, 

                'num_leaves': self.num_leaves,     

                'min_data_in_leaf': self.min_child_samples,   

                'min_sum_hessian_in_leaf':self.min_child_weight, 

                'bagging_fraction': self.bagging_fraction, 

                'bagging_freq': self.bagging_freq,

                'feature_fraction': self.feature_fraction, 

                'lambda_l1': self.l1,

                'lambda_l2': self.l2,

                'seed': self.seed

        }

        

        # cv's seed used to generate folds passed to numpy.random.seed

        bst = lgb.cv(param, dtrain,num_boost_round=num_round, stratified=False, \

                     shuffle=True,early_stopping_rounds=100,verbose_eval=10,seed=0)

        return bst

    

    def get_param(self):

        

        param = {'boosting_type': 'gbdt', 

                 'objective': 'regression',

                'metric': 'rmse', 

                'learning_rate': self.learning_rate, 

                'num_leaves': self.num_leaves,     

                'min_data_in_leaf': self.min_child_samples,   

                'min_sum_hessian_in_leaf':self.min_child_weight, 

                'bagging_fraction': self.bagging_fraction, 

                'bagging_freq': self.bagging_freq,

                'feature_fraction': self.feature_fraction, 

                'lambda_l1': self.l1,

                'lambda_l2': self.l2,

                'seed': self.seed

        }        

        

        return param
current_model = cparams()
# Use default parameters

# num_iterations(num_boost_round) = 100

# max_depth = -1

# num_leaves = 31

# min_data_in_leaf(min_child_samples) = 20

# min_sum_hessian_in_leaf(min_child_weight) = 0.001

# feature_fraction(colsample_bytree) = 1.0

# bagging_fraction(subsample) = 1.0

# bagging_freq(subsample_freq) = 0

# learning_rate = 0.1



params = {

    'objective': 'regression',

    'metric': 'rmse', 

    "verbosity": 1,

    "boosting_type": "gbdt",

    'seed':0

}



eval_history = lgb.cv(

    params, dtrain, verbose_eval=20,

    stratified=False, num_boost_round=1000, early_stopping_rounds=100,

    nfold=5,seed=0)
print('Best score: ', eval_history['rmse-mean'][-1])

print('Number of estimators: ', len(eval_history['rmse-mean']))

current_model.num_iterations = len(eval_history['rmse-mean'])
study_name2 = 'lgb_leaves'

study_leaves = optuna.create_study(study_name=study_name2,direction='maximize',sampler=TPESampler(0))
def opt_leaves(trial):

    

    params = {

        'objective': 'regression',

        'metric': 'rmse', 

        "verbosity": 1,

        "boosting_type": "gbdt",

        'seed':0,

        'num_leaves':int(trial.suggest_loguniform("num_leaves", 3,32))

    }

    

    score = lgb.cv(

        params, dtrain, verbose_eval=0, 

        stratified=False, num_boost_round=current_model.num_iterations,

        nfold=5,seed=0)

    return -score['rmse-mean'][-1]
study_leaves.optimize(opt_leaves, n_trials=50)
print('Total number of trials: ',len(study_leaves.trials))

trial_leaves = study_leaves.best_trial

print('Best score : {}'.format(-trial_leaves.value))

for key, value in trial_leaves.params.items():

    print("    {}: {}".format(key, value))
current_model.num_leaves = int(trial_leaves.params['num_leaves'])
study_name3 = 'lgb_child_weight_sample'

study_sample_weight = optuna.create_study(study_name=study_name3,direction='maximize',sampler=TPESampler(0))
def opt_sample_weight(trial):

    

    params = {

        'objective': 'regression',

        'metric': 'rmse', 

        "verbosity": 1,

        "boosting_type": "gbdt",

        'seed':0,

        'num_leaves':current_model.num_leaves,

        'min_data_in_leaf':int(trial.suggest_discrete_uniform('data_in_leaf',4,32,q=2)),

        'min_sum_hessian_in_leaf':trial.suggest_discrete_uniform('min_hessian',0.001,0.003,q=0.0005)

    }

    

    score = lgb.cv(

        params, dtrain, verbose_eval=0, 

        stratified=False, num_boost_round=current_model.num_iterations,

        nfold=5,seed=0)

    return -score['rmse-mean'][-1]
study_sample_weight.optimize(opt_sample_weight, n_trials=50)
print('Total number of trials: ',len(study_sample_weight.trials))

trial_sample_weight = study_sample_weight.best_trial

print('Best score : {}'.format(-trial_sample_weight.value))

for key, value in trial_sample_weight.params.items():

    print("    {}: {}".format(key, value))
current_model.min_child_samples = int(trial_sample_weight.params['data_in_leaf'])

current_model.min_child_weight = trial_sample_weight.params['min_hessian']
study_name4 = 'lgb_bagging'

study_bagging = optuna.create_study(study_name=study_name4,direction='maximize',sampler=TPESampler(0))
def opt_bagging(trial):

    

    params = {

        'objective': 'regression',

        'metric': 'rmse',  

        "verbosity": 1,

        "boosting_type": "gbdt",

        'seed':0,

        'num_leaves':current_model.num_leaves,

        'min_data_in_leaf':current_model.min_child_samples,

        'min_sum_hessian_in_leaf':current_model.min_child_weight,

        'bagging_fraction': trial.suggest_discrete_uniform('bfrac',0.4,1.0,q=0.05),

        'bagging_freq': int(trial.suggest_discrete_uniform('bfreq',1,7,q=1.0)),

        'feature_fraction':trial.suggest_discrete_uniform('feature',0.4,1.0,q=0.05)

    }

    

    score = lgb.cv(

        params, dtrain, verbose_eval=0, 

        stratified=False, num_boost_round=current_model.num_iterations,

        nfold=5,seed=0)

    return -score['rmse-mean'][-1]
study_bagging.optimize(opt_bagging, n_trials=50)
print('Total number of trials: ',len(study_bagging.trials))

trial_bagging = study_bagging.best_trial

print('Best score : {}'.format(trial_bagging.value))

for key, value in trial_bagging.params.items():

    print("    {}: {}".format(key, value))
current_model.bagging_fraction = trial_bagging.params['bfrac']

current_model.bagging_freq = int(trial_bagging.params['bfreq'])

current_model.feature_fraction = trial_bagging.params['feature']
study_name5 = 'l1_l2'

study_reg = optuna.create_study(study_name=study_name5,direction='maximize',sampler=TPESampler(0))
def opt_reg(trial):

    

    params = {

        'objective': 'regression',

        'metric': 'rmse', 

        "verbosity": 1,

        "boosting_type": "gbdt",

        'seed':0,

        'num_leaves':current_model.num_leaves,

        'min_data_in_leaf':current_model.min_child_samples,

        'min_sum_hessian_in_leaf':current_model.min_child_weight,

        'bagging_fraction': current_model.bagging_fraction,

        'bagging_freq': current_model.bagging_freq,

        'feature_fraction':current_model.feature_fraction,

        'lambda_l1': trial.suggest_loguniform("lambda_l1", 1e-7, 10),

        'lambda_l2': trial.suggest_loguniform("lambda_l2", 1e-7, 10)

    }

    

    score = lgb.cv(

        params, dtrain, verbose_eval=0, 

        stratified=False, num_boost_round=current_model.num_iterations,

        nfold=5,seed=0)

    return -score['rmse-mean'][-1]
study_reg.optimize(opt_reg, n_trials=50)
print('Total number of trials: ',len(study_reg.trials))

trial_reg = study_reg.best_trial

print('Best score : {}'.format(trial_reg.value))

for key, value in trial_reg.params.items():

    print("    {}: {}".format(key, value))
current_model.l1 = trial_reg.params['lambda_l1']

current_model.l2 = trial_reg.params['lambda_l2']
current_model.learning_rate = 0.05

lr1 = current_model.calibrate(10000) 

print('Best score: ', lr1['rmse-mean'][-1])

print('Number of estimators: ', len(lr1['rmse-mean']))
current_model.learning_rate = 0.01

lr2 = current_model.calibrate(10000) 

print('Best score: ', lr2['rmse-mean'][-1])

print('Number of estimators: ', len(lr2['rmse-mean']))
current_model.learning_rate = 0.005

lr3 = current_model.calibrate(10000) 

print('Best score: ', lr3['rmse-mean'][-1])

print('Number of estimators: ', len(lr3['rmse-mean']))
## Get Current Parameters

current_model.learning_rate = 0.005 #Based on what we found above

current_param = current_model.get_param()

for key, value in current_param.items():

    print("    {}: {}".format(key, value))
from sklearn.model_selection import KFold

def cv_training(train_data,y_train_data):

    kFold = KFold(n_splits=5, random_state=0, shuffle=True)

    models = []

    eval_history = []

    oof_pred = []

    oof_target = []

    scores = []

    for fold, (trn_idx, val_idx) in enumerate(kFold.split(train_data)):

        #print(trn_idx)

        #print(val_idx)

        X_train = train_data[trn_idx]

        X_val = train_data[val_idx]

        y_train = y_train_data[trn_idx]

        y_val = y_train_data[val_idx]

        dtrain =  lgb.Dataset(X_train,y_train)

        dval =  lgb.Dataset(X_val,y_val)

        evals_result = {}

        model = lgb.train(current_param, dtrain,num_boost_round=5000,

                          evals_result=evals_result,valid_sets=dval,verbose_eval=20,early_stopping_rounds=100)

        models.append(model)

        y_pred = model.predict(X_val)

        score_temp = np.sqrt(mean_squared_error(y_pred,y_val))

        scores.append(score_temp)

        oof_pred.append(y_pred)

        oof_target.append(y_val)

        eval_history.append(evals_result)

    oof_pred = np.concatenate((oof_pred[0],oof_pred[1],oof_pred[2],

                               oof_pred[3],oof_pred[4]))

    oof_target = np.concatenate((oof_target[0],oof_target[1],oof_target[2],

                                 oof_target[3],oof_target[4]))

    oof_df = pd.DataFrame({'predictions':oof_pred,'target':oof_target})

    return models, eval_history, scores, oof_df
models, eval_history,scores,oof_df = cv_training(X_train_new,y_train)
oof_df.to_csv('oof_df.csv',index=False)

oof_df.head()
score = 0.0

for i in range(len(models)):

    print(models[i].best_iteration)

    print(eval_history[i]['valid_0']['rmse'][-1])

    score = score + eval_history[i]['valid_0']['rmse'][-1]

print('Average score: ',score/5)
# Number of boost_round will be based on what we found above

model_final = lgb.train(current_param, dtrain,num_boost_round=3920)

ypred = model_final.predict(X_test_new)
# Make a histo

plt.hist(np.exp(ypred))
sub = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

submission  = pd.DataFrame({

    'Id': sub['Id'],

    'SalePrice': np.exp(ypred)

})

submission.head()
submission.to_csv('submission.csv',index=False)