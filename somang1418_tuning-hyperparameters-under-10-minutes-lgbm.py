#basic tools 

import os

import numpy as np

import pandas as pd

import warnings



#tuning hyperparameters

from bayes_opt import BayesianOptimization

from skopt  import BayesSearchCV 



#graph, plots

import matplotlib.pyplot as plt

import seaborn as sns



#building models

import lightgbm as lgb

import xgboost as xgb

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

import time

import sys



#metrics 

from sklearn.metrics import roc_auc_score, roc_curve

import shap

warnings.simplefilter(action='ignore', category=FutureWarning)
def reduce_mem_usage(df, verbose=True):

    numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
%%time

train= reduce_mem_usage(pd.read_csv("../input/train.csv"))

test= reduce_mem_usage(pd.read_csv("../input/test.csv"))

print("Shape of train set: ",train.shape)

print("Shape of test set: ",test.shape)
y=train['target']

X=train.drop(['ID_code','target'],axis=1)
%%time



def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, n_folds=3, random_seed=6,n_estimators=10000, output_process=False):

    # prepare data

    train_data = lgb.Dataset(data=X, label=y, free_raw_data=False)

    # parameters

    def lgb_eval(learning_rate,num_leaves, feature_fraction, bagging_fraction, max_depth, max_bin, min_data_in_leaf,min_sum_hessian_in_leaf,subsample):

        params = {'application':'binary', 'metric':'auc'}

        params['learning_rate'] = max(min(learning_rate, 1), 0)

        params["num_leaves"] = int(round(num_leaves))

        params['feature_fraction'] = max(min(feature_fraction, 1), 0)

        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)

        params['max_depth'] = int(round(max_depth))

        params['max_bin'] = int(round(max_depth))

        params['min_data_in_leaf'] = int(round(min_data_in_leaf))

        params['min_sum_hessian_in_leaf'] = min_sum_hessian_in_leaf

        params['subsample'] = max(min(subsample, 1), 0)

        

        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])

        return max(cv_result['auc-mean'])

     

    lgbBO = BayesianOptimization(lgb_eval, {'learning_rate': (0.01, 1.0),

                                            'num_leaves': (24, 80),

                                            'feature_fraction': (0.1, 0.9),

                                            'bagging_fraction': (0.8, 1),

                                            'max_depth': (5, 30),

                                            'max_bin':(20,90),

                                            'min_data_in_leaf': (20, 80),

                                            'min_sum_hessian_in_leaf':(0,100),

                                           'subsample': (0.01, 1.0)}, random_state=200)



    

    #n_iter: How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.

    #init_points: How many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.

    

    lgbBO.maximize(init_points=init_round, n_iter=opt_round)

    

    model_auc=[]

    for model in range(len( lgbBO.res)):

        model_auc.append(lgbBO.res[model]['target'])

    

    # return best parameters

    return lgbBO.res[pd.Series(model_auc).idxmax()]['target'],lgbBO.res[pd.Series(model_auc).idxmax()]['params']



opt_params = bayes_parameter_opt_lgb(X, y, init_round=5, opt_round=10, n_folds=3, random_seed=6,n_estimators=10000)
opt_params[1]["num_leaves"] = int(round(opt_params[1]["num_leaves"]))

opt_params[1]['max_depth'] = int(round(opt_params[1]['max_depth']))

opt_params[1]['min_data_in_leaf'] = int(round(opt_params[1]['min_data_in_leaf']))

opt_params[1]['max_bin'] = int(round(opt_params[1]['max_bin']))

opt_params[1]['objective']='binary'

opt_params[1]['metric']='auc'

opt_params[1]['is_unbalance']=True

opt_params[1]['boost_from_average']=False

opt_params=opt_params[1]

opt_params

%%time 



target=train['target']

features= [c for c in train.columns if c not in ['target','ID_code']]





folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=31416)

oof = np.zeros(len(train))

predictions = np.zeros(len(test))

feature_importance_df = pd.DataFrame()



for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):

    print("Fold {}".format(fold_))

    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx])

    val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx])



    num_round = 15000

    clf = lgb.train(opt_params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=500, early_stopping_rounds = 250)

    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = features

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits



print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:20].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(20,28))

sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))

plt.title('Features importance (averaged/folds)')

plt.tight_layout()

plt.savefig('Feature_Importance.png')
explainer = shap.TreeExplainer(clf)

shap_values = explainer.shap_values(X)



shap.summary_plot(shap_values, X)
#tree visualization

graph = lgb.create_tree_digraph(clf, tree_index=3, name='Tree3' )

graph.graph_attr.update(size="110,110")

graph