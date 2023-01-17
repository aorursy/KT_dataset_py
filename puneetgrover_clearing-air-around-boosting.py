#!pip -q install --upgrade --ignore-installed catboost

#!pip -q install --upgrade --ignore-installed lightgbm

#!pip -q install --upgrade --ignore-installed xgboost

#!pip -q install --upgrade --ignore-installed numpy, scipy
import catboost as cb

import xgboost as xb

import lightgbm as lgb
cb.__version__, xb.__version__, lgb.__version__

# CatBoost,   XGBoost,   LightGBM
import pandas as pd

import numpy as np

import sklearn
np.__version__, pd.__version__, sklearn.__version__
from sklearn.datasets import load_boston
dict_ = load_boston()
dict_.keys()
train_df = pd.DataFrame(dict_.data, index=np.arange(dict_.data.shape[0]), columns=dict_.feature_names)
train_df['target'] = dict_.target
train_df.head()
train_df.nunique()
from sklearn.ensemble import AdaBoostRegressor as AdaBoost

from sklearn.tree import DecisionTreeRegressor as dt
base_est = dt(criterion                = 'mse',  # 'mse', 'friedman_mse', 'mae'

              splitter                 = 'best', # 'best', 'random'

              max_depth                = 2,      #     -->                                                                             ## Can help reduce Overfitting. (by dec.) ##

              min_samples_split        = 10,     # int default = 2 (can overfit), can be float (0, 1.0].                               ## Can help reduce Overfitting. (by inc.) ##

              min_samples_leaf         = 5,      # int deafult = 1 (can overfit), can be float (0, 1.0].                               ## Can help reduce Overfitting. (by inc.) ##

              min_weight_fraction_leaf = 0.05,   # min fraction of total weight of all points that needs to be in a leaf node (float). ## Can help reduce Overfitting. (by dec.) ##

              max_features             = 'sqrt', # int, float (0, 1.0], 'auto', 'sqrt', 'log2', None.                                  ## Can help reduce Overfitting. (by dec.) ##

              random_state             = 2019,

              max_leaf_nodes           = 5,      # int (best leaf nodes are selected by best impurity values), None.                   ## Can help reduce Overfitting. (by dec.) ##

              min_impurity_decrease    = 0.05,   # min required impurity decrease because of a split.                                  ## Can help reduce Overfitting. (by inc.) ##

              presort                  = False)
ab = AdaBoost(base_estimator = base_est,

              n_estimators   = 300,

              learning_rate  = 0.1,

              loss           = 'linear',          # 'linear', 'square', 'exponential'

              random_state   = 2019)
ab = ab.fit(X = train_df.drop('target', axis=1), y = train_df['target'], sample_weight=np.ones(train_df.shape[0]))
print("Base Estimator")

print(ab.base_estimator_)



print("\nEstimator Erros:")

print(ab.estimator_errors_[0:10])



print("\nEstimator Weights:")

print(ab.estimator_weights_[0:10])



print("\nFeature Importances:")

print(ab.feature_importances_)
base_ests_temp = ab.estimators_

print(len(base_ests_temp))
dt_0 = base_ests_temp[0]



print("First Learner's Score (rmse):")

print(np.power(dt_0.score(X = train_df.drop('target', axis=1), y = train_df['target']), 1./2))



print("\nFirst Learner's Features' Importances")

dt_0.feature_importances_
reg_model = xb.XGBRegressor(

                    max_depth         =3,

                    learning_rate     =0.8,

                    silent            =True,

                    objective         ='reg:linear',

                    booster           ='gbtree',    # 'gbtree', 'gblinear', 'dart'

                    nthread           =2,           # Number of parallel 'threads'

                    #n_jobs           =,            # same as nthread 

                    #gamma             =0.01,       # Min Loss reduction required on partition

                    #min_child_weight  =0.05,       # Min. sum of hessian weights needed in a child

                    #max_delta_step    =0.001,      # Max delta step for each tree's weight estimation

                    subsample         =0.8,         # Subsample of data for each tree

                    colsample_bytree  =1.0,         # Subsample by column for each tree

                    colsample_bylevel =0.9,         # Subsample by column by level in each tree

                    #reg_alpha         =0.01,       # L1 regularization

                    #reg_lambda        =0.02,       # L2 regularization

                    #scale_pos_weight =,            # For Balancing of +ve and -ve weights  ## For Binary Classification. But it is present here also...  ##

                    #base_score       =,            # Initial prediction score of all instances, global bias. ## Helpful in Classification. Set it to mean of observations for imbalanced dataset. ##

                    seed              =2019,

                    #random_state     =,            # same as seed

                    missing           =0.0,         # None (np.nan), or some value to replace missing values.

)



# 'dart' is new booster algorithm in xgboost which uses Dropout for trees (randomly) to reduce overfitting.
from sklearn.model_selection import train_test_split as tts



Xtrain, Xvalid, ytrain, yvalid = tts(train_df.drop('target', axis=1), train_df['target'])
reg_model = reg_model.fit(X=Xtrain, y=ytrain,

              sample_weight=np.ones(Xtrain.shape[0]),

              eval_set=[(Xtrain, ytrain), (Xvalid, yvalid)],      # A array of set, of X and y's, as [(X1, y1), (X2, y2) ... ]

              eval_metric='rmse',                                 # Inbuilt eval metric or callable

              early_stopping_rounds=50,                           # For early stopping

              verbose=False,

              xgb_model=None                                      # Used with keep_training_model=True

             )
reg_model.evals_result_['validation_1']['rmse'][0:10]
reg_model.feature_importances_
train_data = xb.DMatrix(Xtrain, label=ytrain, feature_names=train_df.drop('target', axis=1).columns.values)

valid_data = xb.DMatrix(Xvalid, label=yvalid, feature_names=train_df.drop('target', axis=1).columns.values)
#

# Top problem of Boosting is that it can overfit very easily. So most parameters you will see here are to reduce overfitting.

# For better generality and reducing overfitting at the same time you will have to optimize select params (bec. of time boundation).

# One more problem which `still` can occur is target leakage, which CatBoost tries to reduce.

#



params = {

    ####### Learning Task Params #######

    'objective': 'reg:linear',                                                # 'reg:linear', 'reg:logistic', 'binary:logistic', 'binary:logitraw', 'binary:hinge', 'count:poisson' (deafult 'max_delta_step': 0.7), 'survival:cox', 'multi:softmax', 'multi:softprob', 'ran:pairwise', 'rank:ndcg', 'rank:map', 'reg:gamma' or 'reg:tweedie'.

    'base_score': 0.5,                                                        # default = 0.5; initial pred score for all instances, global bias. Given sufficient iterations changing this won't have much effect

    'eval_metric': 'rmse',                                                    # 'rmse', 'mae', 'error', 'error@t', 'merror', 'mlogloss', 'auc', 'aucpr', 'ndcg', 'map', 'ndcg@n', 'map@n', 'ndcg-', 'map-', 'ndcg@-', 'map@-', 'poisson-nloglik', 'gamma-nloglik', 'cox-nloglik', 'gamma-deviance' or 'tweedie-nloglik' (with specified 'tweedie_variance_power')

    'seed': 2019,

    

    ####### General Parameters #######

    'booster': 'gbtree',                                                      # 'gbtree', 'gblinear', 'dart'

    'verbosity': 0,

    'nthread': 2,                                                             # Number of threads to use. Default = max threads available.

    'disable_default_eval_metric': 0,                                         # To disable default eval metric, that is always computed.

    #'num_pbuffer':,                                                          # Automatically set by XGBoost. For storing prediction results of last boosting step. Set eqaul to number of training instances.

    #'num_feature':,                                                          # Automatically set by XGBoost. Features used for boosting. Set to max features by XGBoost.

    

    ####### Params for Booster #######

    'eta': 0.1,                                                               # alias: 'learning_rate'

    #'gamma':0.01,                                                            # Min Loss reduction required on partition

    #'min_child_weight':0.05,                                                 # Min. sum of hessian weights needed in a child

    #'max_delta_step':0.001,                                                  # Max delta step for each tree's weight estimation. Max output for leaves = learning_rate*max_delta_step

    'subsample':0.8,                                                          # Subsample of data for each tree

    'colsample_bytree':1.0,                                                   # Subsample columns for each tree

    'colsample_bylevel':0.9,                                                  # Subsample columns by level in each tree

    'colsample_bynode':0.9,                                                   # Subsample columns by node in each tree

    #'colsample_by*':0.8,                                                     # For setting all colsample_by* params

    #'lambda': 0.01,                                                          # L2 reg

    #'alpha': 0.02,                                                           # L1 reg

    'tree_method': 'auto',                                                    # 'auto', 'exact', 'approx', 'hist', 'gpu_exact' or 'gpu_hist'

    #'sketch_eps': 0.03,                                                      # For 'approx'tree method. For number of bins. n_bins = 1/sketch_eps

    #'scale_pos_weight': 0.5,                                                 # For Balancing of +ve and -ve weights  ## For Binary Classification ##

    'updater': 'grow_colmaker,prune',                                         # Sequence of tree updaters to run. 'grow_colmaker,distcol,grow_histmaker,grow_skmaker,sync,refresh,prune'. For distributed with 'hist' tree method uses 'grow_histmaker,prune'.

    'refresh_leaf': 1,                                                        # 0 -> Update only node stats, 1 -> Update both leaf and node stats (default).

    'process_type': 'default',                                                # 'deafult' -> Normal Boosting. OR 'update' -> For each tree in model specified updaters are run on them. Can have smaller number of trees in final model. Use 'refresh,prune' with it. You cannot use updater that creates new trees with it.

    #'grow_policy': 'depthwise',                                              # For 'hist' tree method. 'depthwise' -> Split nodes closest at root OR 'lossguide' -> Split at highest loss change.

    #'max_leaves': 0,                                                         # For 'lossguide' grow policy. Max nodes to be added.

    #'max_bin': 256,                                                          # For 'hist' tree method. Max bins for continuous features. higher value -> improves optimality of splits -> Higher computation time.

    #'predictor': 'cpu_predictor',                                            # 'cpu_predictor' OR 'gpu_predictor' to be used with 'gpu_exact', 'gpu_hist' tree method.

    #'num_parallel_tree': 1,                                                  # For Boosted Random Forest. For number of parallel trees during each iteration.

    

    ####### For 'reg:tweedie' objective #######

    #'tweedie_variance_power':1.5,                                            # range:(1, 2); towards 1 -> Poisson dist, towards 2 -> Gamma dist

    

    ####### For 'gblinear' booster #######

    #'lambda': 0,                                                             # L2 reg

    #'alpha': 0,

    #'updater': 'shotgun',                                                    # 'shotgun' -> hogwild parallelism (non-deterministic results) or 'coord_descent' -> multithreaded (deterministic)

    #'feature_selector': 'cyclic',                                            # 'cyclic', 'shuffle', 'random', 'greedy', 'thrifty'

    #'top_k': 10                                                              # For 'greedy' and 'thrifty' feature selectors. To speed them up, at cost of some accuracy.

    

    ####### For 'dart' booster #######

    #'sample_type': 'uniform',                                                # 'uniform' or 'weighted'. For selecting trees to drop. 'weighted' uses tree weight used in calculating result.

    #'normalize_type': 'tree',                                                # 'tree' or 'forest'. For setting weight of new and dropped trees.

    #'rate_drop': 0.2,                                                        # Dropout rate

    #'one_drop': True,                                                        # True for always dropping atleast one tree.

    #'skip_drop': 0.3                                                         # Probability of skipping Dropout at each iteration. Higher priority than 'rate_drop' and 'one_drop'.

}
results = {} # Will hold evaluation results
reg_model = xb.train(params, 

                     train_data,                                              # DMatrix

                     num_boost_round=1000,                                    # Number of Boosting rounds/trees.

                     evals=[(train_data, "train"), (valid_data, "valid")],

                     obj=None,                                                # Customized Objective Function

                     feval=None,                                              # Customized evaluation time

                     maximize=False,                                          # if True maximize eval function

                     early_stopping_rounds=50,

                     evals_result=results,                                    # takes dict where results will be stored

                     verbose_eval=100,

                     xgb_model=None,

                     callbacks=None,                                          # list of callback funcs or callback func

                    )
reg_model.best_iteration, reg_model.best_ntree_limit, reg_model.best_score
reg_model.get_split_value_histogram('CRIM').head()      # Get split values by feature name
results['train']['rmse'][-10:]
kwargs = {}   # kwargs is not supported in sklearn. It may cause unexpected issues. (From LightGBM)
reg_model = lgb.LGBMRegressor(

                  boosting_type    ='gbdt',                                     # 'gbdt', 'dart', 'goss', 'rf'(RandomForest)

                  num_leaves       =31,

                  max_depth        =-1,

                  learning_rate    =0.1,

                  n_estimators     =100,                                        # Number of Boosted Trees to fit.

                  subsample_for_bin=200000,                                     # Number of samples for costructing bins.

                  objective        =None,                                       # string, callable or None; eg 'regression', 'binary', 'multiclass', 'lambdarank'

                  class_weight     =None,                                       # For multiclass. dict, 'balanced' or None. Dict as {class_label: weight, ...}. For binary you can use 'is_unbalance' or 'scale_pos_weight'

                  min_split_gain   =0.0,                                        # Min loss reduction required for splitting

                  min_child_weight =0.001,                                      # Min. hessian weight sum required in a leaf.

                  min_child_samples=20,                                         # Min data needed in a leaf

                  subsample        =1.0,

                  subsample_freq   =0,                                          # Frequency of using 'subsample' param

                  colsample_bytree =1.0,                                        # subsample ratio of columns when constructing each tree

                  reg_alpha        =0.0,                                        # L1 regularization

                  reg_lambda       =0.0,                                        # L2 regularization

                  random_state     =2019,

                  n_jobs           =-1,                                         # Number of parallel threads.

                  silent           =True,

                  importance_type  ='split',                                    # Type of feature importances to be filled into 'feature_importancs_'. 'split' or 'gain' -> gains of split which use th feature.

                  **kwargs                                                      # Other params for model supported by LightGBM

)
from sklearn.model_selection import train_test_split as tts



Xtrain, Xvalid, ytrain, yvalid = tts(train_df.drop('target', axis=1), train_df['target'])
feature_names = train_df.drop('target', axis=1).columns.values.tolist()



reg_model = reg_model.fit(

    Xtrain, ytrain,

    sample_weight             =None,                                            # Array of weights for all data points

    init_score                =None,                                            # Array of size n_samples for initial score of training data

    eval_set                  =[(Xtrain, ytrain), (Xvalid, yvalid)],            # set of pairs as [(x1, y1), (x2, y2) ... ] to evaluate during training

    eval_names                =["Train", "Validation"],

    eval_sample_weight        =[None, None],                                    # list of arrays as weights for eval data

    eval_init_score           =None,                                            # list of arrays, initial score for eval data

    eval_metric               ='rmse',                                          # string, list of strings, callable or None. eg: 'l2', 'logloss', 'ndcg'

    early_stopping_rounds     =None,

    verbose                   =50,

    feature_name              =feature_names,                                   # list of strings. Feature names.

    categorical_feature       ='auto',                                          # list of int, list of strings OR 'auto' -> pandas categorical is used to check. ## All values should be less than int32 ## ### Negative values treated as missing. ###

    callbacks                 =None                                             # list of callbacks or None.

)
reg_model.best_iteration_, reg_model.best_score_
reg_model.feature_importances_
# Getting output of certain tree leaf



tree_index = 0

leaf_index = 0



reg_model.booster_.get_leaf_output(tree_index, leaf_index)
from sklearn.model_selection import train_test_split as tts



Xtrain, Xvalid, ytrain, yvalid = tts(train_df.drop('target', axis=1), train_df['target'])
train_dataset = lgb.Dataset(Xtrain, label=ytrain,

                            reference=None,                    # If this Dataset is for validation, training Dataset should be used as reference.

                            weight=None,                       # list or None. Weight for each instance.

                            group=None,                        # list or None. Group/query size for Dataset.

                            init_score=None,                   # list or None. Initial score for dataset.

                            silent=False,

                            feature_name='auto',

                            params=None,                       # dict or None. Other params for dataset

                            free_raw_data=True                 # If True raw data is freed after constructing Dataset.

                           )



valid_dataset = lgb.Dataset(Xvalid, label=yvalid,

                            reference=train_dataset,           # If this Dataset is for validation, training Dataset should be used as reference.

                            weight=None,                       # list or None. Weight for each instance.

                            group=None,                        # list or None. Group/query size for Dataset.

                            init_score=None,                   # list or None. Initial score for dataset.

                            silent=False,

                            feature_name='auto',

                            params=None,                       # dict or None. Other params for dataset

                            free_raw_data=True                 # If True raw data is freed after constructing Dataset.

                           )
#

# Top problem of Boosting is that it can overfit very easily. So most parameters you will see here are to reduce overfitting.

# For better generality and reducing overfitting at the same time you will have to optimize select params (bec. of time boundation).

# One more problem which `still` 'can' occur is target leakage, which CatBoost tries to reduce.

#



params = {

    ####### Core Params #######

    'objective':'regression',                          # 'regression', 'regression_l1', 'regression_l2', 'huber', 'fair', 'poisson', 'quantile', 'mape', 'gamma', 'tweedie', 'binary', 'multiclass', 'multiclassova', 'xentropy', 'xentlambda', 'lambdarank'

    'boosting':'gbdt',                                 # 'gbdt', 'gbrt', 'rf', 'dart', 'goss'

    #'num_iterations':100,                             # Number of Boosting iterations. [Internally LightGBM constructs num_class*num_iterations trees for `multiclass` classifications]

    'learning_rate':0.1,                               # Shrinkage rate. ## In 'dart' it also affects normalizing weights of dropped trees. ##

    'num_leaves':31,                                   # Max number of leaves in one tree

    'tree_learner':'serial',                           # 'serial', 'feature' -> feature parallel tree learner, 'data' -> data paralle tree learner, 'voting' -> voting parallel tree learner

    'num_threads':0,                                   # Best use as number of CPU cores. For parallel learning do not use all CPU cores. This might cause poor performance for network communications.

    'device_type':'cpu',                               # 'cpu', 'gpu' ## Use smaller 'max_bin' for better speedup ## ## For better accuray with GPU use 'gpu_use_dp'=true at speed cost. ##

    'seed':2019,

    

    ####### Learning Control Parameters #######

    'max_depth':-1,                                    # For limiting max depth of tree model

    'min_data_in_leaf':20,                             # Min data in each leaf

    'min_sum_hessian_in_leaf':1e-3,                    # Min sum of hessians in each leaf

    'bagging_fraction':1.0,                            # like 'feature_fraction' but this will randomly select part of data without resampling. ## To enable Bagging 'bagging_freq' should be set to non sero value as well. ##

    'bagging_freq':0,                                  # int. Frequency of bagging. n -> enable bagging at every n iteration. ## To enable bagging set 'bagging_fraction' < 1.0 as well. ##

    'bagging_seed':2019,

    'feature_fraction':1.0,                            # Subsample of features to select for each iteration

    'feature_fraction_seed':2,

    #'early_stopping_round':0,

    'max_delta_step':0.0,                              # Limit max output of tree leaves. Final max output of leaves = learning_rate*max_delta_step

    'reg_alpha':0.0,                                   # L1 Regularization

    'reg_lambda':0.0,                                  # L2 Regularization

    'min_gain_to_split':0.0,                           # Min gain needed to perform split

    

    #'drop_rate':0.1,                                  # Used with 'dart' boosting. Dropout rate

    #'max_drop':50,                                    # Used with 'dart' boosting. Max number of dropped trees

    #'skip_drop':0.5,                                  # Used with 'dart' boosting. Probab of skipping Dropout procedure at each iteration.

    #'xgboost_dart_mode':False,                        # Used with 'dart' boosting. To enable XGBoost's dart mode.

    #'uniform_drop':False,                             # Used with 'dart' boosting. If you want to use Uniform Drop.

    #'drop_seed':4,                                    # Used with 'dart' boosting.

    

    #top_rate:0.2,                                     # Used with 'goss' boosting. Retain ratio of large gradient data.

    #other_rate:0.1,                                   # Used with 'goss' boosting. Retain ratio of small gradient data.

    

    'min_data_per_group':100,                          # Min data per categorical group

    'max_cat_threshold':32,                            # Limit max threshold points in categorical features.

    'cat_l2':10.0,                                     # L2 Regularization in categorical split

    'cat_smooth':10.0,                                 # For reducing noises in categorical features, especially for 'categories' with few data points

    'max_cat_to_onehot':4,                             # If categories less or wqual to this, One-vs-Other split algorithm will be used.

    

    #'top_k':20,                                       # Used in Voting Parallel. Larger value for accurate result at cost of speed.

    

    'monotone_constraints':None,                       # For constraining monotonic featurs. 0 -> no constraint, 1 -> monotonically inc, -1 -> monotonically dec; given for all features as "-1,1,0,0..."

    'feaature_contri':None,                            # For controlling features' split gain. gain[i] = max(0, feature_contri[i])*gain[i]. specify all features in order as "0.3, 0.1,..."

    

    ####### Objective Parameters #######

    'num_class':1,                                     # Used only in multiclass classification

    'is_unbalance':False,                              # Used only in binary. ## Cannot be used with scale_pos_weight. Use only one of them. ##

    #'scale_pos_weight':1.0,                           # Used only in binary. Weight of labels with +ve class. ## Cannot be used with is_unbalance. Use only one of them. ##

    'sigmoid':1.0,                                     # >0.0; Used only in binary and multiclassova classifications, and lambdarank applications. Param for sigmoid function.

    'reg_sqrt':False,                                  # Used only in Regression. Fit sqrt(label) instead of original and preds will automatically to squared. # Might be useful in large range labels. #

    'alpha':0.9,                                       # >0.0; Used only in Huber, Quantile regression. Param for Huber Loss and Quantile Regression.

    'fair_c':1.0,                                      # >0.0; Used only in Fair regression. Param for Fair Loss

    'poisson_max_delta_step':0.7,                      # >0.0; Used only in Poisson regression. Param for poisson regression to safeguard optimization.

    'tweedie_variance_power':1.5,                      # 1.0 <= x < 2.0; Used in Tweedie Regression. Used to control variance of tweedie distribution. Closer to 2 -> Gamma, closer to 1 -> Poisson.

    'max_position':20,                                 # >0; Used only in lambdarank. Optimizes NDCG at this position

    #'label_gain':                                     # Used only for lambdarank. Relevant gain for labels. eg: gain for label 2 will be 3; as "1,2,3,7,15,31,63,...2^30-1"

    

    ####### Metric Params #######

    'metric':"rmse",                                   # 'None','l1', 'l2', 'l2_root', 'quantile', 'mpae', 'huber', 'fair', 'poisson', 'fair', 'poisson', 'gamma', 'gamma_deviance', 'tweedie', 'ndcg', 'map', 'auc', 'binary_logloss', 'binary_error', 'multi_logloss', 'multi_error', 'xentropy', 'xentlambda', 'kldiv'

    'metric_freq':1,                                   # Freq of metric output

    'eval_at':"1,2,3,4,5",                             # Used only with 'ndcg' and 'map'. NDCG and MAP evaluation positions separated by `,`.

    

    ####### IO Params #######

    'verbosity': 1,

    'max_bin': 255,                                    # Max number of bins that feature values will be bucketed in. ## LightGBM compresses memory as per this param. eg uint8_t for 255 ##

    'min_data_in_bin':3,                               # Min data inside one bin

    'bin_construct_sample_cnt':200000,                 # Number of data points sampled to construct histogram bins

    'histogram_pool_size':-1.0,                        # Max cache size in MB for 'historical' histogram. <0 -> No Limit

    'data_random_seed':1,                              # Random seed for data partitioning in parallel learning (excluding feature_parallel mode)

    #'initscore_filename':"",                          # Path of file with training initial scores. ## Only works in case of loading data directly fom file ##

    #'valid_data_initscores':"",                       # Path(s) of file(s) with validation initial scores. as f"{path1},{path2}..." ## Only works in case of loading data directly fom file ##

    'pre_partition':False,                             # Used for Parallel Learning (excluding feature_parallel mode). True if training data is pre partitioned and diff machines use diff partitions

    

    'enable_bundle':True,                              # Set to False to disable Exclusive Feature Bundling (EFB).

    'max_conflict_rate':0.0,                           # max conflict rate for bundles in EFB.

    

    'is_enable_sparse':True,                           # To enable/disable sparse optimization

    'sparse_threshold':0.8,                            # Threshold for zero elements percentage for treating a feature as a sparse one

    'use_missing':True,                                # False to diable special handling of missing values

    'zero_as_missing':False,                           # True for treating zero's as missing value. False -> NA as missing

    

    'two_round':False,                                 # If data is too big to fit in memory.  ## Works only when loading data directly from file ##

    'header':False,                                    # True if input data is header. ## Works only when loading data directly from file ##

    

    'predict_raw_score':False,                         # False -> predict transformed scores.  # Used only in prediction task #

    'predict_leaf_index':False,                        # True -> predct with leaf index of all trees. # Used only in prediction task #

    'predict_contrib':False,                           # True -> predict SHAP values. # Used only in prediction task #

    'num_iteration_predict':-1,                        # Number of trained iterations to be used in prediction. # Used only in prediction task #

    'pred_early_stop':False,                           # True -> Use early stopping to speed up prediction. May affect accuracy. # Used only in prediction task #

    'pred_early_stop_freq':10,                         # Freq of checking early stopping prediction. # Used only in prediction task #

    'pred_early_stop_margin':10.0,                     # Threshold of margin in early stopping prediction. # Used only in prediction task #

    

    ####### Network Params #######

    'num_machines':1,                                  # Number of machines for parallel learning. # Needed to be set for both socket and mpi versions #

    'local_listen_port':12400,                         # TCP listen port for local machines.

    'time_out':120,                                    # socket time-out in minutes.

    

    ####### GPU Params #######

    'gpu_platform_id':-1,                              # OpenCL platform ID. Usually each GPU vendor exposes one OpenCL platform. -1 -> system-wide default plaform.

    'gpu_device_id':-1,                                # OpenCL device ID in specified platform. Each GPU in selected platform has a unique device ID.

    'gpu_use_dp':False,                                # True -> To use double precision math on GPU.

}
result = {}
reg_model = lgb.train(

                      params,

                      train_dataset,

                      num_boost_round=100,                                      #

                      valid_sets=[train_dataset, valid_dataset],                #

                      valid_names=['Train', 'Valid'],                           #

                      fobj=None,                                                #

                      feval=None,                                               #

                      init_model=None,                                          #

                      feature_name='auto',                                      #

                      categorical_feature='auto',                               #

                      early_stopping_rounds=50,                                 #

                      evals_result=result,                                      #

                      verbose_eval=50,                                          #

                      learning_rates=None,                                      #

                      keep_training_booster=False,                              #

                      callbacks=None                                            #

)
reg_model.best_iteration, reg_model.best_score
reg_model.feature_importance()
result['Train']['rmse'][-10:]
reg_model = cb.CatBoostRegressor(

                iterations=1000,                                        # Max number of trees. (default: 1000)

                learning_rate=0.01,

                depth=6,                                                # Depth of tree. Can be upto 16. (def: 6)

                l2_leaf_reg=3.0,                                        # L2 Regularization. (def: 3.0)

                model_size_reg=None,                                    # CPU only; [0, inf). Regularize model size. (def: None [=0.5])

                rsm=0.95,                                               # CPU only;(0, 1]. Random subspace method. Percentage of features to use at each split selection. (def: None[=1])

                loss_function='RMSE',                                   # CPU and GPU; string or object. 'RMSE', 'Logloss', 'MAE', 'CrossEntropy', 'Quantile', 'LogLinQuantile', 'Lq', 'MultiClass', 'MultiClassOneVsAll', 'MAPE', 'Poisson', 'PairLogit', 'PairLogitPairwise', 'QueryRMSE', 'QuerySoftMax', 'YetiRank', 'YetiRankPairwise'. (Custom: 'Quantile:alpha=0.1')

                border_count=None,                                      # CPU and GPU; [1, 255]. Number of splits for numerical features. (def: 254 for CPU, 128 for GPU)

                feature_border_type=None,                               # CPU and GPU; string. Binarization mode for numerical feature. 'Meadin', 'Uniform', 'UniformAndQuantile', 'MaxLogSum', 'MinEntropy', 'GreedyLogSum'. (def: 'GreedyLogSum')

                fold_permutation_block_size=None,                       # CPU and GPU; int. Objects in dataset that are grouped in blocks before random permutations. Smaller -> slower training; Large -> quality degradation. (def: 1)

                od_pval=None,                                           # CPU and GPU; float; [10^-10, 1-^-2] for best results. Threshold for IncToDec overfitting detector type. Larger -> Overfitting is detected earlier. (def: 0[=off]) # Validation Data should be given #  ## Do not use with Iter overfitting detectot type. ##

                od_wait=None,                                           # CPU and GPU; int. Number of iterations after optimal result before stopping. Depends on Overfitting detector type: IncToDec -> Ignore overfitting detector when threshold is reached and countinue learning for specified iterations OR Iter -> like EarlyStoppingRound. (def: 20)

                od_type=None,                                           # CPU and GPU; string. 'IncToDec' or 'Iter'. (def: 'IncToDec')

                nan_mode=None,                                          # CPU and GPU; string. 'Forbidden' -> Missing value will give error, 'Min' -> taken as less than all other values. (a differentiating split can be made) OR 'MAX' -> larger than all other values (differentiating split can be made). (def: 'Min') ## This can be set for individual features with custom quantization and missing value modes input file. ##

                counter_calc_method=None,                               # CPU and GPU; string. For calculating the Counter CTR Type. 'SkipTest' -> Objs from validation not considered OR 'Full' -> All objs from both learn and valid are considered. (def: None[='Full'])

                leaf_estimation_iterations=None,                        # CPU and GPU; int. Number of gradient steps when calculating the values in leaves. (def: None[Depends on training objective])

                leaf_estimation_method=None,                            # CPU and GPU; string. Method for calculating values in leaves. 'Newton' or 'Gradient'. (def: 'Gradient')

                thread_count=None,                                      # CPU and GPU; int. CPU -> Optimizes speed, GPU -> given value for reading data from hard drive and does not affect training; During training one main thread and one thread for each GPU are used. (def: -1[= number of cores])

                random_seed=None,

                use_best_model=None,                                    # CPU and GPU; bool. True -> Use validation set to identify best iterations. (def: True[if validation is provided]). # Requires Validation data. #

                best_model_min_trees=None,                              # CPU and GPU; int. Min number of trees that the best model should have. (def: None)  ## Should be used with 'use_best_model' param. ##

                verbose=None,                                           # CPU and GPU; string. 'Silent', 'Verbose', 'Info', 'Debug'

                silent=None,

                logging_level=None,                                     # CPU and GPU; string. 'Silent', 'Verbose', 'Info', 'Debug'.  (def: None)

                metric_period=None,                                     # CPU and GPU; int; >0. Freq of iterations to calc the values of obj and metrics. (def: 1)

                ctr_leaf_count_limit=None,                              # CPU only; int. Max leaves with categorical features. Only leaves with top freq of values are selected. Reduces model size and memory at cost of quality. (def: None[=not limited])

                store_all_simple_ctr=None,                              # CPU only; bool. Ignore cat features which are not used in feature combinations, when choosing candidates for exclusion. (def: None[= False]) ## Should be used with 'ctr_leaf_count_limit' ##

                max_ctr_complexity=None,                                # CPU and GPU; int. Max number of categorical featurse that can be combined. (def: 4)

                has_time=None,                                          # CPU and GPU; bool. True -> do not random permute during transforming cat features and choosing tree structure. (def: False)

                allow_const_label=None,                                 # CPU and GPU; bool. Use with datasets having equal label values for all objects. (def: False)

                one_hot_max_size=None,                                  # CPU and GPU; int. Max categoried cat feature to one hot encode. (def: 2)

                random_strength=None,                                   # CPU and GPU; float. Amount of randomness for scoring splits when tree structure is selected. (def: 1) # Can avoid Overfitting. # ## Not supported for 'QueryCrossEntropy', 'YetiRankPairwise', 'PairLogitPairwise' loss functions. ##

                name=None,                                              # CPU and GPU; string. Experiment name to display in visualization tools. (def: 'experiment')

                ignored_features=None,                                  # CPU and GPU; list. Indices of features to exclude from training. If training should exclude 1,2,7,42,43,44,45 then "1:2:7:42-45". (def: None)

                train_dir=None,                                         # CPU and GPU; string. Dir for storing the files generated during training. (def: 'catboost_info')

                custom_metric=None,                                     # CPU only; string or list of strings. Metric values to output during training. Custom: 'Quantile:[alpha:0.1;...]'. 'RMSE', 'Logloss', 'MAE', 'CrossEntropy', 'Quantile', 'LogLinQuantile', 'Lq', 'MultiClass', 'MultiClassOneVsAll', 'MAPE', 'Poisson', 'PairLogit', 'PairLogitPairwise', 'QueryRMSE', 'QuerySoftMax', 'SMAPE', 'Recall', 'Precision', 'F1', 'TotalF1', 'Accuracy', 'BalancedAccuracy', 'BalancedErrorRate', 'Kappa', 'WKappa', 'LogLikelihoodOfPrediction', 'AUC', 'R2', 'NumErrors', 'MCC', 'BrierScore', 'HingeLoss', 'HammingLoss', 'ZeroOneLoss', 'MSLE', 'MedianAbsoluteError', 'PairAccuracy', 'AverageGain', 'PFound', 'NDCG', 'PrecisionAt', 'RecallAt', 'MAP', 'CtrFactor'.

                eval_metric=None,                                       # CPU only; string or object.  'RMSE', 'Logloss', 'MAE', 'CrossEntropy', 'Quantile', 'LogLinQuantile', 'Lq', 'MultiClass', 'MultiClassOneVsAll', 'MAPE', 'Poisson', 'PairLogit', 'PairLogitPairwise', 'QueryRMSE', 'QuerySoftMax, 'SMAPE', 'Recall', 'Precision', 'F1', 'TotalF1', 'Accuracy', 'BalancedAccuracy', 'BalancedErrorRate', 'Kappa', 'WKappa', 'LogLikelihoodOfPrediction', 'AUC', 'R2', 'NumErrors', 'MCC', 'BrierScore', 'HingeLoss', 'HammingLoss', 'ZeroOneLoss', 'MSLE', 'MedianAbsoluteError', 'PairAccuracy', 'AverageGain', 'PFound', 'NDCG', 'PrecisionAt', 'RecallAt', 'MAP'.

                bagging_temperature=None,                               # CPU and GPU; float. Param for Bayesian Bootstrap to assign random weights to objects. 0 -> All weight's are '1' OR 1 -> weight's sampled from exponential distribution. (def: 1)

                save_snapshot=None,                                     # CPU and GPU; bool. To enable snapshotting for restoring the training progress after an interruption. (def: None)

                snapshot_file=None,                                     # CPU and GPU; string. Name of file to save training progress in. (def: 'experiment.cbsnapshot')

                snapshot_interval=None,                                 # CPU and GPU; int. Interval between saving snapshots in seconds. Last snapshot at end of training. (def: 600)

                fold_len_multiplier=None,                               # CPU and GPU; float; > 1.0. Coeff. for changing lengths of folds. (def: 2) # Best valid. results with min values. #

                used_ram_limit=None,                                    # CPU only; int. `Attempt` to limit amount of CPU memory used. # Often affects CTR calc memory usage. MB, KB, GB as '32gb'. # ## In some cases it is impossible to limit RAM usage. ##

                gpu_ram_part=None,                                      # GPU only; float. How much of GPU RAM to use for training. (def: 0.95)

                pinned_memory_size=None,                                # GPU only; int. How much pinned(page-locked) CPU RAM to use per GPU. (def: 1073741824)

                allow_writing_files=None,                               # CPU only; bool. Allow to write 'analytical' snapshot files during training. (def: True) # Fasle -> snapshot and data viz tools will be unavailable. #

                final_ctr_computation_mode=None,                        # CPU and GPU; string. Final CTR computation mode. 'Default' -> Compute final CTRs for learn and validation mode OR 'Skip' -> Do not compute final CTRs for learn and validation datasets. In this case the resulting model cannot be applied. Dec.s the size of resulting model. ## Can be useful for research purposes when only metric values have to be calculated. ##

                approx_on_full_history=None,                            # CPU only; bool. Principlees of calculating approximated values. 'False' -> Use only a fraction of fold to calc approx values; size of fraction = 1/coeff OR 'True' -> Use all preceding rows in the fold for calc approx values. (def: False)

                boosting_type=None,                                     # CPU and GPU[only Plain mode for MultiClass loss on GPU]; string. Boosting scheme. 'Ordered' -> Usually provides better quality on small datasets OR 'Plain' -> The classic gradient boosting scheme. (def: depends on number of data points and learning mode)

                simple_ctr=None,                                        # CPU and GPU; string. Binarization settings for simple categorical features. As 'CtrType[:TargetBorderCount=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]...'. CtrType: 'Borders', 'Buckets', 'BinarizedTargetMeanValue', 'Counter' (GPU: 'Borders', 'Buckets', 'FeatureFreq', 'FloatTargetMeanValue'); TargetBorderCount: [1, 255]; TargetBorderType: 'Mean', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum', 'MinEntropy', 'GreedyLogSum'; CtrBorderCount: [1, 255]; CtrBorderType: 'Median', 'Uniform', 'UnifromAndQuantiles', 'MaxLogSum', 'GreedyLogSum'; Prior: number (adds value to numerator) (GPU: two slash delimited numbers -> first to num., second to den.)

                combinations_ctr=None,                                  # CPU and GPU; string. Binarization settings for simple categorical features. As 'CtrType[:TargetBorderCount=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]...'. CtrType: 'Borders', 'Buckets', 'BinarizedTargetMeanValue', 'Counter' (GPU: 'Borders', 'Buckets', 'FeatureFreq', 'FloatTargetMeanValue'); TargetBorderCount: [1, 255]; TargetBorderType: 'Mean', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum', 'MinEntropy', 'GreedyLogSum'; CtrBorderCount: [1, 255]; CtrBorderType: 'Median', 'Uniform', 'UnifromAndQuantiles', 'MaxLogSum', 'GreedyLogSum'; Prior: number (adds value to numerator) (GPU: two slash delimited numbers -> first to num., second to den.)

                per_feature_ctr=None,                                   # CPU and GPU; string. Binarization settings for simple categorical features. As 'CtrType[:TargetBorderCount=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]...'. CtrType: 'Borders', 'Buckets', 'BinarizedTargetMeanValue', 'Counter' (GPU: 'Borders', 'Buckets', 'FeatureFreq', 'FloatTargetMeanValue'); TargetBorderCount: [1, 255]; TargetBorderType: 'Mean', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum', 'MinEntropy', 'GreedyLogSum'; CtrBorderCount: [1, 255]; CtrBorderType: 'Median', 'Uniform', 'UnifromAndQuantiles', 'MaxLogSum', 'GreedyLogSum'; Prior: number (adds value to numerator) (GPU: two slash delimited numbers -> first to num., second to den.)

                #ctr_target_border_count=None,                           # CPU and GPU; int; [1, 255]. Max number of borders to use in target binarization for categorical features that need it. (def: num_class-1(CPU), 1 otherwise) # Overrides one specified in 'simple_ctr', 'combinations_ctr', 'per_feature_ctr' #

                task_type=None,                                         # CPU and GPU; Processing Unit (PU) to use for training. 'CPU' or 'GPU'. (def: 'CPU')

                device_config=None,                                     # 

                devices=None,                                           # GPU only; string. IDs of the GPU devices to use for training. As "0:2:4-7" (def: None[=all])

                bootstrap_type=None,                                    # CPU and GPU; string. Bootstrap type. 'Poisson'(GPU only), 'Bayesian', 'Bernoulli', 'No'. (def: 'Bayesian')

                subsample=None,                                         # CPU and GPU; float. Sample rate for Bagging types 'Poisson', 'Bernoulli'. (def: 0.66)

                dev_score_calc_obj_block_size=None,                     #

                gpu_cat_features_storage=None,                          # GPU only; string. Method for storing cat features values. 'CpuPinnedMemory', 'GpuRam'. (def: None[='GPuRam']) # Use 'CpuPinnedMemory' if feature combinations are used and GPU mem. is not sufficient. #

                data_partition=None,                                    # GPU only; string. Method for splitting input dataset btw multiple workers. 'FeatureParallel' -> split by features and calc their values on diff GPUs OR 'DocParallel' -> split by objects and calc each of these on certain GPU. (def: depends on learning mode and input datasets)

                metadata=None,                                          # 

                early_stopping_rounds=None,                             # CPU and GPU; int. Set overfitting detector type to 'Iter' and stop training after these many iterations after seeing best so far. (def: False)

                cat_features=None,                                      # 

                #growing_policy=None,                                    #

                #min_samples_in_leaf=None,                               #

                #max_leaves_count=None                                   #

)
from sklearn.model_selection import train_test_split as tts



Xtrain, Xvalid, ytrain, yvalid = tts(train_df.drop('target', axis=1), train_df['target'])
reg_model = reg_model.fit(

             Xtrain, y=ytrain,                                          # catboost.Pool or list or array or DataFrame or Series or string(file)

             cat_features=None,                                         # list, array, None. ## Use only if X is not catboost.Pool ##

             sample_weight=None,                                        # list, array, DataFrame, Series, None. Inctance weights.

             baseline=None,                                             # list, array, None. 2D array like data. ## Use only if X is not catboost.Pool ##

             use_best_model=True,                                       # bool, None. Weather to use best model.

             eval_set=[(Xtrain, ytrain), (Xvalid, yvalid)],              # catboost.Pool, list, None. List of pairs to evaluate as [(X1, y1), (X2, y2) ...]

             verbose=100,                                               # bool, int. 

             logging_level=None,                                        # 'Silent', 'Verbose', 'Info', 'Debug'.

             plot=True,                                                 # True to draw train and eval error in Jupyter Notebook.

             column_description=None,                                   # 

             metric_period=None,                                        #

             silent=None,                                               #

             early_stopping_rounds=50,                                  # Activates 'Iter' overfitting detector with od_wait set ti early_stopping_rounds.

             save_snapshot=None,                                        # bool, None. Enable progress snapshot.

             snapshot_file=None,                                        # Learn progress snapshot file path, if None -> will use default filename.

             snapshot_interval=None                                     # int. Interval btw saving snapshots (seconds).

             )
reg_model.best_iteration_, reg_model.best_score_
reg_model.feature_importances_
reg_model.evals_result_['validation_1']['RMSE'][-10:]
from sklearn.model_selection import train_test_split as tts



Xtrain, Xvalid, ytrain, yvalid = tts(train_df.drop('target', axis=1), train_df['target'])
from catboost import FeaturesData as FD         # Class to store features data in optimized form to pass to Pool constructor

from catboost import Pool as P
train_fd = FD(

              num_feature_data=Xtrain.drop(['CHAS', 'RAD'], axis=1).values.astype(np.float32),            # np.ndarray; dtype=np.float32

              cat_feature_data=Xtrain.loc[:, ['CHAS', 'RAD']].astype(str).values,                         # np.ndarray; dtype=object. ## Elements must have bytese type, containing utf-8 encoded strings. ##

              num_feature_names=Xtrain.drop(['CHAS', 'RAD'], axis=1).columns.values.tolist(),

              cat_feature_names=['CHAS', 'RAD']

)



valid_fd = FD(

              num_feature_data=Xvalid.drop(['CHAS', 'RAD'], axis=1).values.astype(np.float32),            # np.ndarray; dtype=np.float32

              cat_feature_data=Xvalid.loc[:, ['CHAS', 'RAD']].astype(str).values,                         # np.ndarray; dtype=object. ## Elements must have bytese type, containing utf-8 encoded strings. ##

              num_feature_names=Xvalid.drop(['CHAS', 'RAD'], axis=1).columns.values.tolist(),

              cat_feature_names=['CHAS', 'RAD']

)
#train_data = Pool(datset_desc_file_path, column_description=cd_file)

train_data = P(

                data=train_fd,

                label=ytrain,

                cat_features=None,

)



valid_data = P(

                data=valid_fd,

                label=yvalid,

                cat_features=None,

)
params = {

          ####### Common Params #######

          'loss_function':'RMSE',                                   # CPU and GPU; string or object. 'RMSE', 'Logloss', 'MAE', 'CrossEntropy', 'Quantile', 'LogLinQuantile', 'Lq', 'MultiClass', 'MultiClassOneVsAll', 'MAPE', 'Poisson', 'PairLogit', 'PairLogitPairwise', 'QueryRMSE', 'QuerySoftMax', 'YetiRank', 'YetiRankPairwise'. (Custom: 'Quantile:alpha=0.1')

          'custom_metric':None,                                     # CPU only; string or list of strings. Metric values to output during training. Custom: 'Quantile:[alpha:0.1;...]'. 'RMSE', 'Logloss', 'MAE', 'CrossEntropy', 'Quantile', 'LogLinQuantile', 'Lq', 'MultiClass', 'MultiClassOneVsAll', 'MAPE', 'Poisson', 'PairLogit', 'PairLogitPairwise', 'QueryRMSE', 'QuerySoftMax', 'SMAPE', 'Recall', 'Precision', 'F1', 'TotalF1', 'Accuracy', 'BalancedAccuracy', 'BalancedErrorRate', 'Kappa', 'WKappa', 'LogLikelihoodOfPrediction', 'AUC', 'R2', 'NumErrors', 'MCC', 'BrierScore', 'HingeLoss', 'HammingLoss', 'ZeroOneLoss', 'MSLE', 'MedianAbsoluteError', 'PairAccuracy', 'AverageGain', 'PFound', 'NDCG', 'PrecisionAt', 'RecallAt', 'MAP', 'CtrFactor'.

          'eval_metric':'RMSE',                                     # CPU only; string or object.  'RMSE', 'Logloss', 'MAE', 'CrossEntropy', 'Quantile', 'LogLinQuantile', 'Lq', 'MultiClass', 'MultiClassOneVsAll', 'MAPE', 'Poisson', 'PairLogit', 'PairLogitPairwise', 'QueryRMSE', 'QuerySoftMax, 'SMAPE', 'Recall', 'Precision', 'F1', 'TotalF1', 'Accuracy', 'BalancedAccuracy', 'BalancedErrorRate', 'Kappa', 'WKappa', 'LogLikelihoodOfPrediction', 'AUC', 'R2', 'NumErrors', 'MCC', 'BrierScore', 'HingeLoss', 'HammingLoss', 'ZeroOneLoss', 'MSLE', 'MedianAbsoluteError', 'PairAccuracy', 'AverageGain', 'PFound', 'NDCG', 'PrecisionAt', 'RecallAt', 'MAP'.

          'iterations':1000,                                        # CPU and GPU; int. Max number of trees. (default: 1000)

          'learning_rate':0.01,

          'random_seed':2019,

          'l2_leaf_reg':3.0,                                        # CPU and GPU; int. L2 Regularization. (def: 3.0)

          'bootstrap_type':'Bayesian',                              # CPU and GPU; string. Bootstrap type. 'Poisson'(GPU only), 'Bayesian', 'Bernoulli', 'No'. (def: 'Bayesian')

          'bagging_temperature':1,                                  # CPU and GPU; float. Param for Bayesian Bootstrap to assign random weights to objects. 0 -> All weight's are '1' OR 1 -> weight's sampled from exponential distribution. (def: 1)

          #'subsample':0.66,                                        # CPU and GPU; float. Sample rate for Bagging types 'Poisson', 'Bernoulli'. (def: 0.66)

          #'sampling_frequency':'PerTreeLevel',                     # CPU and GPU; string. Frequency to sample weights and objects when building trees. 'PerTree', 'PerTreeLevel'.

          'random_strength':1,                                      # CPU and GPU; float. Amount of randomness for scoring splits when tree structure is selected. (def: 1) # Can avoid Overfitting. # ## Not supported for 'QueryCrossEntropy', 'YetiRankPairwise', 'PairLogitPairwise' loss functions. ##

          'use_best_model':True,                                    # CPU and GPU; bool. True -> Use validation set to identify best iterations. (def: True[if validation is provided]). # Requires Validation data. #

          #'best_model_min_trees':None,                             # CPU and GPU; int. Min number of trees that the best model should have. (def: None)  ## Should be used with 'use_best_model' param. ##

          'depth':6,                                                # CPU and GPU; int. Depth of tree. Can be upto 16. (def: 6)

          #'ignored_features':None,                                 # CPU and GPU; list. Indices of features to exclude from training. If training should exclude 1,2,7,42,43,44,45 then "1:2:7:42-45". (def: None)

          'one_hot_max_size':3,                                     # CPU and GPU; int. Max categoried cat feature to one hot encode. (def: 2)

          'has_time':False,                                         # CPU and GPU; bool. True -> do not random permute during transforming cat features and choosing tree structure. (def: False)

          'rsm':0.95,                                               # CPU only;(0, 1]. Random subspace method. Percentage of features to use at each split selection. (def: None[=1])

          'nan_mode':'Min',                                         # CPU and GPU; string. 'Forbidden' -> Missing value will give error, 'Min' -> taken as less than all other values. (a differentiating split can be made) OR 'MAX' -> larger than all other values (differentiating split can be made). (def: 'Min') ## This can be set for individual features with custom quantization and missing value modes input file. ##

          #'fold_permutation_block_size':1,                         # CPU and GPU; int. Objects in dataset that are grouped in blocks before random permutations. Smaller -> slower training; Large -> quality degradation. (def: 1)

          'leaf_estimation_method':'Gradient',                      # CPU and GPU; string. Method for calculating values in leaves. 'Newton' or 'Gradient'. (def: 'Gradient')

          'leaf_estimation_iterations':10,                          # CPU and GPU; int. Number of gradient steps when calculating the values in leaves. (def: None[Depends on training objective])

          'leaf_estimation_backtracking':'AnyImprovment',          # Depends,string. 'No' -> do not use backtracking. 'AnyImprovement' -> reduce descent step to point where loss func value is smaller than previous step. 'Armijo' [CPU only] Reduce descent step until 'Armijo' condition is met.

          'fold_len_multiplier':2,                                  # CPU and GPU; float; > 1.0. Coeff. for changing lengths of folds. (def: 2) # Best valid. results with min values. #

          'approx_on_full_history':False,                           # CPU only; bool. Principlees of calculating approximated values. 'False' -> Use only a fraction of fold to calc approx values; size of fraction = 1/coeff OR 'True' -> Use all preceding rows in the fold for calc approx values. (def: False)

          #'class_weights':None,                                    # CPU and GPU. Used as multipliers for object weights. alias: 'scale_pos_weight' # For Classification probs #  ## You can set it for imbalanced dataset. ##

          'boosting_type':'Ordered',                                # CPU and GPU[only Plain mode for MultiClass loss on GPU]; string. Boosting scheme. 'Ordered' -> Usually provides better quality on small datasets OR 'Plain' -> The classic gradient boosting scheme. (def: depends on number of data points and learning mode)

          'allow_const_label':False,                                # CPU and GPU; bool. Use with datasets having equal label values for all objects. (def: False)

          

          ####### Overfitting Detection Settings #######

          'early_stopping_rounds':50,                               # CPU and GPU; int. Set overfitting detector type to 'Iter' and stop training after these many iterations after seeing best so far. (def: False)

          #'od_type':'IncToDec',                                    # CPU and GPU; string. 'IncToDec' or 'Iter'. (def: 'IncToDec')

          #'od_pval':0,                                             # CPU and GPU; float; [10^-10, 1-^-2] for best results. Threshold for IncToDec overfitting detector type. Larger -> Overfitting is detected earlier. (def: 0[=off]) # Validation Data should be given #  ## Do not use with Iter overfitting detectot type. ##

          #'od_wait':20,                                            # CPU and GPU; int. Number of iterations after optimal result before stopping. Depends on Overfitting detector type: IncToDec -> Ignore overfitting detector when threshold is reached and countinue learning for specified iterations OR Iter -> like EarlyStoppingRound. (def: 20)

          

          ####### Binarization Settings #######

          'border_count':254,                                       # CPU and GPU; [1, 255]. Number of splits for numerical features. (def: 254 for CPU, 128 for GPU)

          'feature_border_type':'GreedyLogSum',                     # CPU and GPU; string. Binarization mode for numerical feature. 'Meadin', 'Uniform', 'UniformAndQuantile', 'MaxLogSum', 'MinEntropy', 'GreedyLogSum'. (def: 'GreedyLogSum')

          

          ####### Multiclassification settings #######

          #'classes_count':None,                                    # CPU and GPU; int. Upper limit for numeric class label.  ## If this is specified, labels in input data should be smaller than this value. ##

          

          ####### Performance settings #######

          'thread_count':2,                                         # CPU and GPU; int. CPU -> Optimizes speed, GPU -> given value for reading data from hard drive and does not affect training; During training one main thread and one thread for each GPU are used. (def: -1[= number of cores])

          'used_ram_limit':'2gb',                                   # CPU only; int. `Attempt` to limit amount of CPU memory used. # Often affects CTR calc memory usage. MB, KB, GB as '32gb'. # ## In some cases it is impossible to limit RAM usage. ##

          #'gpu_ram_part':0.95,                                     # GPU only; float. How much of GPU RAM to use for training. (def: 0.95)

          #'pinned_memory_size':1073741824,                         # GPU only; int. How much pinned(page-locked) CPU RAM to use per GPU. (def: 1073741824)

          #'gpu_cat_features_storage':None,                         # GPU only; string. Method for storing cat features values. 'CpuPinnedMemory', 'GpuRam'. (def: None[='GPuRam']) # Use 'CpuPinnedMemory' if feature combinations are used and GPU mem. is not sufficient. #

          #'data_partition':None,                                   # GPU only; string. Method for splitting input dataset btw multiple workers. 'FeatureParallel' -> split by features and calc their values on diff GPUs OR 'DocParallel' -> split by objects and calc each of these on certain GPU. (def: depends on learning mode and input datasets)

          

          ####### Processing Unit's settings #######

          'task_type':'CPU',                                        # CPU and GPU; Processing Unit (PU) to use for training. 'CPU' or 'GPU'. (def: 'CPU')

          #'devices':None,                                          # GPU only; string. IDs of the GPU devices to use for training. As "0:2:4-7" (def: None[=all])

          

          ####### Viz settings #######

          'name':"CatBoost Proj",                                   # CPU and GPU; string. Experiment name to display in visualization tools. (def: 'experiment')

          

          ####### Output settings #######

          #'logging_level':None,                                    # CPU and GPU; string. 'Silent', 'Verbose', 'Info', 'Debug'.  (def: None)

          'metric_period':1,                                        # CPU and GPU; int; >0. Freq of iterations to calc the values of obj and metrics. (def: 1)

          'verbose':True,                                           # CPU and GPU; string. 'Silent', 'Verbose', 'Info', 'Debug'

          'train_dir':'catboost_info',                              # CPU and GPU; string. Dir for storing the files generated during training. (def: 'catboost_info')

          'model_size_reg':0.5,                                     # CPU only; [0, inf). Regularize model size. (def: None [=0.5])

          'allow_writing_files':True,                               # CPU only; bool. Allow to write 'analytical' snapshot files during training. (def: True) # Fasle -> snapshot and data viz tools will be unavailable. #

          'save_snapshot':False,                                    # CPU and GPU; bool. To enable snapshotting for restoring the training progress after an interruption. (def: None)

          'snapshot_file':"experiment.cbsnapshot",                  # CPU and GPU; string. Name of file to save training progress in. (def: 'experiment.cbsnapshot')

          'snapshot_interval':600,                                  # CPU and GPU; int. Interval between saving snapshots in seconds. Last snapshot at end of training. (def: 600)

          #'roc_file':None,                                         # CPU and GPU; string. Output file to save ROC curve. (def: None[=File not saved]) ## Can only be used in CV mode with 'LogLoss' loss. ##

          

          ####### CTR settings ########

          #'simple_ctr':None,                                       # CPU and GPU; string. Binarization settings for simple categorical features. As 'CtrType[:TargetBorderCount=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]...'. CtrType: 'Borders', 'Buckets', 'BinarizedTargetMeanValue', 'Counter' (GPU: 'Borders', 'Buckets', 'FeatureFreq', 'FloatTargetMeanValue'); TargetBorderCount: [1, 255]; TargetBorderType: 'Mean', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum', 'MinEntropy', 'GreedyLogSum'; CtrBorderCount: [1, 255]; CtrBorderType: 'Median', 'Uniform', 'UnifromAndQuantiles', 'MaxLogSum', 'GreedyLogSum'; Prior: number (adds value to numerator) (GPU: two slash delimited numbers -> first to num., second to den.)

          #'combinations_ctr':None,                                 # CPU and GPU; string. Binarization settings for simple categorical features. As 'CtrType[:TargetBorderCount=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]...'. CtrType: 'Borders', 'Buckets', 'BinarizedTargetMeanValue', 'Counter' (GPU: 'Borders', 'Buckets', 'FeatureFreq', 'FloatTargetMeanValue'); TargetBorderCount: [1, 255]; TargetBorderType: 'Mean', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum', 'MinEntropy', 'GreedyLogSum'; CtrBorderCount: [1, 255]; CtrBorderType: 'Median', 'Uniform', 'UnifromAndQuantiles', 'MaxLogSum', 'GreedyLogSum'; Prior: number (adds value to numerator) (GPU: two slash delimited numbers -> first to num., second to den.)

          #'per_feature_ctr':None,                                  # CPU and GPU; string. Binarization settings for simple categorical features. As 'CtrType[:TargetBorderCount=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]...'. CtrType: 'Borders', 'Buckets', 'BinarizedTargetMeanValue', 'Counter' (GPU: 'Borders', 'Buckets', 'FeatureFreq', 'FloatTargetMeanValue'); TargetBorderCount: [1, 255]; TargetBorderType: 'Mean', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum', 'MinEntropy', 'GreedyLogSum'; CtrBorderCount: [1, 255]; CtrBorderType: 'Median', 'Uniform', 'UnifromAndQuantiles', 'MaxLogSum', 'GreedyLogSum'; Prior: number (adds value to numerator) (GPU: two slash delimited numbers -> first to num., second to den.)

          #'ctr_target_border_count':8,                             # CPU and GPU; int; [1, 255]. Max number of borders to use in target binarization for categorical features that need it. (def: num_class-1(CPU), 1 otherwise) # Overrides one specified in 'simple_ctr', 'combinations_ctr', 'per_feature_ctr' #

          #'counter_calc_method':None,                              # CPU and GPU; string. For calculating the Counter CTR Type. 'SkipTest' -> Objs from validation not considered OR 'Full' -> All objs from both learn and valid are considered. (def: None[='Full'])

          #'max_ctr_complexity':None,                               # CPU and GPU; int. Max number of categorical featurse that can be combined. (def: 4)

          #'ctr_leaf_count_limit':None,                             # CPU only; int. Max leaves with categorical features. Only leaves with top freq of values are selected. Reduces model size and memory at cost of quality. (def: None[=not limited])

          #'store_all_simple_ctr':None,                             # CPU only; bool. Ignore cat features which are not used in feature combinations, when choosing candidates for exclusion. (def: None[= False]) ## Should be used with 'ctr_leaf_count_limit' ##

          #'final_ctr_computation_mode':None,                       # CPU and GPU; string. Final CTR computation mode. 'Default' -> Compute final CTRs for learn and validation mode OR 'Skip' -> Do not compute final CTRs for learn and validation datasets. In this case the resulting model cannot be applied. Dec.s the size of resulting model. ## Can be useful for research purposes when only metric values have to be calculated. ##

          

          #'silent':None,

          #'device_config':None,                                    # 

          #'dev_score_calc_obj_block_size':None,                    #

          #'metadata':None,                                         # 

          #'cat_features':None,                                     # 

          #'growing_policy':None,                                   #

          #'min_samples_in_leaf':None,                              #

          #'max_leaves_count':None                                  #

}
reg_model = cb.train(

                    pool=train_data,                                            # catboost.Pool or tuple (X, y)

                    params=params,                                              # None or dict.  ## These params overrides all. ##

                    logging_level=None,                                         # 'Silent', 'Verbose', 'Info', 'Debug'

                    verbose=False,                                              # bool or int.

                    iterations=1000,                                            # Number of boosting iterations.

                    eval_set=valid_data,                                        # catboost.Pool or tuple (X, y) or list as [(X1, y1), (X2, y2) ...]

                    plot=True,                                                  # Weather to draw train and eval error in Jupyter Notebook.

                    metric_period=None,                                         # Freq of evaluating metrics.

                    #early_stopping_rounds=50,                                  # Activates 'Iter' overfitting detector with 'od_wait' set to this value.

                    save_snapshot=False,                                        # bool. Enable progress snapshotting.

                    snapshot_file="",                                           # string. Snapshot file path.

                    snapshot_interval=None                                      # int. Interval btw saving snapshots(seconds).

)
reg_model.best_iteration_, reg_model.best_score_
reg_model.feature_importances_
reg_model.evals_result_['validation_0']['RMSE'][-10:]