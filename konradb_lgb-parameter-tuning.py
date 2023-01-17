import numpy as np

import pandas as pd

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold



from bayes_opt import BayesianOptimization

import lightgbm as lgb



import os



from sklearn.cluster import KMeans



from sklearn.neighbors import DistanceMetric



from xgboost import XGBClassifier
# control parameters

nfolds = 5
xtrain = pd.read_csv("../input/train.csv")

xtest = pd.read_csv("../input/test.csv")
# separate the data

id_train = xtrain['ID_code']

ytrain = xtrain['target']

id_test = xtest['ID_code']

xtrain.drop(['ID_code', 'target'], axis = 1, inplace = True)

xtest.drop('ID_code', axis = 1, inplace = True)
folds = KFold(n_splits= nfolds, shuffle=True, random_state= 15)



mindex  = np.zeros((len(xtrain),1) )



for fold_, (trn_idx, val_idx) in enumerate(folds.split(xtrain.values, ytrain.values)):

#    print('----')

    print("fold n°{}".format(fold_))

    

    mindex[val_idx, 0] = fold_



xfolds = pd.DataFrame()

xfolds['MachineIdentifier'] = id_train

xfolds['fold_id']= mindex

xfolds['fold_id'] = xfolds['fold_id'].astype(int).round(0)

xfolds.to_csv('xfolds.csv', index = False)
kmeans = KMeans(init='k-means++', n_clusters= 10, n_init=10)

kmeans.fit(xtrain)
# distance of each observation from cluster centers

dist = DistanceMetric.get_metric('euclidean')

ax_tr = dist.pairwise(xtrain, kmeans.cluster_centers_)

ax_te = dist.pairwise(xtest, kmeans.cluster_centers_)
# format into dataframe

ax_tr = pd.DataFrame(ax_tr)

ax_te = pd.DataFrame(ax_te)

xcols =  ['dist' + str(f) for f in range(0, ax_tr.shape[1])]
ax_tr.columns = xcols

ax_te.columns = xcols
m1 = xtrain.max(axis = 1)

m2 = xtrain.min(axis = 1)

m3 = xtrain.median(axis = 1)

m4 = 1/xtrain.std(axis = 1)

m5 = 1/xtrain.mad(axis = 1)



xtrain['xmax'] = m1; xtrain['xmin'] = m2; xtrain['xmed'] = m3; xtrain['xstd'] = m4



m1 = xtest.max(axis = 1)

m2 = xtest.min(axis = 1)

m3 = xtest.median(axis = 1)

m4 = 1/xtest.std(axis = 1)

m5 = 1/xtest.mad(axis = 1)



xtest['xmax'] = m1; xtest['xmin'] = m2; xtest['xmed'] = m3; xtest['xstd'] = m4

# combine

xtrain = pd.concat([xtrain, ax_tr], axis = 1)

xtest = pd.concat([xtest, ax_te], axis = 1)
xtrain.head(3)
def lgbcv(learning_rate, subsample, min_child_samples, max_depth,

                  colsample_bytree, min_child_weight, min_split_gain, 

                  lambda_l1, lambda_l2,bagging_freq, num_leaves,

                  silent=True, seed=1234):



    params = {                        

            'boosting_type': 'gbdt','objective': 'binary', 'metric':'auc',

            'max_depth': -1, 'num_leaves': int(num_leaves),

            'learning_rate': learning_rate, 'max_depth': int(max_depth),

            'min_child_samples': int(min_child_samples), 

           'subsample': subsample, 'colsample_bytree': colsample_bytree, 'bagging_seed': 11,

           'min_child_weight': min_child_weight,  'bagging_freq' : int(bagging_freq),

           'min_split_gain': min_split_gain,'lambda_l1': lambda_l1,'lambda_l2': lambda_l2,

           'nthread': 8

        }



                

    bst1 = lgb.train(params, trn_data, valid_sets=[trn_data, val_data], valid_names=['train','valid'],

                          num_boost_round= 5000, verbose_eval= 5000, early_stopping_rounds = 100)

    

    ypred = bst1.predict(x1)



    loss = roc_auc_score(y1, ypred)

    return loss
## find optimal params

param_list = list()

score_list = list()





for fold_, (trn_idx, val_idx) in enumerate(folds.split(xtrain.values, ytrain.values)):

    print('----')

    print("fold n°{}".format(fold_))

    

    x0,y0 = xtrain.iloc[trn_idx], ytrain[trn_idx]

    x1,y1 = xtrain.iloc[val_idx], ytrain[val_idx]

    

    

    trn_data = lgb.Dataset(x0, label= y0); val_data = lgb.Dataset(x1, label= y1)

    # optimization

    lgbBO = BayesianOptimization(lgbcv, {'learning_rate': (0.0025, 0.05),'max_depth': (int(5), int(15)),

                                           'min_child_samples': (int(25), int(250)),'subsample': (0.2, 0.95),

                                            'colsample_bytree': (0.2, 0.95), 'min_child_weight': (int(1), int(150)),

                                            'min_split_gain': (0.1, 2),'num_leaves': (int(15),int(200)),

                                             'lambda_l1': (10, 200),'lambda_l2': (10, 200),

                                             'bagging_freq': (1,20)

                                                   })

    lgbBO.maximize(init_points= 25, n_iter= 45, xi=0.06)

    print('-' * 53)

    print('Final Results')

    print('LGB: %f' % lgbBO.res['max']['max_val']);  print('LGB: %s' % lgbBO.res['max']['max_params'])



    param_list.append(lgbBO.res['max']['max_params'])

    score_list.append(lgbBO.res['max']['max_val'])