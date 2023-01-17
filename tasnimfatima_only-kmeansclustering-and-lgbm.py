# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import SVR

import os

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, scale

import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostRegressor

from sklearn.cluster import KMeans, AgglomerativeClustering

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

submit_data = pd.read_csv('../input/sample_submission.csv')

print("Data Loaded")
#train_data = train_data.head(300)

#test_data = test_data.head(300)

K = 700
target = train_data['target']

tar = train_data['target']



features = [c for c in train_data.columns if c not in ['ID_code', 'target']]



print ("Data is ready!")
train_data = train_data.drop(["ID_code"], axis=1)

test_data = test_data.drop(["ID_code"], axis=1)
target0 = train_data[target==0]

target1 = train_data[target==1]
for i in range(0,199):

    ch = 'var_' + str(i)

    target0[ch] = np.random.permutation(target0[ch])

for i in range(0,199):

    ch = 'var_' + str(i)

    target1[ch] = np.random.permutation(target1[ch])
target0 = target0.append(target1)

train_data = train_data.append(target0)

train_data = train_data.sample(frac=1)

target = train_data['target']
train_data = train_data.drop(["target"], axis=1)
train_size = train_data.shape[0]

test_size = test_data.shape[0]
# Merge Train Test Data

train_test = train_data

train_test = train_test.append(test_data)
%%time

train_test = pd.DataFrame(scale(train_test.values), columns=train_test.columns, index=train_test.index)
# clustering

kmeans = KMeans(n_clusters=K)

kmeans.fit(train_test)



train_test['k_labels'] = kmeans.labels_
# Adding Some Weights to the Labels 

temp_df = train_test['k_labels']

train_test['weight1'] = temp_df.apply(lambda x: x*3)

train_test['weight2'] = temp_df.apply(lambda x: x*7)

train_test['weight3'] = temp_df.apply(lambda x: x*11)

train_test['weight4'] = temp_df.apply(lambda x: x*5)

train_test['weight5'] = temp_df.apply(lambda x: x*13)



train_test['weight6'] = temp_df.apply(lambda x: x*17)

train_test['weight7'] = temp_df.apply(lambda x: x*19)

train_test['weight8'] = temp_df.apply(lambda x: x*23)

train_test['weight9'] = temp_df.apply(lambda x: x*29)

train_test['weight10'] = temp_df.apply(lambda x: x*31)



train_test['weight11'] = temp_df.apply(lambda x: x*37)

train_test['weight12'] = temp_df.apply(lambda x: x*41)

train_test['weight13'] = temp_df.apply(lambda x: x*43)

train_test['weight14'] = temp_df.apply(lambda x: x*47)

train_test['weight15'] = temp_df.apply(lambda x: x*53)



train_test['weight16'] = temp_df.apply(lambda x: x*59)

train_test['weight17'] = temp_df.apply(lambda x: x*61)

train_test['weight18'] = temp_df.apply(lambda x: x*67)

train_test['weight19'] = temp_df.apply(lambda x: x*71)

train_test['weight20'] = temp_df.apply(lambda x: x*73)



train_test.head(2)  
cluster_count = pd.Series()

for i in range(0,K):

    print(i , " " , train_test.loc[train_test['k_labels']==i].shape[0])

    temp = (train_test.loc[train_test['k_labels']==i].shape[0])

    cluster_count = cluster_count.append(pd.Series(temp))



clusterCountVal = pd.DataFrame()

clusterCountVal = cluster_count.to_frame()
clusterCountVal.head(2)
train_data = train_test[:test_size]

test_data = train_test[train_size:]
train_data.head(2)
params = {}

params['bagging_freq'] = 5 #reducing it as smaller freq & frac reduce overfitting 

params['bagging_fraction'] = 0.0331

params['random_state'] = 42

params['learning_rate'] = 0.0123

params['boost_from_average'] = False

params['boosting_type'] = 'gbdt'

params['feature_fraction'] = 0.045

params['objective'] = 'binary'

params['metric'] = 'auc'

params['min_data_in_leaf'] = 80

params['num_leaves'] = 13

params['num_threads'] = 8

params['tree_learner'] = 'serial'

params['max_depth'] = -1

params['min_sum_hessian_in_leaf'] = 10.0

params['verbosity'] =  1

params['bagging_seed'] = 42

params['seed'] = 42
num_folds = 10

features = [c for c in test_data.columns if c not in ['ID_code', 'target']]

#print(features)

folds = StratifiedKFold(n_splits=num_folds,shuffle=True, random_state=42)

oof = np.zeros(len(train_data))

getVal = np.zeros(len(train_data))

predictions = np.zeros(len(tar))

print(predictions.shape)

feature_importance_df = pd.DataFrame()
%%time

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_data.values, target.values)):

    

    X_train, y_train = train_data.iloc[trn_idx][features], target.iloc[trn_idx]

    X_valid, y_valid = train_data.iloc[val_idx][features], target.iloc[val_idx]

    

    X_tr, y_tr = X_train.values, y_train.values

    X_tr = pd.DataFrame(X_tr)

    

    print("Fold idx:{}".format(fold_ + 1))

    trn_data = lgb.Dataset(X_tr, label=y_tr)

    val_data = lgb.Dataset(X_valid, label=y_valid)

    

    clf = lgb.train(params, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=5000, early_stopping_rounds = 4000)

    oof[val_idx] = clf.predict(train_data.iloc[val_idx][features], num_iteration=clf.best_iteration)

    getVal[val_idx]+= clf.predict(train_data.iloc[val_idx][features], num_iteration=clf.best_iteration) / folds.n_splits

    



    

    predictions += clf.predict(test_data[features], num_iteration=clf.best_iteration) / folds.n_splits

submit_data['target'] = pd.DataFrame(predictions)

submit_data.to_csv("LGBMwithClustering.csv", index=False)