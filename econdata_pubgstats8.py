# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import time
notebookstart= time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Jupyter Specific Packages
import matplotlib.pyplot as plt
import seaborn as sns
import shap
%matplotlib inline
from sklearn.metrics import mean_squared_error
import math
from IPython.display import display

# Gradient Boosting
import lightgbm as lgb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv("../input/train_V2.csv")#.sample(20000)
test = pd.read_csv("../input/test_V2.csv")#.sample(20000)
train.head()
for col in ["matchId","Id","groupId"]:
    print("Does {} Feature Overlap Between Train/Test Set?         {}".format(col, any(np.intersect1d(test[col].unique(), train[col].unique()))))
matchcount = train.matchId.nunique()
print("Number of unique matches: {}".format(train.matchId.nunique()))
print("Train Shape Before: {} Rows, {} Cols".format(*train.shape))
train = train.loc[train.matchId.isin(sorted(train.matchId.unique())[int(matchcount* 0.50):]),:]
print("Train Shape After: {} Rows, {} Cols".format(*train.shape))
# Label Encoder
from sklearn import preprocessing
# Encoder:
lbl = preprocessing.LabelEncoder()
for col in ['matchType']:
    lbl.fit(train[col])
    train[col] = lbl.transform(train[col])
    test[col] = lbl.transform(test[col])
id_cols = ["Id","groupId","matchId"]
exclude = ["Id","groupId","matchId"]
trainlen = train.shape[0]
# LGBM Dataset
matchcount = train.matchId.nunique()

training = train.loc[train.matchId.isin(sorted(train.matchId.unique())[:int(matchcount* 0.85)]),
                    [x for x in train.columns if x not in exclude]]
print("Training Shape: {} Rows, {} Cols".format(*training.shape))
validating = train.loc[train.matchId.isin(sorted(train.matchId.unique())[int(matchcount* 0.85):]),
                       [x for x in train.columns if x not in exclude]]
print("Validating Shape: {} Rows, {} Cols".format(*validating.shape))

train_y = training.winPlacePerc
training.drop("winPlacePerc", axis =1, inplace=True)
valid_y = validating.winPlacePerc
validating.drop("winPlacePerc", axis =1, inplace=True)
                                                             
lgb_train = lgb.Dataset(training, train_y,feature_name = "auto")
lgb_valid = lgb.Dataset(validating, valid_y, feature_name = "auto")
del training, validating
print("Light Gradient Boosting Regressor: ")
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'num_boost_round' : 5000
#     "learning_rate": 0.01,
#     "num_leaves": 180,
#     "feature_fraction": 0.50,
#     "bagging_fraction": 0.50,
#     'bagging_freq': 4,
#     "max_depth": -1,
#     "reg_alpha": 0.3,
#     "reg_lambda": 0.1,
#     #"min_split_gain":0.2,
#     "min_child_weight":10,
#     'zero_as_missing':True
                }
stage = 'model training'
gbm = lgb.train(lgbm_params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=[lgb_train, lgb_valid],
                feature_name='auto',
                early_stopping_rounds=50,
                verbose_eval=250
                )

# Feature Importance Plot
f, ax = plt.subplots(figsize=[7,10])
lgb.plot_importance(gbm, max_num_features=25, ax=ax)
plt.title("Light GBM Feature Importance\n")
plt.show()
pred = gbm.predict(test.loc[:,[x for x in test.columns if x not in id_cols]])
test['winPlacePercPred'] = np.clip(pred, a_min=0, a_max=1)

aux = test.groupby(['matchId','groupId'])['winPlacePercPred'].agg('mean').groupby('matchId').rank(pct=True).reset_index()
aux.columns = ['matchId','groupId','winPlacePerc']
test_sub = test.merge(aux, how='left', on=['matchId','groupId'])
    
submission = test_sub[['Id', 'winPlacePerc']]
submission.to_csv('PubGG_LGBM.csv', index=False)
display(submission.head())
notcat = ["assists","boosts","damageDealt","DBNOs","heals","headshotKills","heals","killPlace","killPoints","kills",
         "killStreaks","longestKill","maxPlace","numGroups","revives","rideDistance","roadKills","swimDistance",
         "teamKills","vehicleDestroys","walkDistance","weaponsAcquired","winPoints"]
y = train.winPlacePerc.copy()
train.drop("winPlacePerc",axis =1, inplace=True)
def agg_dataset(dataset):
    for id_col in id_cols:
        agg_features = dataset.groupby(id_col).agg({k:["sum","mean","std"] for k in
                                ["killPlace","walkDistance","numGroups","maxPlace","kills","longestKill","weaponsAcquired"]})
        agg_features.columns = pd.Index(["{}_agg_".format(id_col) + e[0] +"_"+ e[1] for e in agg_features.columns.tolist()])
        dataset = pd.merge(dataset,agg_features, on = id_col, how= "left")
    return dataset
train = agg_dataset(train)
test = agg_dataset(test)
# Remove Columns with 95%+ Missing
missing = round(train.isnull().sum()/ train.shape[0]*100).reset_index().rename({"index":"columns",0:"missing"}, axis =1 )
high_missing_columns = missing.loc[missing.missing > 65, "columns"]
print("Columns to remove (65% missing Values and Over)\n", list(high_missing_columns))
train.drop(high_missing_columns,axis =1, inplace= True)
test.drop(high_missing_columns,axis =1, inplace= True)
train = pd.concat([train.reset_index(drop=True), y.reset_index(drop=True)], axis=1)

training = train.loc[train.matchId.isin(sorted(train.matchId.unique())[:int(matchcount* 0.85)]),
                    [x for x in train.columns if x not in exclude]]
print("Training Shape: {} Rows, {} Cols".format(*training.shape))
validating = train.loc[train.matchId.isin(sorted(train.matchId.unique())[int(matchcount* 0.85):]),
                       [x for x in train.columns if x not in exclude]]
print("Validating Shape: {} Rows, {} Cols".format(*validating.shape))

train_y = training.winPlacePerc
training.drop("winPlacePerc", axis =1, inplace=True)
valid_y = validating.winPlacePerc
validating.drop("winPlacePerc", axis =1, inplace=True)
                                                             
lgb_train = lgb.Dataset(training, train_y,feature_name = "auto")
lgb_valid = lgb.Dataset(validating, valid_y, feature_name = "auto")
# del training, validating
print("Light Gradient Boosting Regressor: ")
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'num_boost_round' : 5000
#     "learning_rate": 0.01,
#     "num_leaves": 180,
#     "feature_fraction": 0.50,
#     "bagging_fraction": 0.50,
#     'bagging_freq': 4,
#     "max_depth": -1,
#     "reg_alpha": 0.3,
#     "reg_lambda": 0.1,
#     #"min_split_gain":0.2,
#     "min_child_weight":10,
#     'zero_as_missing':True
                }

stage = 'model training'
gbm = lgb.train(lgbm_params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=[lgb_train, lgb_valid],
                feature_name='auto',
                early_stopping_rounds=50,
                verbose_eval=250
                )

# Feature Importance Plot
f, ax = plt.subplots(figsize=[7,10])
lgb.plot_importance(gbm, max_num_features=25, ax=ax)
plt.title("Light GBM Feature Importance\n")
plt.show()
pred = gbm.predict(test.loc[:,[x for x in test.columns if x not in id_cols]])
test['winPlacePercPred'] = np.clip(pred, a_min=0, a_max=1)

aux = test.groupby(['matchId','groupId'])['winPlacePercPred'].agg('mean').groupby('matchId').rank(pct=True).reset_index()
aux.columns = ['matchId','groupId','winPlacePerc']
test_sub = test.merge(aux, how='left', on=['matchId','groupId'])
    
submission = test_sub[['Id', 'winPlacePerc']]
submission.to_csv('AGG_PubGG_LGBM.csv', index=False)
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))
submission.head()
