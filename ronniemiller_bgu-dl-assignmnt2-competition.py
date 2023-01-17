# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_set = pd.read_csv("../input/bgu-dl-assignmnt2-features-extraction/train_set.csv")
test_set = pd.read_csv("../input/bgu-dl-assignmnt2-features-extraction/test_set.csv")
target = pd.read_csv("../input/bgu-dl-assignmnt2-features-extraction/target.csv", header=None)

print("shape of train : ",train_set.shape)
print("shape of test : ",test_set.shape)
print("shape of target : ",target.shape)
train_set.head()
cat_col = ['feature_1','feature_2', 'feature_3', 'merchant_group_id', 'merchant_category_id', 'subsector_id', 'category_1',
          'most_recent_sales_range', 'most_recent_purchases_range', 'category_4', 'city_id', 'state_id', 'category_2']
numeric_col = train_set.columns[~train_set.columns.isin(np.append(cat_col, ['card_id', 'first_active_month']))]
used_col = np.concatenate((cat_col, numeric_col), axis=0)
for col in ['category_1', 'most_recent_sales_range', 'most_recent_purchases_range', 'category_4']:    
    lbl = LabelEncoder()
    lbl.fit(train_set[col].unique().astype('str'))
    train_set[col] = lbl.transform(train_set[col].astype('str'))
    test_set[col] = lbl.transform(test_set[col].astype('str'))
train_set = train_set.fillna(-20)
test_set = test_set.fillna(-20)
lgb_params = {'num_leaves': 111,
             'min_data_in_leaf': 149, 
             'objective':'regression',
             'max_depth': 9,
             'learning_rate': 0.005,
             "boosting": "gbdt",
             "feature_fraction": 0.7522,
             "bagging_freq": 1,
             "bagging_fraction": 0.7083 ,
             "bagging_seed": 11,
             "metric": 'rmse',
             "lambda_l1": 0.2634,
             "random_state": 133,
             "verbosity": -1}

folds = KFold(n_splits=5, shuffle=True, random_state=24)
train_predictions_lgb = np.zeros(len(train_set))
predictions_lgb = np.zeros(len(test_set))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_set.values, target.values)):
    print("lgb " + str(fold_) + "-" * 50)
    train_data_lgb = lgb.Dataset(train_set.iloc[trn_idx][used_col],
                           label=target.iloc[trn_idx],
                           categorical_feature=['feature_1','feature_2', 'feature_3'])
    val_data_lgb = lgb.Dataset(train_set.iloc[val_idx][used_col],
                           label=target.iloc[val_idx],
                           categorical_feature=['feature_1','feature_2', 'feature_3'])

    num_round = 10000
    lgb_model = lgb.train(lgb_params,
                    train_data_lgb,
                    num_round,
                    valid_sets = [train_data_lgb, val_data_lgb],
                    verbose_eval=100,
                    early_stopping_rounds = 200)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = used_col
    fold_importance_df["importance"] = lgb_model.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    train_predictions_lgb[val_idx] = lgb_model.predict(train_set.iloc[val_idx][used_col], num_iteration=lgb_model.best_iteration)
    
    predictions_lgb += lgb_model.predict(test_set[used_col], num_iteration=lgb_model.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(train_predictions_lgb, target)**0.5))
sub_df = pd.DataFrame({"card_id":test_set["card_id"].values})
sub_df["target"] = predictions_lgb
sub_df.to_csv("submit_lgb_1.csv", index=False)
cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,25))
sns.barplot(x="importance",
            y="feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))

plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')
sort_best_fatures = best_features.sort_values(by="importance",ascending=False)
sort_best_fatures.drop_duplicates(subset=['feature'], inplace=True)
top_feature = sort_best_fatures.iloc[:138,0].values
lgb_params = {'num_leaves': 111,
             'min_data_in_leaf': 149, 
             'objective':'regression',
             'max_depth': 9,
             'learning_rate': 0.005,
             "boosting": "gbdt",
             "feature_fraction": 0.7522,
             "bagging_freq": 1,
             "bagging_fraction": 0.7083 ,
             "bagging_seed": 11,
             "metric": 'rmse',
             "lambda_l1": 0.2634,
             "random_state": 133,
             "verbosity": -1}

folds = KFold(n_splits=5, shuffle=True, random_state=24)
train_predictions_lgb = np.zeros(len(train_set))
predictions_lgb = np.zeros(len(test_set))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_set.values, target.values)):
    print("lgb " + str(fold_) + "-" * 50)
    train_data_lgb = lgb.Dataset(train_set.iloc[trn_idx][top_feature],
                           label=target.iloc[trn_idx],
                           categorical_feature=['feature_1'])
    val_data_lgb = lgb.Dataset(train_set.iloc[val_idx][top_feature],
                           label=target.iloc[val_idx],
                           categorical_feature=['feature_1'])

    num_round = 20000
    lgb_model = lgb.train(lgb_params,
                    train_data_lgb,
                    num_round,
                    valid_sets = [train_data_lgb, val_data_lgb],
                    verbose_eval=100,
                    early_stopping_rounds = 200)
    
    train_predictions_lgb[val_idx] = lgb_model.predict(train_set.iloc[val_idx][top_feature], num_iteration=lgb_model.best_iteration)
    
    predictions_lgb += lgb_model.predict(test_set[top_feature], num_iteration=lgb_model.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(train_predictions_lgb, target)**0.5))
sub_df = pd.DataFrame({"card_id":test_set["card_id"].values})
sub_df["target"] = predictions_lgb
sub_df.to_csv("submit_lgb_2.csv", index=False)