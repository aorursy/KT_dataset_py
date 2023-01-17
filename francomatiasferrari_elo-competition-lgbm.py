!ls ../input
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import sys

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from time import time
# Remember to change directory path
train = pd.read_csv("../input/train.csv", parse_dates=['first_active_month'])
test = pd.read_csv("../input/test.csv", parse_dates=['first_active_month'])
print(train.shape)
print(test.shape)
data = pd.concat([train,test])
print(data.head(5))
data.dtypes
data.describe()
target_col = "target"

plt.figure(figsize=(8,6))
plt.scatter(range(train.shape[0]), np.sort(train[target_col].values))
plt.xlabel('index', fontsize=12)
plt.ylabel('Loyalty Score', fontsize=12)
plt.show()
plt.figure(figsize=(12,8))
sns.distplot(train[target_col].values, bins=50, kde=False, color="red")
plt.title("Histogram of Loyalty score")
plt.xlabel('Loyalty score', fontsize=12)
plt.show()
(train['target']<-30).sum()
cnt_srs_1 = train['first_active_month'].dt.date.value_counts()
cnt_srs_1 = cnt_srs_1.sort_index()
cnt_srs_2 = test['first_active_month'].dt.date.value_counts()
cnt_srs_2 = cnt_srs_2.sort_index()

sns.set(rc={'figure.figsize':(14, 6)})
sns.barplot(cnt_srs_1.index, cnt_srs_1.values, alpha = 0.5, color = 'green')
sns.barplot(cnt_srs_2.index, cnt_srs_2.values, alpha = 0.5, color = 'red')
#plt.bar(cnt_srs_1.index, cnt_srs_1.values, alpha = 0.5, color = 'green')
#plt.bar(cnt_srs_2.index, cnt_srs_2.values, alpha = 0.5, color = 'red')

plt.xticks(rotation = 'vertical')
#plt.xlabel('First active month', fontsize=12)
#plt.ylabel('Number of cards', fontsize=12)
#plt.title("First active month count in train set")

plt.show()
print(data.feature_1.unique())
# feature 1
plt.figure(figsize=(16,8))
sns.boxplot(x="feature_1", y=data.target, data=data)
plt.xticks(rotation='vertical')
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Feature 1 distribution")
plt.show()
print(data.feature_2.unique())
# feature 2
plt.figure(figsize=(16,8))
sns.boxplot(x="feature_2", y=data.target, data=data)
plt.xticks(rotation='vertical')
plt.xlabel('Feature 2', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Feature 2 distribution")
plt.show()
print(data.feature_3.unique())
# feature 3
plt.figure(figsize=(16,8))
sns.boxplot(x="feature_3", y=data.target, data=data)
plt.xticks(rotation='vertical')
plt.xlabel('Feature 3', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Feature 3 distribution")
plt.show()
hist = pd.read_csv('../input/historical_transactions.csv')
hist.head(5)
hist.dtypes
# Number of historical transactiones for each card_id
gdf = hist.groupby('card_id')
#print(gdf.head(5))

gdf = gdf['purchase_amount'].size().reset_index()
print(gdf.head(5))

gdf.columns = ['card_id', 'num_hist_transactions']
train = pd.merge(train, gdf, on='card_id', how='left')
test = pd.merge(test, gdf, on='card_id', how='left')
data = pd.merge(data, gdf, on='card_id', how='left')
print(data.head(5))
bins = [0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 500, 10000]
train['binned_num_hist_transactions'] = pd.cut(train['num_hist_transactions'], bins)
cnt_srs = train.groupby("binned_num_hist_transactions")['target'].mean()

plt.figure(figsize=(12,8))
sns.boxplot(x="binned_num_hist_transactions", y='target', data=train, showfliers=False)
plt.xticks(rotation='vertical')
plt.xlabel('binned_num_hist_transactions', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("binned_num_hist_transactions distribution")
plt.show()
gdf = hist.groupby("card_id")
gdf = gdf["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
gdf.columns = ["card_id", "sum_hist_trans", "mean_hist_trans", "std_hist_trans", "min_hist_trans", "max_hist_trans"]
train = pd.merge(train, gdf, on="card_id", how="left")
test = pd.merge(test, gdf, on="card_id", how="left")
bins = np.percentile(train["sum_hist_trans"], range(0,101,10))
train['binned_sum_hist_trans'] = pd.cut(train['sum_hist_trans'], bins)
#cnt_srs = train_df.groupby("binned_sum_hist_trans")[target_col].mean()

plt.figure(figsize=(12,8))
sns.boxplot(x="binned_sum_hist_trans", y='target', data=train, showfliers=False)
plt.xticks(rotation='vertical')
plt.xlabel('binned_sum_hist_trans', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Sum of historical transaction value (Binned) distribution")
plt.show()
gdf = hist.groupby("card_id")
gdf = gdf["installments"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
gdf.columns = ["card_id", "sum_hist_installments", "mean_hist_installments", "std_hist_installments", 
               "min_hist_installments", "max_hist_installments"]
train = pd.merge(train, gdf, on="card_id", how="left")
test = pd.merge(test, gdf, on="card_id", how="left")
bins = np.percentile(train["sum_hist_installments"], range(0,101,10))
train['binned_sum_hist_installments'] = pd.cut(train['sum_hist_installments'], bins, duplicates = 'drop')
#cnt_srs = train_df.groupby("binned_sum_hist_trans")[target_col].mean()

plt.figure(figsize=(12,8))
sns.boxplot(x="binned_sum_hist_installments", y='target', data=train, showfliers=False)
plt.xticks(rotation='vertical')
plt.xlabel('binned_sum_hist_installments', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Sum of historical transaction installments (Binned) distribution")
plt.show()
new_trans = pd.read_csv("../input/new_merchant_transactions.csv")
new_trans.head(5)
new_trans.dtypes
gdf = new_trans.groupby("card_id")
gdf = gdf["purchase_amount"].size().reset_index()
gdf.columns = ["card_id", "num_merch_transactions"]
train = pd.merge(train, gdf, on="card_id", how="left")
test = pd.merge(test, gdf, on="card_id", how="left")
bins = [0, 10, 20, 30, 40, 50, 75, 10000]
train['binned_num_merch_transactions'] = pd.cut(train['num_merch_transactions'], bins)
cnt_srs = train.groupby("binned_num_merch_transactions")['target'].mean()

plt.figure(figsize=(12,8))
sns.boxplot(x="binned_num_merch_transactions", y='target', data=train, showfliers=False)
plt.xticks(rotation='vertical')
plt.xlabel('binned_num_merch_transactions', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Number of new merchants transaction (Binned) distribution")
plt.show()
gdf = new_trans.groupby("card_id")
gdf = gdf["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
gdf.columns = ["card_id", "sum_merch_trans", "mean_merch_trans", "std_merch_trans", "min_merch_trans", "max_merch_trans"]
train = pd.merge(train, gdf, on="card_id", how="left")
test = pd.merge(test, gdf, on="card_id", how="left")
bins = np.nanpercentile(train["sum_merch_trans"], range(0,101,10))
train['binned_sum_merch_trans'] = pd.cut(train['sum_merch_trans'], bins)
#cnt_srs = train_df.groupby("binned_sum_hist_trans")[target_col].mean()

plt.figure(figsize=(12,8))
sns.boxplot(x="binned_sum_merch_trans", y='target', data=train, showfliers=False)
plt.xticks(rotation='vertical')
plt.xlabel('binned sum of new merchant transactions', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Sum of New merchants transaction value (Binned) distribution")
plt.show()
gdf = new_trans.groupby("card_id")
gdf = gdf["installments"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
gdf.columns = ["card_id", "sum_merch_installments", "mean_merch_installments", "std_merch_installments", 
               "min_merch_installments", "max_merch_installments"]
train = pd.merge(train, gdf, on="card_id", how="left")
test = pd.merge(test, gdf, on="card_id", how="left")
bins = np.nanpercentile(train["sum_merch_installments"], range(0,101,10))
train['binned_sum_merch_installments'] = pd.cut(train['sum_merch_installments'], bins, duplicates = 'drop')
#cnt_srs = train_df.groupby("binned_sum_hist_trans")[target_col].mean()

plt.figure(figsize=(12,8))
sns.boxplot(x="binned_sum_merch_installments", y='target', data=train, showfliers=False)
plt.xticks(rotation='vertical')
plt.xlabel('binned sum of new merchant transactions installments', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Sum of New merchants transaction installments value (Binned) distribution")
plt.show()
train["year"] = train["first_active_month"].dt.year
train["month"] = train["first_active_month"].dt.month
test["month"] = test["first_active_month"].dt.month
test["year"] = test["first_active_month"].dt.year

# data['year'] = data['first_active_month'].dt.year
# data['month'] = data['first_active_month'].dt.month

cols_to_use = ["feature_1", "feature_2", "feature_3", "year", "month", 
               "num_hist_transactions", "sum_hist_trans", "mean_hist_trans", "std_hist_trans", 
               "min_hist_trans", "max_hist_trans",
               "sum_hist_installments", "mean_hist_installments", "std_hist_installments", 
               "min_hist_installments", "max_hist_installments",            
               "num_merch_transactions", "sum_merch_trans", "mean_merch_trans", "std_merch_trans",
               "min_merch_trans", "max_merch_trans",
               "sum_merch_installments", "mean_merch_installments", "std_merch_installments",
               "min_merch_installments", "max_merch_installments",
              ]


train_X = train[cols_to_use]
train_y = train['target'].values
test_X = test[cols_to_use]
print(train_X.shape[0])
print(train_y.shape[0])
# checking minable view: get into consideration there are missing values 
# for indicators coming from new_merchants_transactions file.
print(train_X.head(5))
print(train_X.info(5))
print(train_X.isnull().sum())
print(train_y[:5])
print(np.info(train_y))
print(np.isnan(train_y).sum())
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold
import warnings
import time
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

param = {'num_leaves': 100,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': 6,
         'learning_rate': 0.005,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1}
folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(train[cols_to_use]))
predictions = np.zeros(len(test[cols_to_use]))
start = time.time()
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train[cols_to_use].values, train['target'].values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][cols_to_use], label=train['target'].iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx][cols_to_use], label=train['target'].iloc[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)
    oof[val_idx] = clf.predict(train.iloc[val_idx][cols_to_use], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = cols_to_use
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test[cols_to_use], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(oof, train['target'])**0.5))
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
sub_df = pd.DataFrame({"card_id":test["card_id"].values})
sub_df["target"] = predictions
sub_df.to_csv("submit.csv", index=False)