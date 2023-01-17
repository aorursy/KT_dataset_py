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
'''
train = train.drop(train[train.target<-30].index)
plt.figure(figsize=(12,8))
sns.distplot(train[target_col].values, bins=50, kde=False, color="red")
plt.title("Histogram of Loyalty score")
plt.xlabel('Loyalty score', fontsize=12)
plt.show()
'''
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
print(train.feature_1.unique())
# feature 1
plt.figure(figsize=(16,8))
sns.boxplot(x="feature_1", y=train.target, data=train)
plt.xticks(rotation='vertical')
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Feature 1 distribution")
plt.show()
print(train.feature_2.unique())
# feature 2
plt.figure(figsize=(16,8))
sns.boxplot(x="feature_2", y=train.target, data=train)
plt.xticks(rotation='vertical')
plt.xlabel('Feature 2', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Feature 2 distribution")
plt.show()
print(train.feature_3.unique())
# feature 3
plt.figure(figsize=(16,8))
sns.boxplot(x="feature_3", y=train.target, data=train)
plt.xticks(rotation='vertical')
plt.xlabel('Feature 3', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Feature 3 distribution")
plt.show()
hist = pd.read_csv('../input/historical_transactions.csv')
hist.head(5)
hist.dtypes
# Number of historical transactions for each card_id
gdf = hist.groupby('card_id')
#print(gdf.head(5))

gdf = gdf['purchase_amount'].size().reset_index()
print(gdf.head(5))

gdf.columns = ['card_id', 'num_hist_transactions']
train = pd.merge(train, gdf, on='card_id', how='left')
test = pd.merge(test, gdf, on='card_id', how='left')
#data = pd.merge(data, gdf, on='card_id', how='left')
print(train.head(5))
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
gdf = new_trans.groupby("card_id")
gdf = gdf["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
gdf.columns = ["card_id", "sum_merch_trans", "mean_merch_trans", "std_merch_trans", 
               "min_merch_trans", "max_merch_trans"]
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
from xgboost.sklearn import XGBRegressor
import lightgbm as lgb

#modelname = 'lightgbm'
modelname = 'XGBRegressor'
#model = lgb()
model = XGBRegressor()

params_lgb = {
    'num_leaves': 100,
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
    "verbosity": -1
}


params_xgbr = {
    'nthread':[4], #when use hyperthread, xgboost may become slower
    'objective':['reg:linear'],
    'learning_rate': [.03, 0.05, .07], #so called `eta` value
    'max_depth': [5, 6, 7],
    'min_child_weight': [4],
    'silent': [1],
    'subsample': [0.7],
    'colsample_bytree': [0.7],
    'n_estimators': [500]
}
from sklearn.model_selection import RandomizedSearchCV

start = time()
rf_random = RandomizedSearchCV(estimator = model, param_distributions = params_xgbr, 
                               n_iter = 2, cv = 3, verbose = 2, random_state = 42, n_jobs = -1)

rf_random.fit(train_X, train_y)
# print(time() - start)
print('Total of %.5f seconds' % (time() - start))
print(rf_random.cv_results_)
final_model = rf_random.best_estimator_
predictions = final_model.predict(test_X)
print(predictions[:5])
test_X.info(5)
test_X.shape
predictions.shape
df_submission = pd.DataFrame({'card_id':test['card_id']})
df_submission["target"] = predictions
df_submission.to_csv("newton.csv", index=False)