import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.linear_model import Ridge
import time
from sklearn import preprocessing
import warnings
import datetime
warnings.filterwarnings("ignore")
import gc
from tqdm import tqdm

from scipy.stats import describe
%matplotlib inline

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])
test = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])
print("{} observations and {} features in train set.".format(train.shape[0],train.shape[1]))
print("{} observations and {} features in test set.".format(test.shape[0],test.shape[1]))

train.head()
train.describe()
train["month"] = train["first_active_month"].dt.month
test["month"] = test["first_active_month"].dt.month
train["year"] = train["first_active_month"].dt.year
test["year"] = test["first_active_month"].dt.year
train['elapsed_time'] = (datetime.date(2018, 2, 1) - train['first_active_month'].dt.date).dt.days
test['elapsed_time'] = (datetime.date(2018, 2, 1) - test['first_active_month'].dt.date).dt.days
train.head()
hist_trans = pd.read_csv("../input/historical_transactions.csv")
hist_trans.head()
print("{} transactions and {} columns in trans set.".format(hist_trans.shape[0],hist_trans.shape[1]))
new_trans = pd.read_csv("../input/new_merchant_transactions.csv")
new_trans.head()
print("{} transactions and {} columns in new merchant trans set.".format(new_trans.shape[0],new_trans.shape[1]))
#Target variable
plt.figure(figsize=(12, 5))
plt.hist(train.target.values, bins=100)
plt.title('Target counts')
plt.xlabel('N')
plt.ylabel('Target')
plt.show()
# Feature 1
plt.figure(figsize=(12, 5))
plt.hist(train.feature_1.values, bins=100)
plt.title('Histogram feature_1 counts')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()
#Feature 2
plt.figure(figsize=(12, 5))
plt.hist(train.feature_2.values, bins=100)
plt.title('Histogram feature_2 counts')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()
#Feature3
plt.figure(figsize=(12, 5))
plt.hist(train.feature_3.values, bins=100)
plt.title('Histogram feature_3 counts')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()
train_date_sum = train['first_active_month'].dt.date.value_counts()
train_date_sum = train_date_sum.sort_index()
plt.figure(figsize=(14,8))
sns.barplot(train_date_sum.index,train_date_sum.values,color='red')
plt.xticks(rotation='vertical')
plt.xlabel('first_active_month')
plt.ylabel('No.of records')
plt.title(' Training data - First active month')
plt.show()
test_date_sum = test['first_active_month'].dt.date.value_counts()
test_date_sum = test_date_sum.sort_index()
plt.figure(figsize=(14,8))
sns.barplot(test_date_sum.index,test_date_sum.values,color='red')
plt.xticks(rotation='vertical')
plt.xlabel('first_active_month')
plt.ylabel('No.of records')
plt.title(' Test data - First active month')
plt.show()
hist_trans.head()
trans_count = hist_trans.groupby("card_id")
trans_count = trans_count["purchase_amount"].size().reset_index()
trans_count.columns = ["card_id", "num_trans"]

train = pd.merge(train, trans_count, on="card_id", how="left")
test = pd.merge(test, trans_count, on="card_id", how="left")
trans_summ = hist_trans.groupby("card_id")
trans_summ = trans_summ["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
trans_summ.columns = ["card_id","sum_trans_amt","mean_trans_amt","std_trans_amt","min_trans_amt","max_trans_amt"]

train = pd.merge(train, trans_summ, on="card_id", how="left")
test = pd.merge(test, trans_summ, on="card_id", how="left")
bins = [10,20,50,100,200,500,1000,5000,10000]
train["num_trans_bin"] = pd.cut(train["num_trans"],bins)
#bin_summ = train.groupby("num_trans_bin")["target"].mean()

fig = plt.figure(figsize=(14,7))
sns.boxplot(x = "num_trans_bin", y = "target", data = train,showfliers=False)
plt.show()
bins = np.percentile(train["sum_trans_amt"],range(0,101,10))
train['sum_trans_amt_bin'] = pd.cut(train['sum_trans_amt'], bins)

fig = plt.figure(figsize=(14,7))
sns.boxplot(x = "sum_trans_amt_bin", y = "target", data = train,showfliers=False)
plt.show()
new_trans.head()
trans_count = new_trans.groupby("card_id")
trans_count = trans_count["purchase_amount"].size().reset_index()
trans_count.columns = ["card_id", "new_merch_trans"]

train = pd.merge(train, trans_count, on="card_id", how="left")
test = pd.merge(test, trans_count, on="card_id", how="left")
trans_summ = new_trans.groupby("card_id")
trans_summ = trans_summ["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
trans_summ.columns = ["card_id","new_merch_sum_trans_amt","new_merch_mean_trans_amt","new_merch_std_trans_amt","new_merch_min_trans_amt","new_merch_max_trans_amt"]

train = pd.merge(train, trans_summ, on="card_id", how="left")
test = pd.merge(test, trans_summ, on="card_id", how="left")
train.head()
bins = np.nanpercentile(train["new_merch_sum_trans_amt"],range(0,101,10))
train['new_merch_sum_trans_amt_bin'] = pd.cut(train['new_merch_sum_trans_amt'], bins)

fig = plt.figure(figsize=(14,7))
sns.boxplot(x = "new_merch_sum_trans_amt_bin", y = "target", data = train,showfliers=False)
plt.show()
bins = [10,20,50,100,200,500,1000,5000,10000]
train["new_merch_trans_bin"] = pd.cut(train["new_merch_trans"],bins)
#bin_summ = train.groupby("num_trans_bin")["target"].mean()

fig = plt.figure(figsize=(14,7))
sns.boxplot(x = "new_merch_trans_bin", y = "target", data = train,showfliers=False)
plt.show()
train.info()
print(test.columns)
from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb

from sklearn.ensemble import *
import xgboost as xgb
cols = [ 'feature_1', 'feature_2', 'feature_3',
       'month', 'year', 'elapsed_time', 'num_trans', 'sum_trans_amt',
       'mean_trans_amt', 'std_trans_amt', 'min_trans_amt', 'max_trans_amt',
       'new_merch_trans', 'new_merch_sum_trans_amt',
       'new_merch_mean_trans_amt', 'new_merch_std_trans_amt',
       'new_merch_min_trans_amt', 'new_merch_max_trans_amt']

target_col = ['target']


def run_basemodel(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 30,
        "min_child_weight" : 50,
        "learning_rate" : 0.05,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100, evals_result=evals_result)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result


train_X = train[cols]
test_X = test[cols]
train_y = train[target_col].values
train_y = train_y.ravel()

pred_test = 0
kf = model_selection.KFold(n_splits=5, random_state=2018, shuffle=True)
for dev_index, val_index in kf.split(train):
    dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    #print(val_X)
    
    pred_test_tmp, model, evals_result = run_basemodel(dev_X, dev_y, val_X, val_y, test_X)
    pred_test += pred_test_tmp
pred_test /= 5.
    
fig, ax = plt.subplots(figsize=(12,10))
lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()
sub_df = pd.DataFrame({"card_id":test["card_id"].values})
sub_df["target"] = pred_test
sub_df.to_csv("submission.csv", index=False)
