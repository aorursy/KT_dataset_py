# Load in libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn import model_selection, preprocessing, metrics
color = sns.color_palette()
# Check for Correct Files
print(os.listdir('../input'))
# Read in datasets to use

train = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])
test = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])
hist = pd.read_csv("../input/historical_transactions.csv")
new_trans = pd.read_csv("../input/new_merchant_transactions.csv")
hist.head()

# Find most frequent merchant used by each cardID

cats = hist['merchant_category_id'].unique()


freq_merch = hist.groupby("card_id")
freq_merch = freq_merch['merchant_category_id'] \
            .agg(lambda x: x.value_counts().index[0]) \
            .reset_index()

freq_merch['merchant_category_id'] = freq_merch['merchant_category_id'] \
                                    .astype('category', categories = cats)

freq_merch['merchant_category_id'].dtype
# Find total number of individual purchases
num_purch = hist.groupby('card_id')
num_purch = num_purch['purchase_amount'] \
            .agg(lambda x: len(x)) \
            .reset_index()
num_purch.head()
# Merge and aggregate DFs


gdf = hist.groupby("card_id")
gdf = gdf["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()

gdf = pd.merge(gdf, freq_merch, on = 'card_id', how = 'left')
gdf = pd.merge(gdf, num_purch, on = 'card_id', how = 'left')
gdf.columns = ["card_id", "sum_hist_trans", "mean_hist_trans",
              "std_hist_trans", "min_hist_trans", "max_hist_trans", 'freq_merch',
              'num_purchases']

train = pd.merge(train, gdf, on="card_id", how="left")
test = pd.merge(test, gdf, on="card_id", how="left")

train.head()
len(train['freq_merch'].unique())
# Freq Merch boxplot
plt.figure(figsize=(12,8))
sns.boxplot(x="freq_merch", y='target', data=train, showfliers=False)
plt.xticks(rotation='vertical')
plt.xlabel('Most Common Merchant', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Frequent Merchants")
plt.show()
# Number new merchant transactions

gdf = new_trans.groupby("card_id")
gdf = gdf["purchase_amount"].size().reset_index()
gdf.columns = ["card_id", "num_merch_transactions"]
train = pd.merge(train, gdf, on="card_id", how="left")
test = pd.merge(test, gdf, on="card_id", how="left")

# Aggregate new merchant transactions 

gdf = new_trans.groupby("card_id")
gdf = gdf["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()

gdf.columns = ["card_id", "sum_merch_trans", "mean_merch_trans",
               "std_merch_trans", "min_merch_trans", "max_merch_trans"]

train = pd.merge(train, gdf, on="card_id", how="left")
test = pd.merge(test, gdf, on="card_id", how="left")


train.head()
# Final Prep before baseline model

train["year"] = train["first_active_month"].dt.year
test["year"] = test["first_active_month"].dt.year
train["month"] = train["first_active_month"].dt.month
test["month"] = test["first_active_month"].dt.month

cols_to_use = ["feature_1", "feature_2", "feature_3", "year", "month", "sum_hist_trans",
               "mean_hist_trans", "std_hist_trans", "min_hist_trans", "max_hist_trans",
               'freq_merch', 'num_purchases', "num_merch_transactions",
               "sum_merch_trans", "mean_merch_trans", "std_merch_trans",
               "min_merch_trans", "max_merch_trans"]
# Define LGBM function
def run_lgb(train_X, train_y, val_X, val_y, test_X):
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
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100,
                      verbose_eval=100, evals_result=evals_result)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result

train_X = train[cols_to_use]
test_X = test[cols_to_use]
train_y = train['target'].values


pred_test = 0
kf = model_selection.KFold(n_splits=5, random_state=2018, shuffle=True)
for dev_index, val_index in kf.split(train):
    dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    
    pred_test_tmp, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)
    pred_test += pred_test_tmp
pred_test /= 5.
fig, ax = plt.subplots(figsize=(12,10))
lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()
# Submit
sub_df = pd.DataFrame({"card_id":test["card_id"].values})
sub_df["target"] = pred_test
sub_df.to_csv("baseline_lgb.csv", index=False)
