import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb
!ls ../input/
train = pd.read_csv("../input/train.csv")
train.head()
train["card_id"].head()
card_id_groupby_count = train.groupby("card_id")["target","first_active_month"].agg(["count","size"]).reset_index()
card_id_groupby_count.columns = pd.Index(["_".join(col) for col in card_id_groupby_count.columns.tolist()])
card_id_groupby_count.head()
card_id_groupby_count[card_id_groupby_count["target_count"] > 1].count()
card_id_groupby_count[card_id_groupby_count["target_size"] > 1].count()
card_id_groupby_count[card_id_groupby_count["first_active_month_count"] > 1].count()
card_id_groupby_count[card_id_groupby_count["first_active_month_size"] > 1].count()
test = pd.read_csv("../input/test.csv")
test.head()
card_id_test_groupby_count = test.groupby("card_id")["first_active_month"].agg(["count","size"]).reset_index()
card_id_test_groupby_count[card_id_test_groupby_count["count"] > 1].count()
card_id_test_groupby_count[card_id_test_groupby_count["size"] > 1].count()
train["first_active_month"] = pd.to_datetime(train["first_active_month"])
test["first_active_month"] = pd.to_datetime(test["first_active_month"])
cnt_srs = train['first_active_month'].dt.date.value_counts()
cnt_srs = cnt_srs.sort_index()
plt.figure(figsize=(14,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')
plt.xticks(rotation='vertical')
plt.xlabel('First active month', fontsize=12)
plt.ylabel('Number of cards', fontsize=12)
plt.title("First active month count in train set")
plt.show()

cnt_srs = test['first_active_month'].dt.date.value_counts()
cnt_srs = cnt_srs.sort_index()
plt.figure(figsize=(14,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')
plt.xticks(rotation='vertical')
plt.xlabel('First active month', fontsize=12)
plt.ylabel('Number of cards', fontsize=12)
plt.title("First active month count in test set")
plt.show()


train["target"].describe()
plt.figure(figsize=(8,6))
plt.scatter(range(train.shape[0]), np.sort(train["target"].values))
plt.xlabel('index', fontsize=12)
plt.ylabel('Loyalty Score', fontsize=12)
plt.show()
plt.figure(figsize=(12,8))
sns.distplot(train["target"],bins=50, kde=False, color="red")
plt.title("Histogram of Loyalty score")
plt.xlabel('Loyalty score', fontsize=12)
plt.show()
def plotViolin(feature):
    plt.figure(figsize=(8,4))
    sns.violinplot(x=feature, y="target", data=train)
    plt.xticks(rotation='vertical')
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Loyalty score', fontsize=12)
    plt.title("Feature 1 distribution")
    plt.show()

plotViolin("feature_1")
plotViolin("feature_2")
plotViolin("feature_3")
sns.distplot(train["feature_1"],kde=False)
sns.distplot(train["feature_2"],kde=False)
sns.distplot(train["feature_3"],kde=False)
historical_transaction = pd.read_csv("../input/historical_transactions.csv")
historical_transaction.describe(include="all")
historical_transaction.columns
historical_transaction.count()
columns_check = ['authorized_flag', 'category_1', 'installments',\
                 'category_3','month_lag','category_2','state_id','subsector_id']
def plot_distribution(column,df):
    distribution  = df[column].value_counts()
    plt.figure(figsize=(10,5))
    sns.barplot(distribution.index, distribution.values, alpha=0.8)
    plt.title('Distribution of ' + column)
    plt.ylabel("distribution", fontsize=12)
    plt.xlabel(column, fontsize=12)
    plt.show()
for col in columns_check:
    plot_distribution(col,historical_transaction)
merchants = pd.read_csv("../input/merchants.csv")
merchants.head(5)
merchants.head()
columns_to_check_merchant = [
       'subsector_id', 'category_1',
       'most_recent_sales_range', 'most_recent_purchases_range',
       'category_4', 'city_id', 'state_id', 'category_2']
['merchant_group_id', 'merchant_category_id',
       'numerical_1', 'numerical_2', 'category_1',
       'most_recent_sales_range', 'most_recent_purchases_range',       
       'category_4', 'city_id', 'state_id', 'category_2']
for col in columns_to_check_merchant:
    plot_distribution(col,merchants)
merchants["numerical_1"].value_counts()
merchants["numerical_2"].value_counts()
merchants[["numerical_1","numerical_2"]].corr()

new_trans_df = pd.read_csv("../input/new_merchant_transactions.csv")
new_trans_df.head()
new_trans_df.columns == historical_transaction.columns
gdf = historical_transaction.groupby("card_id")
gdf = gdf["purchase_amount"].size().reset_index()
gdf.columns = ["card_id", "num_hist_transactions"]
train = pd.merge(train, gdf, on="card_id", how="left")
test = pd.merge(test, gdf, on="card_id", how="left")
gdf = historical_transaction.groupby("card_id")
gdf = gdf["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
gdf.columns = ["card_id", "sum_hist_trans", "mean_hist_trans", "std_hist_trans", "min_hist_trans", "max_hist_trans"]
train = pd.merge(train, gdf, on="card_id", how="left")
test = pd.merge(test, gdf, on="card_id", how="left")
train.columns
gdf = new_trans_df.groupby("card_id")
gdf = gdf["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
gdf.columns = ["card_id", "sum_merch_trans", "mean_merch_trans", "std_merch_trans", "min_merch_trans", "max_merch_trans"]
train = pd.merge(train, gdf, on="card_id", how="left")
test = pd.merge(test, gdf, on="card_id", how="left")
gdf = new_trans_df.groupby("card_id")
gdf = gdf["purchase_amount"].size().reset_index()
gdf.columns = ["card_id", "num_merch_transactions"]
train = pd.merge(train, gdf, on="card_id", how="left")
test = pd.merge(test, gdf, on="card_id", how="left")
target_col = "target"
train["year"] = train["first_active_month"].dt.year
test["year"] = test["first_active_month"].dt.year
train["month"] = train["first_active_month"].dt.month
test["month"] = test["first_active_month"].dt.month

cols_to_use = ["feature_1", "feature_2", "feature_3", "year", "month", 
               "num_hist_transactions", "sum_hist_trans", "mean_hist_trans", "std_hist_trans", 
               "min_hist_trans", "max_hist_trans",
               "num_merch_transactions", "sum_merch_trans", "mean_merch_trans", "std_merch_trans",
               "min_merch_trans", "max_merch_trans",
              ]

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
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100, evals_result=evals_result)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result

train_X = train[cols_to_use]
test_X = test[cols_to_use]
train_y = train[target_col].values

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

sub_df = pd.DataFrame({"card_id":test["card_id"].values})
sub_df["target"] = pred_test
sub_df.to_csv("baseline_lgb.csv", index=False)

