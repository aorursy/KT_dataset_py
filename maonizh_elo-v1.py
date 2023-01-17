# from SRK, thanks!

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

import json
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

%matplotlib inline

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
# train.csv                     训练数据
# merchant.csv                  关于数据集中所有商户/商户id的附加信息
# sample_submission.csv         正确格式的示例提交文件——包含期望预测的所有card_id
# test.csv                      测试数据
# historical_transaction.csv    每个card_id三个月内的历史交易信息
# new_merchant_transaction.csv  每个card_id两个月的数据,包含在历史数据中未访问过的merchant_ids上的所有购买行为      
train_df = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])   # 将首次活动时间解析为日期
test_df = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])
print("Number of rows and columns in train set : ",train_df.shape)
print("Number of rows and columns in test set : ",test_df.shape)
train_df.head()  # 首次活跃时间 卡号 特征1 特征2 特征3 目标值
# target_col 分布图 

target_col = 'target'

plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df[target_col].values))   # 绘制散点图
plt.xlabel('index', fontsize=12)
plt.ylabel('Loyalty Score', fontsize=12)    # 历史点击率
plt.show()
# target_col 直方图
# 直方图表示通过沿数据范围形成分箱，然后绘制条以显示落入每个分箱的观测次数的数据分布

plt.figure(figsize=(12,8))
sns.distplot(train_df[target_col].values, bins=50, color='red')
plt.title('Histograme of Loyalty score')
plt.xlabel('Loyalty score', fontsize=12)       # 历史点击率柱状图
plt.show()
(train_df[target_col]<-30).sum()
# 训练集所有首次活动时间频次统计
#条形图表示数值变量与每个矩形高度的中心趋势的估计值，并使用误差线提供关于该估计值附近的不确定性的一些指示

cnt_srs = train_df['first_active_month'].dt.date.value_counts()  # 获取首次活动时间的频次统计
cnt_srs = cnt_srs.sort_index()                                   # 按首次活动时间排序
plt.figure(figsize=(14,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')  # 索引为x 值为y
plt.xticks(rotation='vertical')                                       # 设置标签方向
plt.xlabel('First active month', fontsize=12)
plt.ylabel('Number of cards', fontsize=12)
plt.title('First active month count in train set')
plt.show()

# 测试集所有首次活动时间频次统计

cnt_srs = test_df['first_active_month'].dt.date.value_counts()
cnt_srs = cnt_srs.sort_index()
plt.figure(figsize=(14,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='red')
plt.xticks(rotation='vertical')
plt.xlabel('First active month', fontsize=12)
plt.ylabel('Number of cards', fontsize=12)
plt.title('First active month count in train set')
plt.show()
# 它显示了定量数据在一个（或多个）分类变量的多个层次上的分布，这些分布可以进行比较。
# 不像箱形图中所有绘图组件都对应于实际数据点，小提琴绘图以基础分布的核密度估计为特征。

# feature 1
plt.figure(figsize=(8, 4))
sns.violinplot(x='feature_1', y=target_col, data=train_df)         
plt.xticks(rotation='vertical')
plt.xlabel('Feature_1', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title('Feature_1 distribution')     
plt.show()

# feature 2
plt.figure(figsize=(8,4))
sns.violinplot(x='feature_2', y=target_col, data=train_df)
plt.xticks(rotation='vertical')
plt.xlabel('Feature_2', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title('Feature_2 distribution')
plt.show()

# feature 3
plt.figure(figsize=(8,4))
sns.violinplot(x='feature_3', y=target_col,data=train_df)
plt.xticks(rotation='vertical')
plt.xlabel('Feature_3', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title('Feature_3 distribution')
plt.show()
hist_df = pd.read_csv('../input/historical_transactions.csv', parse_dates=['purchase_date'])
hist_df.head()
gdf = hist_df.groupby('card_id')
gdf = gdf['purchase_amount'].size().reset_index()
gdf.columns = ['card_id', 'num_hist_transactions']
train_df = pd.merge(train_df, gdf, on='card_id', how='left')
test_df = pd.merge(test_df, gdf, on='card_id', how='left')
cnt_srs = train_df.groupby("num_hist_transactions")[target_col].mean()     # 每个交易数目数目下的点击率均值
cnt_srs = cnt_srs.sort_index()
cnt_srs = cnt_srs[:-50]

def scatter_plot(cnt_srs, color):
    trace = go.Scatter(
        x=cnt_srs.index[::-1],
        y=cnt_srs.values[::-1],
        showlegend=False,
        marker=dict(
            color=color,
        ),
    )
    return trace

trace = scatter_plot(cnt_srs, "orange")
layout = dict(
    title='Loyalty score by Number of historical transactions',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Histtranscnt")
bins = [0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 500, 10000]  # 指定区间
train_df['binned_num_hist_transactions'] = pd.cut(train_df['num_hist_transactions'], bins)  # 确定每个数所在区间
cnt_srs = train_df.groupby("binned_num_hist_transactions")[target_col].mean()

plt.figure(figsize=(12,8))
sns.boxplot(x="binned_num_hist_transactions", y=target_col, data=train_df, showfliers=False)
plt.xticks(rotation='vertical')
plt.xlabel('binned_num_hist_transactions', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("binned_num_hist_transactions distribution")
plt.show()
gdf = hist_df.groupby("card_id")
gdf = gdf["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
gdf.columns = ["card_id", "sum_hist_trans", "mean_hist_trans", "std_hist_trans", "min_hist_trans", "max_hist_trans"]
train_df = pd.merge(train_df, gdf, on="card_id", how="left")
test_df = pd.merge(test_df, gdf, on="card_id", how="left")
bins = np.percentile(train_df["sum_hist_trans"], range(0,101,10))             # 处于p%位置的值称第p百分位数
train_df['binned_sum_hist_trans'] = pd.cut(train_df['sum_hist_trans'], bins)  # 将train_df按10个分位数分为10份，每个值替换为为其所在区间
#cnt_srs = train_df.groupby("binned_sum_hist_trans")[target_col].mean()

plt.figure(figsize=(12,8))
sns.boxplot(x="binned_sum_hist_trans", y=target_col, data=train_df, showfliers=False)
plt.xticks(rotation='vertical')
plt.xlabel('binned_sum_hist_trans', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Sum of historical transaction value (Binned) distribution")
plt.show()
bins = np.percentile(train_df["mean_hist_trans"], range(0,101,10))
train_df['binned_mean_hist_trans'] = pd.cut(train_df['mean_hist_trans'], bins)
#cnt_srs = train_df.groupby("binned_mean_hist_trans")[target_col].mean()

plt.figure(figsize=(12,8))
sns.boxplot(x="binned_mean_hist_trans", y=target_col, data=train_df, showfliers=False)
plt.xticks(rotation='vertical')
plt.xlabel('Binned Mean Historical Transactions', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Mean of historical transaction value (Binned) distribution")
plt.show()
new_trans_df = pd.read_csv("../input/new_merchant_transactions.csv", parse_dates=['purchase_date'])
new_trans_df.head()
gdf = new_trans_df.groupby("card_id")
gdf = gdf["purchase_amount"].size().reset_index()             # 交易记录数量
gdf.columns = ["card_id", "num_merch_transactions"]
train_df = pd.merge(train_df, gdf, on="card_id", how="left")
test_df = pd.merge(test_df, gdf, on="card_id", how="left")
bins = [0, 10, 20, 30, 40, 50, 75, 10000]
train_df['binned_num_merch_transactions'] = pd.cut(train_df['num_merch_transactions'], bins)
cnt_srs = train_df.groupby("binned_num_merch_transactions")[target_col].mean()

plt.figure(figsize=(12,8))
sns.boxplot(x="binned_num_merch_transactions", y=target_col, data=train_df, showfliers=False)
plt.xticks(rotation='vertical')
plt.xlabel('binned_num_merch_transactions', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Number of new merchants transaction (Binned) distribution")
plt.show()
gdf = new_trans_df.groupby("card_id")
gdf = gdf["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
gdf.columns = ["card_id", "sum_merch_trans", "mean_merch_trans", "std_merch_trans", "min_merch_trans", "max_merch_trans"]
train_df = pd.merge(train_df, gdf, on="card_id", how="left")
test_df = pd.merge(test_df, gdf, on="card_id", how="left")
bins = np.nanpercentile(train_df["sum_merch_trans"], range(0,101,10))
train_df['binned_sum_merch_trans'] = pd.cut(train_df['sum_merch_trans'], bins)
#cnt_srs = train_df.groupby("binned_sum_hist_trans")[target_col].mean()

plt.figure(figsize=(12,8))
sns.boxplot(x="binned_sum_merch_trans", y=target_col, data=train_df, showfliers=False)
plt.xticks(rotation='vertical')
plt.xlabel('binned sum of new merchant transactions', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Sum of New merchants transaction value (Binned) distribution")
plt.show()
bins = np.nanpercentile(train_df["mean_merch_trans"], range(0,101,10))
train_df['binned_mean_merch_trans'] = pd.cut(train_df['mean_merch_trans'], bins)
#cnt_srs = train_df.groupby("binned_sum_hist_trans")[target_col].mean()

plt.figure(figsize=(12,8))
sns.boxplot(x="binned_mean_merch_trans", y=target_col, data=train_df, showfliers=False)
plt.xticks(rotation='vertical')
plt.xlabel('binned mean of new merchant transactions', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Mean of New merchants transaction value (Binned) distribution")
plt.show()
train_df["year"] = train_df["first_active_month"].dt.year
test_df["year"] = test_df["first_active_month"].dt.year
train_df["month"] = train_df["first_active_month"].dt.month
test_df["month"] = test_df["first_active_month"].dt.month

cols_to_use = ["feature_1", "feature_2", "feature_3", "year", "month", 
               "num_hist_transactions", "sum_hist_trans", "mean_hist_trans", "std_hist_trans", 
               "min_hist_trans", "max_hist_trans",
               "num_merch_transactions", "sum_merch_trans", "mean_merch_trans", "std_merch_trans",
               "min_merch_trans", "max_merch_trans",
              ]
def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 144,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.9,
        'bagging_seed':0,
        'bagging_freq': 1,
        'verbose': 1,
        'reg_alpha':3,
        'reg_lambda':5
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100, evals_result=evals_result)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result
train_X = train_df[cols_to_use]
test_X = test_df[cols_to_use]
train_y = train_df[target_col].values

pred_test = 0
kf = model_selection.KFold(n_splits=5, random_state=1000, shuffle=True)
for dev_index, val_index in kf.split(train_df):
    dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    
    pred_test_tmp, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)
    pred_test += pred_test_tmp
pred_test /= 5.
