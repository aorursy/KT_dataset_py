# loading necessary packages
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (explained_variance_score, median_absolute_error, mean_absolute_error, mean_squared_error)
train = pd.read_csv("../input/sales_train.csv")
items = pd.read_csv("../input/items.csv")
item_categories = pd.read_csv("../input/item_categories.csv")
test = pd.read_csv("../input/test.csv")
sub_sample = pd.read_csv("../input/sample_submission.csv")
shops = pd.read_csv("../input/shops.csv")
train.shape, test.shape
train.head()
train.describe()
train.info()
train.head()
train["date"] = pd.to_datetime(train["date"], format='%d.%m.%Y')
train["date"].head()
train["month"] = train["date"].dt.month
train["month"].head()
train["year"] = train["date"].dt.year
train["year"].head()
train = train.drop(["date","item_price"], axis=1)
train.head()
[count for count in train.columns if count not in ["item_cnt_day"]]
train["date_block_num"].unique()
train.groupby("date_block_num", as_index=False)["item_cnt_day"].sum()
train = train.groupby([count for count in train.columns if count not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()
train = train.rename(columns={'item_cnt_day':'item_cnt_month'})
train[['shop_id','item_id','item_cnt_month']].groupby(['shop_id','item_id'], as_index=False)[['item_cnt_month']].mean().head()
shop_item_monthly_mean = train[['shop_id','item_id','item_cnt_month']].groupby(['shop_id','item_id'], as_index=False)[['item_cnt_month']].mean()
shop_item_monthly_mean.head()
shop_item_monthly_mean = shop_item_monthly_mean.rename(columns={'item_cnt_month':'item_cnt_month_mean'})
train.head()
train = pd.merge(train, shop_item_monthly_mean, how='left', on=['shop_id','item_id'])
train.head()
# Last Month (Oct 2015)
shop_item_prev_month = train[train['date_block_num']==33][['shop_id','item_id','item_cnt_month']]
shop_item_prev_month.head()
shop_item_prev_month = shop_item_prev_month.rename(columns={'item_cnt_month':'item_cnt_prev_month'})
shop_item_prev_month.head()
train = pd.merge(train, shop_item_prev_month, how='left', on=['shop_id','item_id']).fillna(0.)
train.head()
train = pd.merge(train, items, how='left', on='item_id')
train = pd.merge(train, item_categories, how='left', on='item_category_id')
train.head()
train = pd.merge(train, shops, how='left', on='shop_id')
test["month"] = 11
test["year"] = 2015
test["date_block_num"] = 34
test.head()
test = pd.merge(test, shop_item_monthly_mean, how='left', on=['shop_id','item_id']).fillna(0.)
test.head()
test = pd.merge(test, shop_item_prev_month, how='left', on=['shop_id','item_id']).fillna(0.)
test = pd.merge(test, items, how='left', on='item_id')
test = pd.merge(test, item_categories, how='left', on='item_category_id')
test.head()
test = pd.merge(test, shops, how='left', on='shop_id')
test['item_cnt_month'] = 0
plt.subplots(figsize=(8, 6))
sns.heatmap(train.corr())
selected_feature = ["item_category_id", "item_cnt_month", "item_cnt_month_mean",
                    "item_cnt_prev_month", "item_id", "month","shop_id" , "year"]
train = train[selected_feature]
selected_feature.insert(0, "ID")
test = test[selected_feature]
plt.subplots(figsize=(10,6))
train.boxplot()
plt.xticks(rotation=90)
plt.subplots(figsize=(10,6))
test.boxplot()
plt.xticks(rotation=90)
train.groupby("item_cnt_month").count()
selected_feature.remove("item_cnt_month")
selected_feature.remove("ID")
scale = MinMaxScaler()
scaled_data = scale.fit_transform(train[selected_feature])
scaled_train = pd.DataFrame(data = scaled_data, columns = selected_feature)
scaled_train["item_cnt_month"] = train["item_cnt_month"]
scaled_train.head()
scaled_data = scale.fit_transform(test[selected_feature])
scaled_test = pd.DataFrame(data = scaled_data, columns = selected_feature)
scaled_test["item_cnt_month"] = test["item_cnt_month"]
scaled_train.head()
scaled_test.head()
