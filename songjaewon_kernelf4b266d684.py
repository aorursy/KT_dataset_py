# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

import lightgbm as lgb

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



sns.set_style('darkgrid')

%matplotlib inline



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
train_df.head()
train_df.describe()
train_df.nunique()
%%time

train_df["date"] = pd.to_datetime(train_df.date)

train_df["year"] = train_df['date'].dt.year

train_df["month"] = train_df['date'].dt.month

train_df["day"] = train_df['date'].dt.day

train_df["dayofweek"] = train_df["date"].dt.dayofweek

#0 : Monday, 6: Sunday in dayofweek
%%time

test_df["date"] = pd.to_datetime(test_df.date)

test_df["year"] = test_df['date'].dt.year

test_df["month"] = test_df['date'].dt.month

test_df["day"] = test_df['date'].dt.day

test_df["dayofweek"] = test_df["date"].dt.dayofweek

#0 : Monday, 6: Sunday in dayofweek
tmp_df = train_df.groupby(["year","month","store","item"]).sum().reset_index()[["year","month","store","item","sales"]]



plt.figure(figsize=(20,12))

sns.pointplot(data=tmp_df[(tmp_df.year == 2013) & (tmp_df.store == 1)], x="item", y="sales", hue="month")
item_list = train_df.item.unique().tolist()

split_item_list = [item_list[x:x+10] for x in range(0,len(item_list),10)]

split_item_list
tmp_df = train_df[train_df.item.isin(split_item_list[2])].groupby(["year","month","store","item"]).sum().reset_index()[["year","month","store","item","sales"]]

plt.figure(figsize=(16,6))

sns.pointplot(data=tmp_df[(tmp_df.year == 2013) & (tmp_df.store == 1)], x="item", y="sales", hue="month")
train_df["season"] = train_df.month.map({1:"Winter", 2:"Winter", 3:"Spring", 4:"Spring",5:"Spring",6:"Summer",

                                          7:"Summer", 8:"Summer", 9:"Fall", 10:"Fall", 11:"Fall", 12:"Winter"})

test_df["season"] = test_df.month.map({1:"Winter", 2:"Winter", 3:"Spring", 4:"Spring",5:"Spring",6:"Summer",

                                          7:"Summer", 8:"Summer", 9:"Fall", 10:"Fall", 11:"Fall", 12:"Winter"})
tmp_df = train_df[train_df.item.isin(split_item_list[0])].groupby(["year","season","store","item"]).sum().reset_index()[["year","season","store","item","sales"]]

plt.figure(figsize=(12,6))

sns.pointplot(data=tmp_df[(tmp_df.year == 2013) & (tmp_df.store == 1)], x="item", y="sales", hue="season")
tmp_df = train_df[train_df.item.isin(split_item_list[1])].groupby(["year","season","store","item"]).sum().reset_index()[["year","season","store","item","sales"]]

plt.figure(figsize=(12,6))

sns.pointplot(data=tmp_df[(tmp_df.year == 2013) & (tmp_df.store == 1)], x="item", y="sales", hue="season")
tmp_df = train_df[train_df.item.isin(split_item_list[2])].groupby(["year","season","store","item"]).sum().reset_index()[["year","season","store","item","sales"]]

plt.figure(figsize=(12,6))

sns.pointplot(data=tmp_df[(tmp_df.year == 2013) & (tmp_df.store == 1)], x="item", y="sales", hue="season")
tmp_df = train_df[train_df.item.isin(split_item_list[3])].groupby(["year","season","store","item"]).sum().reset_index()[["year","season","store","item","sales"]]

plt.figure(figsize=(12,6))

sns.pointplot(data=tmp_df[(tmp_df.year == 2013) & (tmp_df.store == 1)], x="item", y="sales", hue="season")
tmp_df = train_df[train_df.item.isin(split_item_list[4])].groupby(["year","season","store","item"]).sum().reset_index()[["year","season","store","item","sales"]]

plt.figure(figsize=(12,6))

sns.pointplot(data=tmp_df[(tmp_df.year == 2013) & (tmp_df.store == 1)], x="item", y="sales", hue="season")
sns.boxplot(data =train_df, x="dayofweek", y='sales')
"""train_df["is_weekend"] = train_df.dayofweek.map({0:0, 1:0, 2:0, 3:0, 4:0, 5:1, 6:1})

test_df["is_weekend"] = test_df.dayofweek.map({0:0, 1:0, 2:0, 3:0, 4:0, 5:1, 6:1})"""
train_df.head()
sales_by_year = train_df[["year","item","sales"]].groupby(["year","item"]).sum().reset_index().pivot("item","year","sales").reset_index()

sales_by_year
abs((np.array(sales_by_year[2013])-np.array(sales_by_year[2017]))/np.array(sales_by_year[2013]))/4 * 100
increase_13_to_14 = abs((np.array(sales_by_year[2013])-np.array(sales_by_year[2014]))/np.array(sales_by_year[2013])).mean()

increase_14_to_15 = abs((np.array(sales_by_year[2014])-np.array(sales_by_year[2015]))/np.array(sales_by_year[2014])).mean()

increase_15_to_16 = abs((np.array(sales_by_year[2015])-np.array(sales_by_year[2016]))/np.array(sales_by_year[2015])).mean()

increase_16_to_17 = abs((np.array(sales_by_year[2016])-np.array(sales_by_year[2017]))/np.array(sales_by_year[2016])).mean()
print("13_to_14 : {}, 14_to_15 : {}\n15_to_16 : {}, 16_to_17 : {}".format(increase_13_to_14, increase_14_to_15, increase_15_to_16, increase_16_to_17))
train_df['increase'] = train_df.year.map({2013:1, 2014:1+increase_13_to_14, 2015:1+increase_14_to_15, 2016:1+increase_15_to_16, 2017:1+increase_16_to_17})

test_df['increase'] = 0.05
sales_by_StoreAndItem = train_df[["store","item","sales"]].groupby(["store","item"]).sum().reset_index().sort_values(by="sales", ascending=False)
sales_by_StoreAndItem["store_item_index"] = pd.cut(train_df[["store","item","sales"]].groupby(["store","item"]).sum().reset_index().sort_values(by="sales", ascending=False).sales,10, labels=range(1,11))
train_df = pd.DataFrame.merge(train_df, sales_by_StoreAndItem.drop("sales", axis=1).reset_index().drop("index", axis=1) , on=["store","item"])

test_df = pd.DataFrame.merge(test_df,sales_by_StoreAndItem.drop("sales", axis=1).reset_index().drop("index", axis=1) , on=["store","item"])
train_df[["store_item_index"]].groupby("store_item_index").size()
train_df = train_df.set_index("date")

test_df = test_df.set_index("date")
train_df["season"] = train_df.season.map({"Winter":4, "Spring":1, "Summer":2, "Fall":3})

test_df["season"] = test_df.season.map({"Winter":4, "Spring":1, "Summer":2, "Fall":3})
def smape(actual, target):

    return 100 * np.mean(2 * np.abs(actual - target)/(np.abs(actual) + np.abs(target)))
model = lgb.LGBMRegressor(n_jobs=-1, n_estimators=2000, max_depth=8, objective='regression_l1', random_state=123)
valid_df = train_df[(train_df.year == 2017) & (train_df.month.isin([10, 11, 12]))]
%%time

train_df_dropped = train_df.drop(valid_df.index, axis=0)
X = train_df_dropped.drop("sales", axis=1)

y= train_df_dropped['sales']

valid_X = valid_df.drop("sales", axis=1)

valid_y = valid_df['sales']
model.fit(X, y, eval_set=[(valid_X, valid_y)], eval_metric=['mape', smape])
feature_importances = pd.DataFrame({'importance': model.feature_importances_, 'name': X.columns})

sns.barplot( data=feature_importances.sort_values('importance', ascending=False), x='importance', y='name')
pred = model.predict(valid_X)

smape(valid_y, pred)
test_df = test_df.drop("id", axis=1)

pred = model.predict(test_df)
f = pd.DataFrame({'sales' : pred }, index=range(0,len(pred))).reset_index()

f.columns = ["id", "sales"]

f.to_csv("submission_demandForecasting.csv", index=False)