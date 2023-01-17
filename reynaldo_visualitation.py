import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn import model_selection, preprocessing

import xgboost as xgb

import datetime

#now = datetime.datetime.now()



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

macro = pd.read_csv('../input/macro.csv')



train["price_doc"] /= 1e6



train.sample(3)
#sns.distplot(train["price_doc"].values)
#let's look more closer

#sns.distplot(train[train["price_doc"] <= 20]["price_doc"].values)
train["timestamp"] = pd.to_datetime(train["timestamp"])

train["year"], train["month"], train["day"] = train["timestamp"].dt.year,train["timestamp"].dt.month,train["timestamp"].dt.day

train['yearmonth'] = train['timestamp'].apply(lambda x: str(x)[:4]+str(x)[5:7])



test["timestamp"] = pd.to_datetime(test["timestamp"])

test["year"], test["month"], test["day"] = test["timestamp"].dt.year,test["timestamp"].dt.month,test["timestamp"].dt.day

test['yearmonth'] = test['timestamp'].apply(lambda x: str(x)[:4]+str(x)[5:7])
train["count"] = 1

count_year = train.groupby("year").count().reset_index()

sns.barplot(count_year["year"],count_year["count"])
test["count"] = 1

count_year = test.groupby("year").count().reset_index()

sns.barplot(count_year["year"],count_year["count"])
total = int(len(train)*0.2)

train["count"] = 1

train_inc = train[-total:]

count_year = train_inc.groupby("year").count().reset_index()

sns.barplot(count_year["year"],count_year["count"])
train.groupby("yearmonth").aggregate(np.mean).reset_index()

plt.figure(figsize= (12,8))

plt.xticks(rotation="vertical")

sns.barplot(train["yearmonth"].values,train["price_doc"].values)
train.groupby("build_year").aggregate(np.mean).reset_index()

plt.figure(figsize= (20,12))

plt.xticks(rotation="vertical")

sns.barplot(train["build_year"],train["full_sq"])
def corr_plot(dataframe, top_n, target, fig_x, fig_y):

    corrmat = dataframe.corr()

    #top_n - top n correlations +1 since price is included

    top_n = top_n + 1 

    cols = corrmat.nlargest(top_n, target)[target].index

    cm = np.corrcoef(train[cols].values.T)

    f, ax = plt.subplots(figsize=(fig_x,fig_y))

    sns.set(font_scale=1.25)

    cmap = plt.cm.viridis

    hm = sns.heatmap(cm, cbar=False, annot=True, square=True,cmap = cmap, fmt='.2f', annot_kws={'size': 10}, 

                 yticklabels=cols.values, xticklabels=cols.values)

    plt.show()

    return cols

corr_20 = corr_plot(train, 20, 'price_doc', 10,10)
#train.groupby("full_sq").size()
small_train = train.dropna()

print(len(train))

print(len(small_train))
square_per_room = small_train["life_sq"]/small_train["num_room"]

#square_per_room = square_per_room[square_per_room < 60]

plt.scatter(small_train["num_room"],square_per_room,color="red")

#plt.scatter(small_train["num_room"],small_train["price_doc"],color="blue")
avg_df = small_train.groupby("num_room").mean().reset_index()
print(len(test))

print(len(test.dropna()))