import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import math

import re



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import os
train=pd.read_csv('../input/train.tsv', sep='\t', encoding='utf-8')

test=pd.read_csv('../input/test.tsv', sep='\t', encoding='utf-8')

sample = pd.read_csv('../input/sample_submission.csv', sep='\t', encoding='utf-8')
# price to logprice + 1

train["logprice"] = np.log(train["price"]+1)



# Push "Other/Other/Other" into NaN category name

train.loc[train["category_name"].isnull(), ["category_name"]] = "Other/Other/Other" #Merge into others

test.loc[test["category_name"].isnull(), ["category_name"]] = "Other/Other/Other" #Merge into others



# make 1st / 2nd level category label

train["1st_category"] = train["category_name"].str.extract('([^/]+)/[^/]+/[^/]+')

train["2nd_category"] = train["category_name"].str.extract('([^/]+/[^/]+)/[^/]+')

test["1st_category"] = test["category_name"].str.extract('([^/]+)/[^/]+/[^/]+')

test["2nd_category"] = test["category_name"].str.extract('([^/]+/[^/]+)/[^/]+')



train.head(20)
print(train["1st_category"].drop_duplicates().count())

print(train["2nd_category"].drop_duplicates().count())

print(train["category_name"].drop_duplicates().count())
c1 = train.groupby(["1st_category"])["logprice"].std()

print(c1.mean())



c2 = train.groupby(["2nd_category"])["logprice"].std()

print(c2.mean())



c3 = train.groupby(["category_name"])["logprice"].std()

print(c3.mean())
set(["hoge", "piyo", "fuga"]).difference(set(["hoge", "piyo", "foo"]))
train_cat = train["category_name"].drop_duplicates().values.tolist()

test_cat = test["category_name"].drop_duplicates().values.tolist()

set(test_cat).difference(set(train_cat))
len(set(train_cat).difference(set(test_cat))) # number only to avoid long list
train_cat2 = train["2nd_category"].drop_duplicates().values.tolist()

test_cat2 = test["2nd_category"].drop_duplicates().values.tolist()

print(set(test_cat2).difference(set(train_cat2)))

print(set(train_cat2).difference(set(test_cat2)))
group1 = train.groupby(["1st_category"])

cat1 = pd.DataFrame(group1["price"].mean())

cat1["num"] = group1["1st_category"].count()

cat1["logprice"] = group1["logprice"].mean()

cat1["logstd"] = group1["logprice"].std()

cat1["min"] = group1["price"].min()

cat1["max"] = group1["price"].max()

cat1["std"] = group1["price"].std()

cat1["median"] = group1["price"].median()

cat1 = cat1.sort_values(by='num', ascending = False)

cat1
f1 = train[["logprice", "1st_category"]]

plt.figure(figsize=(15, 8))

ax = sns.lvplot(y=f1["1st_category"], x=f1["logprice"])
group2 = train.groupby(["2nd_category"])

cat2 = pd.DataFrame(group2["price"].mean())

cat2["num"] = group2["2nd_category"].count()

cat2["logprice"] = group2["logprice"].mean()

cat2["logstd"] = group2["logprice"].std()

cat2["min"] = group2["price"].min()

cat2["max"] = group2["price"].max()

cat2["std"] = group2["price"].std()

cat2["median"] = group2["price"].median()

cat2 = cat2.sort_values(by='num', ascending = False)

cat2.head(20)
plt.figure(figsize=(15, 8))

#ax = sns.lvplot(x=cat.index, y = cat.num)

ax = sns.stripplot(x=cat2.index, y = cat2.num, color="Red")

d = ax.set_ylim(0,)

d = ax.set(xlabel='category_name', ylabel='Items in category')

d = ax.set(xticklabels=[])
plt.figure(figsize=(15, 8))

#ax = sns.lvplot(x=cat.index, y = cat.num)

ax = sns.stripplot(x=cat2.tail(60).index, y = cat2.tail(60).num, color="Red")

d = ax.set_ylim(0,)

d = ax.set(xlabel='category_name', ylabel='Items in category')

d = ax.set(xticklabels=[])
#f2 = train.loc[train["2nd_category"].isin(cat2)][["logprice", "2nd_category"]]

f2 = train[["logprice", "2nd_category"]].sort_values(by=["2nd_category"])

plt.figure(figsize=(9,27))

ax = sns.lvplot(y=f2["2nd_category"], x =f2["logprice"])
group3 = train.groupby(["category_name"])

cat3 = pd.DataFrame(group3["price"].mean())

cat3["num"] = group3["category_name"].count()

cat3["logprice"] = group3["logprice"].mean()

cat3["logstd"] = group3["logprice"].std()

cat3["min"] = group3["price"].min()

cat3["max"] = group3["price"].max()

cat3["std"] = group3["price"].std()

cat3["median"] = group3["price"].median()

cat3 = cat3.sort_values(by='num', ascending = False)

cat3.head(20)
plt.figure(figsize=(15, 8))

#ax = sns.lvplot(x=cat.index, y = cat.num)

ax = sns.stripplot(x=cat3.index, y = cat3.num, color="Red")

d = ax.set_ylim(0,)

d = ax.set(xlabel='category_name', ylabel='Items in category')

d = ax.set(xticklabels=[])
print(len(cat3.index)) # 1287

print(len(cat3.where(cat3["num"]<=100).dropna()))

print(len(cat3.where(cat3["num"]<=15).dropna()))
cat3_top30 = list(cat3.head(30).index) # top 30 categories

cat3_top30

f3 = train.loc[train["category_name"].isin(cat3_top30)].sort_values(by="category_name")[["price", "category_name"]]



f3["logprice"] = np.log(f3["price"]+1)

plt.figure(figsize=(12, 9))

ax = sns.lvplot(y=f3["category_name"], x =f3["logprice"])

d = ax.set(ylabel='category_name(top 30)', xlabel='logprice')
group3 = train.groupby(["category_name"])

#c32 = pd.DataFrame()

#c32["logprice"] = group3["logprice"].mean()

cat3["logstd"] = group3["logprice"].std()

#c32["num"] =  group3["category_name"].count()

#c32 = c32.sort_values(by="num", ascending=False)



plt.figure(figsize=(12, 9))

ax = sns.stripplot(y=cat3["logstd"], x =cat3.index)

d = ax.set(ylabel='logstd', xlabel='<- popular  category    unpopular->')

d = ax.set(xticklabels=[])