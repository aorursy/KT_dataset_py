import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns



train = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

test = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")

target = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

target_nonscored = pd.read_csv("/kaggle/input/lish-moa/train_targets_nonscored.csv")

sub = pd.read_csv("/kaggle/input/lish-moa/sample_submission.csv")
print(train.shape)

train.head()
train.info()
sub = train.select_dtypes("object")

col = train.columns.tolist()

print(sub.columns)

for c in sub.columns:

    col.remove(c)

df = train[col]

df.shape
print(sub['cp_type'].value_counts())

sub['cp_type'].value_counts().plot(kind="bar")
print(sub['cp_dose'].value_counts())

sub['cp_dose'].value_counts().plot(kind="bar")
df.filter(regex="^g-").describe()
df.filter(regex="^c-").describe()
set(df.columns)-set(df.filter(regex="^g-").columns.tolist() + df.filter(regex="^c-").columns.tolist())
print(df['cp_time'].value_counts())

df['cp_time'].value_counts().plot(kind="bar")
is_null = train.isnull().sum()

is_null[is_null!=0]
g=sns.PairGrid(data=train,x_vars=["cp_time","cp_type","cp_dose"],y_vars=["g-770","g-1","c-7","c-95"])

g.map(sns.boxenplot)
data = pd.concat([train,target],axis=1)

print(data.shape)

data.head()
target.shape
data["count"] = data.iloc[:,-207:].sum(axis=1)
sns.catplot(data=data,x="cp_type",y="count",kind="boxen")
data.iloc[:,-207:][data["cp_type"]=="ctl_vehicle"].head()
sns.catplot(data=data,x="cp_time",y="count",kind="boxen")
sns.catplot(data=data,x="cp_dose",y="count",kind="boxen")
print(data["count"].value_counts())

data["count"].value_counts().plot(kind="barh")
so = data.filter(regex="^g-").corr().unstack().sort_values(kind="quicksort")

sns.heatmap(data[so[so!=1].sort_values(ascending =False)[:10].reset_index(level=[0]).index].corr(),annot = True)
sns.pairplot(data[["g-770","g-1","cp_dose"]],hue="cp_dose")
sns.lineplot(data=data[["g-770","g-1"]])
data.filter(regex="^g-").mean().hist()
sns.boxplot(data.filter(regex="^g-").mean())
gini_mean = data.filter(regex="^g-")

gini_mean[gini_mean<-0.5 ].index
gini_row_mean = data.filter(regex="^g-").mean(axis=1)

mean = gini_row_mean.mean()

outlayer_index = gini_row_mean[(gini_row_mean>3*mean) | (gini_row_mean <-3*mean)].index

outlayer_index
data["cp_type"][outlayer_index].value_counts().plot(kind="barh")
so = data.filter(regex="^c-").corr().unstack().sort_values(kind="quicksort")

sns.heatmap(data[so[so!=1].sort_values(ascending =False)[:10].reset_index(level=[0]).index].corr(),annot = True)
sns.pairplot(data[["c-7","c-95","cp_dose"]],hue="cp_dose")
sns.lineplot(data=data[["c-7","c-95"]])
data.filter(regex="^c-").mean().hist()
sns.boxplot(data.filter(regex="^c-").mean())
gini_row_mean = data.filter(regex="^c-").mean(axis=1)

mean = gini_row_mean.mean()

outlayer_index = gini_row_mean[(gini_row_mean>3*mean) | (gini_row_mean <-3*mean)].index

outlayer_index
data["cp_type"][outlayer_index].value_counts().plot(kind="barh")