# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Warnings

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

plt.rcParams['figure.figsize'] = (10, 7)

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
train.info()
test.info()
train.head()
# set datatime

train['date'] = pd.to_datetime(train['date'])

test['date'] = pd.to_datetime(test['date'])
# create datetime variable

train['tyear'] = train['date'].dt.year

train['tmonth'] = train['date'].dt.month

train['tday'] = train['date'].dt.day





test['tyear'] = test['date'].dt.year

test['tmonth'] = test['date'].dt.month

test['tday'] = test['date'].dt.day
g=sns.FacetGrid(train,col="store", col_order=[1,2,3,4,5,6,7,8,9,10],col_wrap=2,size=5)

g.map(sns.barplot,"tyear","sales")
train[['sales','store']].groupby(["store"]).mean().plot.bar(color='c')

plt.show()
train[['sales','tmonth']].groupby(["tmonth"]).mean().plot.bar(color='g')

plt.show()
train[['sales','tyear']].groupby(["tyear"]).mean().plot.bar(color='lightblue')

plt.show()
train[['sales','tday']].groupby(["tday"]).mean().plot.bar(color='lightgreen')

plt.show()
data_1=train.loc[train['store'] == 1]
data_1.head()
data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).plot.bar()

plt.show()
print("Top 5 selling item in store 1")

print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).head())
print("lest 5 selling item in store 1")

print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).tail())
data_1=train.loc[train['store'] == 2]
data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).plot.bar(color='lightblue')

plt.show()
print("Top 5 selling item in store 2")

print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).head())
print("lest 5 selling item in store 2")

print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).tail())
data_1=train.loc[train['store'] == 3]
data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).plot.bar(color='g')

plt.show()
print("Top 5 selling item in store 3")

print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).head())
print("lest 5 selling item in store 3")

print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).tail())
data_1=train.loc[train['store'] == 4]
data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).plot.bar(color='y')

plt.show()
print("Top 5 selling item in store 4")

print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).head())
print("lest 5 selling item in store 4")

print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).tail())
data_1=train.loc[train['store'] == 1]
data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).plot.bar(color='lightblue')

plt.show()
print("Top 5 selling item in store 5")

print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).head())
print("lest 5 selling item in store 5")

print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).tail())
data_1=train.loc[train['store'] == 6]
data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).plot.bar()

plt.show()
print("Top 5 selling item in store 6")

print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).head())
print("lest 5 selling item in store 6")

print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).tail())
data_1=train.loc[train['store'] == 7]
data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).plot.bar(color='y')

plt.show()
print("Top 5 selling item in store 7")

print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).head())
print("lest 5 selling item in store 7")

print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).tail())
data_1=train.loc[train['store'] == 8]
data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).plot.bar(color='lightblue')

plt.show()
print("Top 5 selling item in store 8")

print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).head())
print("lest 5 selling item in store 8")

print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).tail())
data_1=train.loc[train['store'] == 9]
data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).plot.bar(color='b')

plt.show()
print("Top 5 selling item in store 9")

print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).head())
print("lest 5 selling item in store 9")

print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).tail())
data_1=train.loc[train['store'] == 10]
data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).plot.bar()

plt.show()
print("Top 5 selling item in store 10")

print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).head())
print("lest 5 selling item in store 10")

print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).tail())