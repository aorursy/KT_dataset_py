import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import skew

from sklearn.linear_model import Lasso, LassoCV

from sklearn.preprocessing import StandardScaler, RobustScaler

import os

print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
print(train.shape)

print(test.shape)
train.head()
test.head()
train.describe()
test.describe()
train['SalePrice'].describe()
corr=train.corr().abs()

n_most_correlated=12

most_correlated_feature=corr['SalePrice'].sort_values(ascending=False)[:n_most_correlated].drop('SalePrice')

most_correlated_feature_name=most_correlated_feature.index.values

f, ax = plt.subplots(figsize=(10,5))

plt.xticks(rotation='90')

sns.barplot(x=most_correlated_feature_name, y=most_correlated_feature)

plt.title("SalePrice korelasyon")
def draw_scatter_pairs(data,cols=4, rows=3):

    feature_names=data.columns.values



    counter=0

    fig, axarr = plt.subplots(rows,cols,figsize=(22,16))

    for i in range(rows):

        for j in range(cols):

            if counter>=len(feature_names):

                break



            name=feature_names[counter]

            axarr[i][j].scatter(x = data[name], y = data['SalePrice'])

            axarr[i][j].set(xlabel=name, ylabel='SalePrice')



            counter+=1





    plt.show()

feature_names =list(most_correlated_feature_name) + ['SalePrice']

draw_scatter_pairs(train[feature_names], rows=4, cols=3)
fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.xlabel('GrLivArea', fontsize=10)

plt.ylabel('SalePrice', fontsize=10)

plt.show()
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)



fig, ax = plt.subplots()

ax.scatter(train['GrLivArea'], train['SalePrice'])

plt.xlabel('GrLivArea', fontsize=10)

plt.ylabel('SalePrice', fontsize=10)

plt.show()
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
print("Yeni train.shape:",train.shape)
fig, ax = plt.subplots()

ax.scatter(x = train['GarageArea'], y = train['SalePrice'])

plt.xlabel('GarageArea', fontsize=10)

plt.ylabel('SalePrice', fontsize=10)

plt.show()
train = train.drop(train[(train['GarageArea']>1200) & (train['SalePrice']<600000)].index)



fig, ax = plt.subplots()

ax.scatter(train['GarageArea'], train['SalePrice'])

plt.xlabel('GarageArea', fontsize=10)

plt.ylabel('SalePrice', fontsize=10)

plt.show()
sns.distplot(train['SalePrice']);
train["SalePrice"] = np.log1p(train["SalePrice"])

sns.distplot(train['SalePrice']);
sns.set(style="white")

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(15, 10))

cmap = sns.diverging_palette(110, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
corrmat = train.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)
corrmat = test.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)