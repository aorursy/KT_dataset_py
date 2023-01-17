import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

%matplotlib inline
# inputフォルダのファイル

import glob

for f in glob.glob("../input/*"):

    print(f)
df = pd.read_csv("../input/train.csv")
df

# 1460行、81列
df.head()
df.columns
df.describe()
df.SalePrice.describe()
sns.distplot(df.SalePrice)
corrmat = df.corr()

corrmat
f, ax = plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=.8, square=True)
k = 10

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f', annot_kws={'size':10},

                yticklabels=cols.values,

                xticklabels=cols.values)

plt.show()
cols = corrmat.nlargest(38, 'SalePrice')['SalePrice'].index[-10:]

cols = cols.append(pd.indexes.base.Index(['SalePrice']))

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f', annot_kws={'size':10},

                yticklabels=cols.values,

                xticklabels=cols.values)

plt.show()
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df[cols], size = 2.5)

plt.show()
var = 'OverallQual'

data = pd.concat([df['SalePrice'], df[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
var = 'GrLivArea'

data = pd.concat([df['SalePrice'], df[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
from sklearn.linear_model import LinearRegression



X = df[["GrLivArea"]].values

y = df["SalePrice"].values



slr = LinearRegression()

slr.fit(X,y)



print('傾き：{0}'.format(slr.coef_[0]))

print('y切片: {0}'.format(slr.intercept_))
plt.scatter(X,y)

plt.plot(X,slr.predict(X),color='red')

plt.show()
df["GrLivArea"].values
df["SalePrice"].values
df_test = pd.read_csv("../input/test.csv")
df_test.head()
# SalePriceはない。

df_test.columns
X_test = df_test[["GrLivArea"]].values

y_test_pred = slr.predict(X_test)
y_test_pred
df_test["SalePrice"] = y_test_pred
df_test.head()
df_test[["Id","SalePrice"]].head()
df_test[["Id","SalePrice"]].to_csv("../output/submission.csv",index=False)