# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd 

%matplotlib inline

import matplotlib.pyplot as plt  

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')



from scipy import stats

from scipy.stats import norm, skew #for some statistics



import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
#訓練データ

df_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

#テストデータ

df_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")



#df_trainとdf_testを行方向に結合して、結合箇所をlen_trainで保持

len_train=df_train.shape[0]

#全データ

df_all=pd.concat([df_train,df_test], sort=False)



print(df_train.shape)

print(df_test.shape)

print(df_all.shape)
# ターゲット(SalePrice)との相関関係

correlations = df_train.corr()['SalePrice']
var = "MSSubClass"



#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['misTotal', 'Percent'])

missing_data.loc[[var]]

#統計量確認（対象：全データ）

df_all[var].describe()
#データ数量確認（対象：全データ）

df_all[var].value_counts()
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=45)

plt.title("{} count".format(var))

#———
#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

#———
#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

plt.title("{} & Median Price".format(var))

sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')
#8

#欠測処理：×

#はずれ値処理：×

#数値データ標準化処理：×

#文字列の数値化（順序変数）：◯



#int64をstrに変更

df_all['MSSubClass'] = df_all['MSSubClass'].apply(str)
#1

var = "Functional"



#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———
#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———
#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=45)

plt.title("{} count".format(var))

#———

#6

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

#———
#7

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

plt.title("{} & Median Price".format(var))

sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

#———
#8

#欠測処理：〇

#はずれ値処理：

#数値データ標準化処理：

#文字列の数値化（順序変数）：

#対数変換標準化（数値変数）：
#1

var = "SaleType"



#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———

#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———
#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=45)

plt.title("{} count".format(var))

#———
#6

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

#———
#7

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

plt.title("{} & Median Price".format(var))

sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

#———
#1

var = "SaleCondition"



#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———

#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———

#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=45)

plt.title("{} count".format(var))

#———
#6

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

#———
#7

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

plt.title("{} & Median Price".format(var))

sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

#———
#1

var = "MoSold"



#相関係数確認（対象：訓練データ）：数値タイプのみ

print("SalePriceとの相関関係係数：{:.4f}".format(correlations[var]))

#-------
#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———
#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———

#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=45)

plt.title("{} count".format(var))

#———
#6

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

#———

#7

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

plt.title("{} & Median Price".format(var))

sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

#———
#1

var = "YrSold"



#相関係数確認（対象：訓練データ）：数値タイプのみ

print("SalePriceとの相関関係係数：{:.4f}".format(correlations[var]))

#-------
#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———
#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———

#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=45)

plt.title("{} count".format(var))

#———
#6

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

#———
#7

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

plt.title("{} & Median Price".format(var))

sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

#———
#1

var = "TotRmsAbvGrd"



#データ型確認

print(df_all[var])



#相関係数確認（対象：訓練データ）：数値タイプのみ

print("SalePriceとの相関関係係数：{:.4f}".format(correlations[var]))

#-------
#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———
#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———

#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=45)

plt.title("{} count".format(var))

#———
#6

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

#———
#7

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

plt.title("{} & Median Price".format(var))

sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

#———
#1

var = "1stFlrSF"



#データ型確認

print(df_all[var])



#相関係数確認（対象：訓練データ）：数値タイプのみ

print("SalePriceとの相関関係係数：{:.4f}".format(correlations[var]))

#-------
#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———

#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———

#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———

#5：数値変数のみ

# 価格との散布図

plt.figure(figsize=(8,4))

sns.scatterplot(df_train[var],df_train['SalePrice'])

#———

#6：数値変数のみ

# データ分布確認

plt.figure(figsize=(8,4))

sns.distplot(df_all[var])

plt.show()
#1

var = "2ndFlrSF"



#データ型確認

print(df_all[var])



#相関係数確認（対象：訓練データ）：数値タイプのみ

print("SalePriceとの相関関係係数：{:.4f}".format(correlations[var]))

#-------
#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———
#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———

#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：数値変数のみ

# 価格との散布図

plt.figure(figsize=(8,4))

sns.scatterplot(df_train[var],df_train['SalePrice'])

#———
#6：数値変数のみ

# データ分布確認

plt.figure(figsize=(8,4))

sns.distplot(df_all[var])

plt.show()
#1

var = "LowQualFinSF"



#データ型確認

print(df_all[var])



#相関係数確認（対象：訓練データ）：数値タイプのみ

print("SalePriceとの相関関係係数：{:.4f}".format(correlations[var]))

#-------
#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———
#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———
#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：数値変数のみ

# 価格との散布図

plt.figure(figsize=(8,4))

sns.scatterplot(df_train[var],df_train['SalePrice'])

#———
#6：数値変数のみ

# データ分布確認

plt.figure(figsize=(8,4))

sns.distplot(df_all[var])

plt.show()
#1

var = "GrLivArea"



#データ型確認

print(df_all[var])



#相関係数確認（対象：訓練データ）：数値タイプのみ

print("SalePriceとの相関関係係数：{:.4f}".format(correlations[var]))

#-------
#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———
#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———
#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：数値変数のみ

# 価格との散布図

plt.figure(figsize=(8,4))

sns.scatterplot(df_train[var],df_train['SalePrice'])

#———
#6：数値変数のみ

# データ分布確認

plt.figure(figsize=(8,4))

sns.distplot(df_all[var])

plt.show()
#1

var = "Neighborhood"



#データ型確認

print(df_all[var])



#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———
#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———
#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=5, aspect=2)

g.set_xticklabels(rotation=90)

plt.title("{} count".format(var))

#———
#6：カテゴリ変数のみ

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

g = sns.boxplot(x=var, y="SalePrice", data=data)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.axis(ymin=0, ymax=800000);

#———

#7：カテゴリ変数のみ

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

g = sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.title("{} & Median Price".format(var))

#———
#1

var = "Condition1"



#データ型確認

print(df_all[var])



#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———
#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———
#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=90)

plt.title("{} count".format(var))

#———
#6：カテゴリ変数のみ

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

g = sns.boxplot(x=var, y="SalePrice", data=data)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.axis(ymin=0, ymax=800000);

#———
#7：カテゴリ変数のみ

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

g = sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.title("{} & Median Price".format(var))

#———
#1

var = "Condition2"



#データ型確認

print(df_all[var])



#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———
#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———
#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———


#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=90)

plt.title("{} count".format(var))

#———
#1

var = "MSZoning"



#データ型確認

print(df_all[var])

#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———
#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———
#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=90)

plt.title("{} count".format(var))

#———
#6：カテゴリ変数のみ

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

g = sns.boxplot(x=var, y="SalePrice", data=data)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.axis(ymin=0, ymax=800000);

#———
#7：カテゴリ変数のみ

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

g = sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.title("{} & Median Price".format(var))

#———
#1

var = "Street"



#データ型確認

print(df_all[var])



#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———
#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———
#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=90)

plt.title("{} count".format(var))

#———
#6：カテゴリ変数のみ

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

g = sns.boxplot(x=var, y="SalePrice", data=data)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.axis(ymin=0, ymax=800000);

#———
#7：カテゴリ変数のみ

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

g = sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.title("{} & Median Price".format(var))

#———
#1

var = "Alley"



#データ型確認

print(df_all[var])



#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———
#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———
#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=90)

plt.title("{} count".format(var))

#———
#6：カテゴリ変数のみ

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

g = sns.boxplot(x=var, y="SalePrice", data=data)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.axis(ymin=0, ymax=800000);

#———
#7：カテゴリ変数のみ

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

g = sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.title("{} & Median Price".format(var))

#———
#1

var = "LandContour"



#データ型確認

print(df_all[var])



#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———
#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———
#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=90)

plt.title("{} count".format(var))

#———
#6：カテゴリ変数のみ

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

g = sns.boxplot(x=var, y="SalePrice", data=data)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.axis(ymin=0, ymax=800000);

#———
#7：カテゴリ変数のみ

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

g = sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.title("{} & Median Price".format(var))

#———
#1

var = "LotConfig"



#データ型確認

print(df_all[var])



#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———
#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———
#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=90)

plt.title("{} count".format(var))

#———
#6：カテゴリ変数のみ

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

g = sns.boxplot(x=var, y="SalePrice", data=data)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.axis(ymin=0, ymax=800000);

#———
#7：カテゴリ変数のみ

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

g = sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.title("{} & Median Price".format(var))

#———
#1

var = "LotShape"



#データ型確認

print(df_all[var])



#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———
#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———
#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=90)

plt.title("{} count".format(var))

#———
#6：カテゴリ変数のみ

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

g = sns.boxplot(x=var, y="SalePrice", data=data)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.axis(ymin=0, ymax=800000);

#———
#7：カテゴリ変数のみ

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

g = sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.title("{} & Median Price".format(var))

#———
#1

var = "LandSlope"



#データ型確認

print(df_all[var])



#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———
#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———
#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=90)

plt.title("{} count".format(var))

#———
#6：カテゴリ変数のみ

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

g = sns.boxplot(x=var, y="SalePrice", data=data)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.axis(ymin=0, ymax=800000);

#———
#7：カテゴリ変数のみ

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

g = sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.title("{} & Median Price".format(var))

#———
#1

var = "LotFrontage"



#データ型確認

print(df_all[var])



#相関係数確認（対象：訓練データ）：数値タイプのみ

print("SalePriceとの相関関係係数：{:.4f}".format(correlations[var]))

#-------
#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———
#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———
#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：数値変数のみ

# 価格との散布図

plt.figure(figsize=(8,4))

sns.scatterplot(df_train[var],df_train['SalePrice'])

#———
#1

var = "LotArea"



#データ型確認

print(df_all[var])



#相関係数確認（対象：訓練データ）：数値タイプのみ

print("SalePriceとの相関関係係数：{:.4f}".format(correlations[var]))

#-------
#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———
#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———

#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：数値変数のみ



# 価格との散布図

plt.figure(figsize=(8,4))

sns.scatterplot(df_train[var] ,df_train['SalePrice'])

#sns.scatterplot(df_train_temp ,df_train['SalePrice'])

#———
#LotArea > 100000 を除外して再描画

col = df_train[var]

df_train_temp = col[np.abs(col) < 100000]



# 価格との散布図

plt.figure(figsize=(8,4))

sns.scatterplot(df_train_temp ,df_train['SalePrice'])

#———
# LotArea > 100000 を除外したデータで相関係数算出

df_train['SalePrice'].corr(df_train_temp)
#6：数値変数のみ

# データ分布確認

plt.figure(figsize=(8,4))

sns.distplot(df_all[var])

#sns.distplot(df_all[var])

plt.show()
# LotArea > 100000 を除外したデータで分布図再描画

col = df_all[var]

df_all_temp = col[np.abs(col) < 100000]



#6：数値変数のみ

# データ分布確認

plt.figure(figsize=(8,4))

sns.distplot(df_all_temp)

plt.show()

#1

var = "BldgType"



#データ型確認

print(df_all[var])



#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———
#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———
#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=90)

plt.title("{} count".format(var))

#———
#6：カテゴリ変数のみ

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

g = sns.boxplot(x=var, y="SalePrice", data=data)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.axis(ymin=0, ymax=800000);

#———
#7：カテゴリ変数のみ

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

g = sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.title("{} & Median Price".format(var))

#———
#1

var = "HouseStyle"



#データ型確認

print(df_all[var])



#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———

#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———

#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=90)

plt.title("{} count".format(var))

#———
#6：カテゴリ変数のみ

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

g = sns.boxplot(x=var, y="SalePrice", data=data)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.axis(ymin=0, ymax=800000);

#———
#7：カテゴリ変数のみ

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

g = sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.title("{} & Median Price".format(var))

#———

#1

var = "RoofStyle"



#データ型確認

print(df_all[var])



#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———

#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———
#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=90)

plt.title("{} count".format(var))

#———
#6：カテゴリ変数のみ

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

g = sns.boxplot(x=var, y="SalePrice", data=data)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.axis(ymin=0, ymax=800000);

#———

#7：カテゴリ変数のみ

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

g = sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.title("{} & Median Price".format(var))

#———
#1

var = "RoofMatl"



#データ型確認

print(df_all[var])



#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———

#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———

#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=90)

plt.title("{} count".format(var))

#———
#6：カテゴリ変数のみ

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

g = sns.boxplot(x=var, y="SalePrice", data=data)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.axis(ymin=0, ymax=800000);

#———
#7：カテゴリ変数のみ

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

g = sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.title("{} & Median Price".format(var))

#———
#1

var = "Exterior1st"



#データ型確認

print(df_all[var])



#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———

#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———
#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=45)

plt.title("{} count".format(var))

#———
#6：カテゴリ変数のみ

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

g = sns.boxplot(x=var, y="SalePrice", data=data)

g.set_xticklabels(g.get_xticklabels(), rotation=45)

g.axis(ymin=0, ymax=800000);

#———
#7：カテゴリ変数のみ

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

g = sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

g.set_xticklabels(g.get_xticklabels(), rotation=45)

plt.title("{} & Median Price".format(var))

#———
#1

var = "Exterior2nd"



#データ型確認

print(df_all[var])



#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———
#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———

#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=45)

plt.title("{} count".format(var))

#———
#6：カテゴリ変数のみ

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

g = sns.boxplot(x=var, y="SalePrice", data=data)

g.set_xticklabels(g.get_xticklabels(), rotation=45)

g.axis(ymin=0, ymax=800000);

#———
#7：カテゴリ変数のみ

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

g = sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

g.set_xticklabels(g.get_xticklabels(), rotation=45)

plt.title("{} & Median Price".format(var))

#———
#1

var = "MasVnrType"



#データ型確認

print(df_all[var])



#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———

#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———
#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=45)

plt.title("{} count".format(var))

#———

#6：カテゴリ変数のみ

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

g = sns.boxplot(x=var, y="SalePrice", data=data)

g.set_xticklabels(g.get_xticklabels(), rotation=45)

g.axis(ymin=0, ymax=800000);

#———
#7：カテゴリ変数のみ

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

g = sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

g.set_xticklabels(g.get_xticklabels(), rotation=45)

plt.title("{} & Median Price".format(var))

#———
#1

var = "Foundation"



#データ型確認

print(df_all[var])



#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———

#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———
#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=45)

plt.title("{} count".format(var))

#———
#6：カテゴリ変数のみ

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

g = sns.boxplot(x=var, y="SalePrice", data=data)

g.set_xticklabels(g.get_xticklabels(), rotation=45)

g.axis(ymin=0, ymax=800000);

#———
#7：カテゴリ変数のみ

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

g = sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

g.set_xticklabels(g.get_xticklabels(), rotation=45)

plt.title("{} & Median Price".format(var))

#———
#1

var = "ExterQual"



#データ型確認

print(df_all[var])



#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———

#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———
#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=45)

plt.title("{} count".format(var))

#———
#6：カテゴリ変数のみ

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

g = sns.boxplot(x=var, y="SalePrice", data=data)

g.set_xticklabels(g.get_xticklabels(), rotation=45)

g.axis(ymin=0, ymax=800000);

#———
#7：カテゴリ変数のみ

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

g = sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

g.set_xticklabels(g.get_xticklabels(), rotation=45)

plt.title("{} & Median Price".format(var))

#———
#1

var = "ExterCond"



#データ型確認

print(df_all[var])



#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———

#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———
#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=45)

plt.title("{} count".format(var))

#———
#6：カテゴリ変数のみ

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

g = sns.boxplot(x=var, y="SalePrice", data=data)

g.set_xticklabels(g.get_xticklabels(), rotation=45)

g.axis(ymin=0, ymax=800000);

#———
#7：カテゴリ変数のみ

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

g = sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

g.set_xticklabels(g.get_xticklabels(), rotation=45)

plt.title("{} & Median Price".format(var))

#———

#1

var = "OverallQual"



#データ型確認

print(df_all[var])



#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———

#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———
#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=45)

plt.title("{} count".format(var))

#———
#6：カテゴリ変数のみ

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

g = sns.boxplot(x=var, y="SalePrice", data=data)

g.set_xticklabels(g.get_xticklabels(), rotation=45)

g.axis(ymin=0, ymax=800000);

#———
#7：カテゴリ変数のみ

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

g = sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

g.set_xticklabels(g.get_xticklabels(), rotation=45)

plt.title("{} & Median Price".format(var))

#———
#1

var = "OverallCond"



#データ型確認

print(df_all[var])



#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———

#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———

#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=4, aspect=2)

g.set_xticklabels(rotation=45)

plt.title("{} count".format(var))

#———
#6：カテゴリ変数のみ

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

g = sns.boxplot(x=var, y="SalePrice", data=data)

g.set_xticklabels(g.get_xticklabels(), rotation=45)

g.axis(ymin=0, ymax=800000);

#———
#7：カテゴリ変数のみ

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

g = sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

g.set_xticklabels(g.get_xticklabels(), rotation=45)

plt.title("{} & Median Price".format(var))

#———
#1

var = "YearBuilt"



#データ型確認

print(df_all[var])



#相関係数確認（対象：訓練データ）：数値タイプのみ

print("SalePriceとの相関関係係数：{:.4f}".format(correlations[var]))

#-------
#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———
#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———
#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=5.5, aspect=2)

g.set_xticklabels(rotation=90)

plt.title("{} count".format(var))

#———
#5：数値変数のみ

# 価格との散布図

plt.figure(figsize=(8,4))

sns.scatterplot(df_train[var],df_train['SalePrice'])

#———
#6：カテゴリ変数のみ

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(12, 5))

g = sns.boxplot(x=var, y="SalePrice", data=data)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.axis(ymin=0, ymax=800000);

#———
#7：カテゴリ変数のみ

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(12,5))

g = sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.title("{} & Median Price".format(var))

#———

#1

var = "YearRemodAdd"



#データ型確認

print(df_all[var])



#相関係数確認（対象：訓練データ）：数値タイプのみ

print("SalePriceとの相関関係係数：{:.4f}".format(correlations[var]))

#-------
#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———
#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———
#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=5.5, aspect=2)

g.set_xticklabels(rotation=90)

plt.title("{} count".format(var))

#———
#6：カテゴリ変数のみ

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(12, 5))

g = sns.boxplot(x=var, y="SalePrice", data=data)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.axis(ymin=0, ymax=800000);

#———
#7：カテゴリ変数のみ

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(12,5))

g = sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.title("{} & Median Price".format(var))

#———
#1

var = "MasVnrArea"



#データ型確認

print(df_all[var])



#相関係数確認（対象：訓練データ）：数値タイプのみ

print("SalePriceとの相関関係係数：{:.4f}".format(correlations[var]))

#-------
#2

#欠測確認（対象：全データ）

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[[var]]

#———
#3

#統計量確認（対象：全データ）

df_all[var].describe()

#———

#4

#データ数量確認（対象：全データ）

df_all[var].value_counts()

#———
#5：カテゴリ変数のみ

# データ分布確認(対象：全データ)

sns.set()

g = sns.factorplot(x = var, kind='count', data = df_all, height=6, aspect=2)

g.set_xticklabels(rotation=90)

plt.title("{} count".format(var))

#———

#6：カテゴリ変数のみ

#箱ひげ（四分位データ）確認

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 5))

g = sns.boxplot(x=var, y="SalePrice", data=data)

g.set_xticklabels(g.get_xticklabels(), rotation=45)

g.axis(ymin=0, ymax=800000);

#———
#7：カテゴリ変数のみ

#グループ化して、ターゲット(SalePrice)の中央値との関係確認

Grouped = df_train.groupby(var, as_index=True).median()



# 価格とのグラフ

plt.figure(figsize=(10,5))

g = sns.barplot(x=Grouped.index, y='SalePrice', data=Grouped, palette = 'viridis')

g.set_xticklabels(g.get_xticklabels(), rotation=45)

plt.title("{} & Median Price".format(var))

#———